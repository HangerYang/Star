# coding=utf-8
"""LEAP Llama decoder with zone-aware, whole-layer speculative actions.

This is SWIFT's `model/swift/modeling_llama.py` re-purposed for **LEAP**. SWIFT drafts by
skipping *attention* and *MLP* sub-layers independently (two global skip-sets). LEAP
instead assigns each *whole* transformer layer one of three actions and drafts with that
per-layer configuration:

    * ``LayerAction.EXECUTE`` – run the layer normally.
    * ``LayerAction.SKIP``    – skip the whole layer (identity/residual pass-through, the
                                 layer contributes nothing and writes no KV).
    * ``LayerAction.REPEAT``  – apply the layer twice to deepen the draft (the second pass
                                 overwrites, rather than appends to, the KV cache so cache
                                 lengths stay consistent across layers).

The action map is a ``Dict[int, str]`` set via ``set_layer_config`` and is only applied
while drafting (inside the ``self_draft()`` context, i.e. ``enabled_draft=True``). During
verification the full model runs (every layer EXECUTE), exactly like SWIFT. All the KV-cache
and tree-attention plumbing is unchanged, so LEAP reuses SWIFT's tree-decoding utilities.

Runs in the ``swift`` conda env (transformers 4.37).
"""
import math
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaMLP as _LlamaMLP
from transformers.models.llama.modeling_llama import LlamaAttention as _LlamaAttention
from transformers.models.llama.modeling_llama import LlamaModel as _LlamaModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM as _LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, \
    LlamaDynamicNTKScalingRotaryEmbedding

logger = logging.get_logger(__name__)

from contextlib import contextmanager


class LayerAction:
    """Whole-layer actions explored by LEAP's zone-aware MCTS."""
    EXECUTE = "EXECUTE"
    SKIP = "SKIP"
    REPEAT = "REPEAT"


enabled_draft = False
# LEAP per-layer action map: {physical_layer_idx: LayerAction}. Missing -> EXECUTE.
_layer_config: Dict[int, str] = {}

print('(Re-)Loading modeling (LEAP)...')


def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaAttention(_LlamaAttention):

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, base=getattr(self.config, 'rope_theta', 10000),
                                                   max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, base=getattr(self.config, 'rope_theta', 10000),
                    max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, base=getattr(self.config, 'rope_theta', 10000),
                    max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            repeat_overwrite: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        key_states = self.k_proj(hidden_states)
        query_states = self.q_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # On a REPEAT second pass, rewind the cache so the recomputed K/V for the
            # current tokens overwrite (rather than duplicate) the first pass.
            if repeat_overwrite:
                past_key_value[0].current_length.sub_(q_len)
                past_key_value[1].current_length.sub_(q_len)
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaMLP(_LlamaMLP):
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_id = layer_id

    def _execute_once(self, hidden_states, attention_mask, position_ids, past_key_value,
                      output_attentions, use_cache, repeat_overwrite=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            repeat_overwrite=repeat_overwrite,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, self_attn_weights, present_key_value

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        action = _layer_config.get(self.layer_id, LayerAction.EXECUTE) if enabled_draft else LayerAction.EXECUTE

        if action == LayerAction.SKIP:
            # Whole-layer skip: identity pass-through, no KV written for this layer.
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (None,)
            return outputs

        # EXECUTE (once) or REPEAT (twice, 2nd pass overwrites cache).
        hidden_states, self_attn_weights, present_key_value = self._execute_once(
            hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache, repeat_overwrite=False,
        )
        if action == LayerAction.REPEAT:
            hidden_states, self_attn_weights, present_key_value = self._execute_once(
                hidden_states, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache, repeat_overwrite=True,
            )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class LlamaModel(_LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(_LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = True
        self.post_init()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, torch.float32, device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        if hasattr(self, "swift_mask") and self.swift_mask is not None and not enabled_draft:
            swift_mask = self.swift_mask
            swift_len = swift_mask.size(-1)
            combined_attention_mask[:, :, -swift_len:, -swift_len:][swift_mask == 0] = combined_attention_mask.min()
        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            for past_key_value in past_key_values:
                if past_key_value is not None:
                    past_key_values_length = past_key_value[0].shape[2]
                    break
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(_LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(_LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @contextmanager
    def self_draft(self, enabled=True, *args, **kwds):
        global enabled_draft
        enabled_draft = enabled
        try:
            yield None
        finally:
            enabled_draft = False

    def set_layer_config(self, layer_config: Dict[int, str]):
        """Set the LEAP per-layer action map used while drafting."""
        global _layer_config
        _layer_config = dict(layer_config) if layer_config is not None else {}

    def get_layer_config(self) -> Dict[int, str]:
        return dict(_layer_config)

    def get_skip_layers(self):
        """Compatibility shim for the shared `eval.py` summary. Reports the current
        LEAP config as (skipped_layers, repeated_layers)."""
        skip = [k for k, v in _layer_config.items() if v == LayerAction.SKIP]
        repeat = [k for k, v in _layer_config.items() if v == LayerAction.REPEAT]
        return skip, repeat

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
