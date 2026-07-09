"""SmolVLM-256M (Idefics3) adapted for SWIFT self-speculative decoding.

SWIFT accelerates an LLM by skipping intermediate transformer layers to draft tokens,
then verifying them with the full model — **no auxiliary model, no training**. Because
SmolVLM's text backbone is a plain Llama decoder, the SWIFT algorithm transfers to it
unchanged; the only new work is on the **vision side**:

  * encode the image with SmolVLM's SigLIP vision tower + pixel-shuffle connector, and
  * fuse the resulting image embeddings into the token-embedding stream at the
    `<image>` placeholder positions, exactly once, during the SWIFT *prefill*.

Everything downstream (drafting, tree verification, layer-set Bayesian optimization)
runs on text tokens against a KV cache that already holds the fused image context, so
the language-model / decoding side is untouched.

Design
------
* ``self.model``  -> ``SmolVLMSwiftTextModel`` (subclass of the vendored SWIFT Llama
  decoder in ``modeling_llama_compat``). All of ``model/swift/utils.py`` calls
  ``model.model(input_ids=...)`` / ``model.lm_head(...)`` and works verbatim.
* ``self.vision_model`` / ``self.connector`` -> SmolVLM's stock HF modules.
* ``set_image(...)`` encodes the image once and stashes the features; the text model
  merges them into ``inputs_embeds`` on the first (prefill) forward and then clears the
  stash so speculative steps stay text-only.
"""
from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoConfig, Idefics3ForConditionalGeneration

from . import modeling_llama_compat as _c
from .modeling_llama_compat import LlamaForCausalLM as _SwiftLlamaForCausalLM
from .modeling_llama_compat import LlamaModel as _SwiftLlamaModel


class SmolVLMSwiftTextModel(_SwiftLlamaModel):
    """SWIFT Llama decoder that fuses cached image features at the prefill step."""

    def __init__(self, config):
        super().__init__(config)
        self._pending_image_hidden_states = None
        self._image_token_id = None

    def set_pending_image(self, image_hidden_states, image_token_id):
        self._pending_image_hidden_states = image_hidden_states
        self._image_token_id = image_token_id

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        # Fuse image features only on the prefill (input_ids present, embeds not yet
        # built, and an image is pending). Consume once so drafting stays text-only.
        if (
            self._pending_image_hidden_states is not None
            and input_ids is not None
            and inputs_embeds is None
        ):
            embeds = self.embed_tokens(input_ids)
            img = self._pending_image_hidden_states.to(dtype=embeds.dtype, device=embeds.device)
            img = img.reshape(-1, embeds.size(-1))
            mask = input_ids == self._image_token_id
            n_img_tok = int(mask.sum().item())
            if n_img_tok != img.size(0):
                raise ValueError(
                    f"#<image> placeholders ({n_img_tok}) != #image embeddings ({img.size(0)}). "
                    "Check the processor / prompt image-token expansion."
                )
            embeds = embeds.clone()
            embeds[mask] = img
            self._pending_image_hidden_states = None
            return super().forward(input_ids=None, inputs_embeds=embeds, **kwargs)
        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)


class SmolVLMSwiftForCausalLM(_SwiftLlamaForCausalLM):
    """SmolVLM/Idefics3 wired for SWIFT. Language side reuses the vendored SWIFT
    decoder verbatim; the vision tower + connector are the stock SmolVLM modules."""

    def __init__(self, text_config, vlm_config):
        super().__init__(text_config)
        # Swap the plain SWIFT decoder for the image-fusing one.
        self.model = SmolVLMSwiftTextModel(text_config)
        self.vlm_config = vlm_config
        self.image_token_id = getattr(vlm_config, "image_token_id", None)
        self.patch_size = vlm_config.vision_config.patch_size
        # Vision modules are attached in ``from_pretrained``.
        self.vision_model = None
        self.connector = None

    # ── vision encoding (mirrors Idefics3Model.get_image_features) ──────────────
    @torch.inference_mode()
    def get_image_features(self, pixel_values, pixel_attention_mask=None):
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype, device=self.device)
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Drop fully-padded images.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=torch.bool, device=pixel_values.device,
            )
        else:
            pixel_attention_mask = pixel_attention_mask.view(
                batch_size * num_images, *pixel_attention_mask.shape[2:]
            )
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        image_hidden_states = self.vision_model(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask,
        ).last_hidden_state
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states

    def set_image(self, pixel_values, pixel_attention_mask=None):
        """Encode the image once and hand the features to the text model for prefill
        fusion. Call right before ``swift_forward`` for each example."""
        if pixel_values is None:
            self.model.set_pending_image(None, self.image_token_id)
            return
        image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
        self.model.set_pending_image(image_hidden_states, self.image_token_id)

    # ── loading ────────────────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=torch.float16, device_map="auto", **kwargs):
        vlm_config = AutoConfig.from_pretrained(model_path)
        text_config = vlm_config.text_config
        # SWIFT reads these off ``model.config``; make sure they exist.
        if not hasattr(text_config, "pretraining_tp"):
            text_config.pretraining_tp = 1
        if getattr(text_config, "pad_token_id", None) is None:
            text_config.pad_token_id = getattr(vlm_config, "pad_token_id", None)

        hf = Idefics3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )

        model = cls(text_config, vlm_config)
        model = model.to(torch_dtype)

        # Transplant weights: HF param names match the vendored SWIFT decoder.
        missing, unexpected = model.model.load_state_dict(hf.model.text_model.state_dict(), strict=False)
        if missing:
            raise RuntimeError(f"Missing text-model weights: {missing[:8]} ...")
        model.lm_head.load_state_dict(hf.lm_head.state_dict())
        model.vision_model = hf.model.vision_model
        model.connector = hf.model.connector

        del hf
        if torch.cuda.is_available():
            model = model.cuda()
            torch.cuda.empty_cache()
        model.eval()
        return model
