def _patch_transformers_tokenizers() -> None:
    try:
        from transformers import GPT2Tokenizer, GPT2TokenizerFast
    except Exception:
        return

    def _all_special_tokens_extended(self):
        return list(self.all_special_tokens)

    for cls in (GPT2Tokenizer, GPT2TokenizerFast):
        if not hasattr(cls, "all_special_tokens_extended"):
            cls.all_special_tokens_extended = property(_all_special_tokens_extended)


_patch_transformers_tokenizers()
