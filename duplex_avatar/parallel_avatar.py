"""NVIDIA sequence attention and ordered commands for a native FlashHead GPU group."""
def configure_attention():
    # FlashHead uses neither xFuser KV caching nor joint attention. Calling the
    # underlying yunchang implementation avoids xFuser's unconditional NPU import,
    # which is incompatible with the installed CUDA-only yunchang distribution.
    import torch
    from yunchang import LongContextAttention
    from flash_head.src.modules import flash_head_model

    class Attention(LongContextAttention):
        @torch.compiler.disable
        def forward(self, attn, query, key, value):
            if attn is not None:
                raise ValueError("FlashHead sequence attention does not use external KV state")
            return super().forward(query, key, value)

    flash_head_model.xFuserLongContextAttention = Attention
