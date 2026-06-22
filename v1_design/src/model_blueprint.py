from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ForwardOutputs:
    logits: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class ParamCountSummary:
    trainable_params: int
    non_trainable_params: int
    total_params: int


def _require_even(value: int, name: str) -> None:
    if value % 2 != 0:
        raise ValueError(f"{name} must be even for RoPE, got {value}")


def create_captioning_mask(num_vis_tokens: int, num_text_tokens: int, device: torch.device) -> torch.Tensor:
    """
    Build a 2D boolean prefix-causal mask for [vision -> text] captioning.

    True = allowed attention, False = masked attention.
    - Vision queries attend all vision keys.
    - Vision queries cannot attend text keys.
    - Text queries attend all vision keys + causal text keys.
    """
    if num_vis_tokens < 0 or num_text_tokens < 0:
        raise ValueError("num_vis_tokens and num_text_tokens must be >= 0")

    total = num_vis_tokens + num_text_tokens
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)

    if num_vis_tokens > 0:
        mask[:num_vis_tokens, :num_vis_tokens] = True

    if num_text_tokens > 0:
        text_rows_start = num_vis_tokens
        mask[text_rows_start:, :num_vis_tokens] = True

        text_idx = torch.arange(num_text_tokens, device=device)
        text_causal = text_idx.unsqueeze(1) >= text_idx.unsqueeze(0)
        mask[text_rows_start:, text_rows_start:] = text_causal

    return mask


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return parameter count for a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def summarize_parameters(module: nn.Module) -> ParamCountSummary:
    """Return trainable/non-trainable/total parameter counts."""
    trainable = count_parameters(module, trainable_only=True)
    total = count_parameters(module, trainable_only=False)
    frozen = total - trainable
    return ParamCountSummary(
        trainable_params=trainable,
        non_trainable_params=frozen,
        total_params=total,
    )


def format_param_count(summary: ParamCountSummary) -> str:
    """Human-readable parameter count report."""
    million = 1_000_000
    lines = [
        "Parameter Count Summary",
        f"- Trainable: {summary.trainable_params:,} ({summary.trainable_params / million:.2f}M)",
        (
            f"- Non-trainable: {summary.non_trainable_params:,} "
            f"({summary.non_trainable_params / million:.2f}M)"
        ),
        f"- Total: {summary.total_params:,} ({summary.total_params / million:.2f}M)",
    ]
    return "\n".join(lines)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, force_fp32: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = torch.float32 if self.force_fp32 else x.dtype
        x_float = x.to(compute_dtype)
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_float * rms
        y = y.to(x.dtype)
        return y * self.weight.to(x.dtype)


class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6, force_fp32: bool = True):
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps=eps, force_fp32=force_fp32)
        self.k_norm = RMSNorm(head_dim, eps=eps, force_fp32=force_fp32)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


def apply_rope_1d(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim], positions: [batch, seq] or [seq]
    head_dim = x.shape[-1]
    _require_even(head_dim, "head_dim")

    if positions.dim() == 1:
        positions = positions.unsqueeze(0)

    device = x.device
    dtype = x.dtype
    half_dim = head_dim // 2

    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    sin, cos = angles.sin(), angles.cos()

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    sin = sin.unsqueeze(1).to(dtype)
    cos = cos.unsqueeze(1).to(dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class VisualProjector(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 384, out_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(visual_tokens)


class TransformerBlockV1(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_multiplier: float = 8.0 / 3.0,
        qk_norm: bool = True,
        force_fp32_norm_ops: bool = True,
        force_fp32_qk_norm_ops: bool = True,
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = embed_dim // n_heads
        self.group_size = n_heads // n_kv_heads

        hidden_dim = int(ffn_multiplier * embed_dim)
        hidden_dim = max(hidden_dim, embed_dim)

        self.attn_norm = RMSNorm(embed_dim, force_fp32=force_fp32_norm_ops)
        self.ffn_norm = RMSNorm(embed_dim, force_fp32=force_fp32_norm_ops)

        self.q_proj = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, embed_dim, bias=False)

        self.qk_norm = QKNorm(self.head_dim, force_fp32=force_fp32_qk_norm_ops) if qk_norm else None

        self.ff_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ff_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ff_down = nn.Linear(hidden_dim, embed_dim, bias=False)

    def _gqa_attention(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if x.dtype == torch.bfloat16:
            raise TypeError("bfloat16 is not supported for this T4-optimized block; use fp16 or fp32")

        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        q = apply_rope_1d(q, positions)
        k = apply_rope_1d(k, positions)

        if self.group_size > 1:
            # Zero-copy GQA head expansion pattern for compiler-friendly fusion.
            k = (
                k.unsqueeze(2)
                .expand(bsz, self.n_kv_heads, self.group_size, seq_len, self.head_dim)
                .reshape(bsz, self.n_heads, seq_len, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(bsz, self.n_kv_heads, self.group_size, seq_len, self.head_dim)
                .reshape(bsz, self.n_heads, seq_len, self.head_dim)
            )

        sdpa_mask: Optional[torch.Tensor] = None
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise ValueError("attn_mask must be boolean with True as allowed attention")
            if attn_mask.dim() == 2:
                if attn_mask.shape != (seq_len, seq_len):
                    raise ValueError("2D attn_mask must have shape [seq_len, seq_len]")
                sdpa_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape != (bsz, seq_len, seq_len):
                    raise ValueError("3D attn_mask must have shape [batch, seq_len, seq_len]")
                sdpa_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:
                sdpa_mask = attn_mask
            else:
                raise ValueError("attn_mask must be 2D, 3D, or 4D boolean tensor")

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != (bsz, seq_len):
                raise ValueError("key_padding_mask must have shape [batch, seq_len]")
            key_keep = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
            sdpa_mask = key_keep if sdpa_mask is None else (sdpa_mask & key_keep)

        # SDPA picks the best available backend on T4 (math/cudnn/cutlass path); no FlashAttention required.
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.o_proj(attn_out)

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_down(F.silu(self.ff_gate(x)) * self.ff_up(x))

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self._gqa_attention(
            self.attn_norm(x),
            positions=positions,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        # SwiGLU form kept explicit so torch.compile can fuse gate/up/down elementwise chain.
        x = x + self._swiglu(self.ffn_norm(x))
        return x


class CaptioningTransformerV1(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16000,
        embed_dim: int = 384,
        n_layers: int = 30,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        qk_norm: bool = True,
        use_gradient_checkpointing: bool = True,
        force_fp32_norm_ops: bool = True,
        force_fp32_qk_norm_ops: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.modality_embedding = nn.Embedding(2, embed_dim)  # 0=image, 1=text
        self.visual_projector = VisualProjector(768, embed_dim, embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlockV1(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    qk_norm=qk_norm,
                    force_fp32_norm_ops=force_fp32_norm_ops,
                    force_fp32_qk_norm_ops=force_fp32_qk_norm_ops,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(embed_dim, force_fp32=force_fp32_norm_ops)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _full_sequence_positions(batch_size: int, total_seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(total_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    def _run_block(
        self,
        block: TransformerBlockV1,
        x: torch.Tensor,
        positions: torch.Tensor,
        captioning_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                lambda hidden, pos, c_mask, k_mask: block(
                    hidden,
                    positions=pos,
                    attn_mask=c_mask,
                    key_padding_mask=k_mask,
                ),
                x,
                positions,
                captioning_mask,
                key_padding_mask,
                use_reentrant=False,
            )
        return block(
            x,
            positions=positions,
            attn_mask=captioning_mask,
            key_padding_mask=key_padding_mask,
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        captioning_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ForwardOutputs:
        if image_tokens.dtype == torch.bfloat16:
            raise TypeError("bfloat16 inputs are disabled for T4 path; provide fp16 or fp32 image tokens")

        image_hidden = self.visual_projector(image_tokens)
        text_hidden = self.token_embedding(text_tokens)

        image_type_ids = torch.zeros(image_hidden.shape[:-1], device=image_hidden.device, dtype=torch.long)
        text_type_ids = torch.ones(text_hidden.shape[:-1], device=text_hidden.device, dtype=torch.long)

        image_hidden = image_hidden + self.modality_embedding(image_type_ids)
        text_hidden = text_hidden + self.modality_embedding(text_type_ids)

        x = torch.cat([image_hidden, text_hidden], dim=1)
        positions = self._full_sequence_positions(x.shape[0], x.shape[1], x.device)
        if captioning_mask is None:
            captioning_mask = create_captioning_mask(
                num_vis_tokens=image_hidden.shape[1],
                num_text_tokens=text_hidden.shape[1],
                device=x.device,
            )

        for block in self.blocks:
            x = self._run_block(
                block,
                x,
                positions,
                captioning_mask=captioning_mask,
                key_padding_mask=attention_mask,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return ForwardOutputs(logits=logits, hidden_states=x)


def build_v1_model() -> CaptioningTransformerV1:
    return CaptioningTransformerV1(vocab_size=16000, embed_dim=384, n_layers=30)
