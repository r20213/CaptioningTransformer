from dataclasses import dataclass
from typing import Optional

import torch

from model_blueprint import CaptioningTransformerV1


@dataclass
class TrainStepConfig:
    use_amp_fp16: bool = True
    grad_clip_norm: Optional[float] = 1.0


def train_step(
    model: CaptioningTransformerV1,
    batch_image_tokens: torch.Tensor,
    batch_text_tokens: torch.Tensor,
    batch_attention_mask: Optional[torch.Tensor],
    target_token_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: TrainStepConfig,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    amp_enabled = cfg.use_amp_fp16 and torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        outputs = model(
            image_tokens=batch_image_tokens,
            text_tokens=batch_text_tokens,
            attention_mask=batch_attention_mask,
        )
        logits = outputs.logits[:, -target_token_ids.shape[1] :, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_token_ids.reshape(-1),
            reduction="mean",
        )

    scaler.scale(loss).backward()

    if cfg.grad_clip_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

    scaler.step(optimizer)
    scaler.update()

    return float(loss.detach().item())
