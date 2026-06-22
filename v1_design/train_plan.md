# V1 Training and Build Plan

## Milestone 1 - Config and Budget Lock
- Keep `model_spec.yaml` as the source of truth.
- Validate parameter budget is within 58M-60M with `src/param_budget.py`.
- Lock shape conventions for multimodal sequence assembly.

## Milestone 2 - Model Skeleton
- [x] Implement token embedding, modality embeddings, and visual projector.
- [x] Implement transformer block with:
  - RMSNorm
  - GQA + QKNorm attention
  - SwiGLU MLP
- [x] Add tied LM head wiring.

## Milestone 3 - Forward Contract
- [x] Define `forward(image_tokens, text_tokens, attention_mask)` output contract.
- [x] Apply single monotonic RoPE index over combined visual+text sequence.
- [x] Add shape assertions and basic smoke tests.

## Milestone 4 - Training Loop Contract
- Add AMP FP16 + GradScaler.
- Add gradient checkpointing hooks.
- Add `torch.compile(mode="reduce-overhead")` gating.
- Prepare DDP/FSDP launch interface.

## Milestone 5 - Optimizer Split
- [x] Create parameter grouping utility:
  - Muon for dense matrices with rank >= 2
  - AdamW for embeddings, norms, biases, visual projector
- [x] Add logging to print parameter counts by group.

## Milestone 6 - Validation Checklist
- Confirm parameter count and memory profile for 2x T4.
- Confirm training step stability (no underflow in norm/qk path).
- Confirm tied weights remain tied after compile/distribution wrappers.
