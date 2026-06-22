# V1 Design Workspace

This folder is a clean area to design and iterate on the v1 captioning model before full implementation.

## Scope
- Base multimodal captioning model only (v1)
- Approximately 60M parameter budget (target around 59.7M)
- Training constraints aligned to 2x T4
- No v2 alignment or preference optimization work

## Folder Layout
- `model_spec.yaml`: Source-of-truth architecture and training knobs
- `train_plan.md`: Milestones and implementation order
- `src/config_schema.py`: Typed config schema and validation helpers
- `src/model_blueprint.py`: Implemented v1 model stack (RMSNorm, RoPE, GQA/QKNorm, SwiGLU)
- `src/param_budget.py`: Parameter estimation utilities
- `src/optim_groups.py`: Muon/AdamW parameter grouping utilities
- `src/train_contract.py`: AMP + GradScaler train-step contract
- `tests/smoke_test_v1.py`: Forward-pass and optimizer-group smoke test
- `overfit/run_overfit_100.py`: 100-example overfit harness with stability checks + image/caption inference report
- `overfit/queries/duckdb_queries.yaml`: Externalized DuckDB SQL templates used by the overfit harness

## Current V1 Decisions Captured
- `vocab_size=16000`, `embed_dim=384`, `n_layers=30`
- Tied token embedding and LM head weights
- Visual projector MLP `768 -> 384 -> 384` with GELU
- GQA + QKNorm attention, RMSNorm, SwiGLU MLP
- 1D RoPE over full image+text sequence
- Modality type embeddings for image/text tokens
- FP16 + GradScaler, gradient checkpointing
- `torch.compile(mode="reduce-overhead")`
- Distributed training via DDP or FSDP
- Hybrid optimizer split (Muon + AdamW)
