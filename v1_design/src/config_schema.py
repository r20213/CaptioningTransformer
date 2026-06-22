from dataclasses import dataclass, field
from typing import List


@dataclass
class ProjectorSpec:
    layers: List[int] = field(default_factory=lambda: [768, 384, 384])
    activation: str = "gelu"


@dataclass
class VisualSpec:
    encoder_dim: int = 768
    projector: ProjectorSpec = field(default_factory=ProjectorSpec)
    modality_type_embeddings: bool = True


@dataclass
class ModelSpec:
    vocab_size: int = 16000
    embed_dim: int = 384
    n_layers: int = 30
    tie_word_embeddings: bool = True
    rope_type: str = "rope_1d"
    rope_sequence_layout: str = "full_multimodal"
    attention_type: str = "gqa"
    qk_norm: bool = True
    norm_type: str = "rmsnorm"
    force_fp32_norm_ops: bool = True
    force_fp32_qk_norm_ops: bool = True
    ffn_type: str = "swiglu"
    visual: VisualSpec = field(default_factory=VisualSpec)


@dataclass
class TrainingSpec:
    precision: str = "fp16"
    use_grad_scaler: bool = True
    gradient_checkpointing: bool = True
    compile_enabled: bool = True
    compile_mode: str = "reduce-overhead"
    distributed_strategy: str = "ddp_or_fsdp"


@dataclass
class OptimizerSpec:
    muon_targets: str = "dense_weight_matrices_rank_ge_2"
    adamw_targets: List[str] = field(
        default_factory=lambda: ["embeddings", "norms", "biases", "visual_projection"]
    )


@dataclass
class BudgetSpec:
    target_total_params_millions: float = 60.0
    expected_total_params_millions: float = 59.7


@dataclass
class V1Spec:
    version: str = "v1"
    name: str = "captioning_transformer_v1"
    model: ModelSpec = field(default_factory=ModelSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    budget: BudgetSpec = field(default_factory=BudgetSpec)


def validate_spec(spec: V1Spec) -> None:
    if spec.model.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if spec.model.embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    if spec.model.n_layers <= 0:
        raise ValueError("n_layers must be positive")

    layers = spec.model.visual.projector.layers
    if len(layers) < 2:
        raise ValueError("visual projector must have at least input and output dims")
    if layers[0] != spec.model.visual.encoder_dim:
        raise ValueError("projector input dim must match visual encoder dim")
    if layers[-1] != spec.model.embed_dim:
        raise ValueError("projector output dim must match embed_dim")
