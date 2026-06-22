from dataclasses import dataclass


@dataclass
class BudgetReport:
    embedding_params: int
    visual_projector_params: int
    norm_params: int
    overhead_params: int
    per_layer_params: int
    layer_total_params: int
    grand_total_params: int


def estimate_v1_params(vocab_size: int = 16000, embed_dim: int = 384, n_layers: int = 30) -> BudgetReport:
    embedding_params = vocab_size * embed_dim

    # MLP projector: 768->384 and 384->384 with biases.
    visual_projector_params = (768 * 384 + 384) + (384 * 384 + 384)

    # Rough RMSNorm estimate: 2 norms per layer + final norm.
    norm_params = (2 * n_layers * embed_dim) + embed_dim

    overhead_params = embedding_params + visual_projector_params + norm_params

    # Approximation from design notes: attention + SwiGLU per layer.
    per_layer_params = 1_770_000
    layer_total_params = per_layer_params * n_layers
    grand_total_params = overhead_params + layer_total_params

    return BudgetReport(
        embedding_params=embedding_params,
        visual_projector_params=visual_projector_params,
        norm_params=norm_params,
        overhead_params=overhead_params,
        per_layer_params=per_layer_params,
        layer_total_params=layer_total_params,
        grand_total_params=grand_total_params,
    )


def format_report(report: BudgetReport) -> str:
    million = 1_000_000
    lines = [
        "V1 Parameter Budget Report",
        f"- Embeddings: {report.embedding_params:,} ({report.embedding_params / million:.2f}M)",
        f"- Visual projector: {report.visual_projector_params:,} ({report.visual_projector_params / million:.2f}M)",
        f"- Norms: {report.norm_params:,} ({report.norm_params / million:.2f}M)",
        f"- Overhead: {report.overhead_params:,} ({report.overhead_params / million:.2f}M)",
        f"- Per layer approx: {report.per_layer_params:,} ({report.per_layer_params / million:.2f}M)",
        f"- Layers total: {report.layer_total_params:,} ({report.layer_total_params / million:.2f}M)",
        f"- Grand total: {report.grand_total_params:,} ({report.grand_total_params / million:.2f}M)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(estimate_v1_params()))
