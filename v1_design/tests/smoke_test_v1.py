import os
import sys

import torch

# Allow running as a standalone script from repository root.
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model_blueprint import build_v1_model, format_param_count, summarize_parameters
from optim_groups import build_optimizer_groups, summarize_groups


def run_smoke_test() -> None:
    torch.manual_seed(7)

    model = build_v1_model()
    model.eval()

    batch_size = 2
    image_seq_len = 32
    text_seq_len = 24

    image_tokens = torch.randn(batch_size, image_seq_len, 768)
    text_tokens = torch.randint(low=0, high=16000, size=(batch_size, text_seq_len))

    total_seq = image_seq_len + text_seq_len
    attention_mask = torch.ones(batch_size, total_seq, dtype=torch.long)

    with torch.no_grad():
        outputs = model(image_tokens=image_tokens, text_tokens=text_tokens, attention_mask=attention_mask)

    assert outputs.logits.shape == (batch_size, total_seq, 16000)
    assert outputs.hidden_states.shape == (batch_size, total_seq, 384)

    # Verify tied embeddings remain tied.
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()

    groups = build_optimizer_groups(model)
    assert len(groups.muon) > 0
    assert len(groups.adamw) > 0
    param_summary = summarize_parameters(model)

    print("SMOKE TEST: PASS")
    print(f"logits shape: {tuple(outputs.logits.shape)}")
    print(f"hidden shape: {tuple(outputs.hidden_states.shape)}")
    print(format_param_count(param_summary))
    print(summarize_groups(groups))


if __name__ == "__main__":
    run_smoke_test()
