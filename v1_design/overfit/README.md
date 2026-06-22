# Overfit Test (100 Examples)

This folder contains an end-to-end overfit harness for v1.

## What it does
- Loads exactly 100 train examples from the encoded patch-token dataset.
- Trains v1 on this micro-set to verify learning and memorization behavior.
- Runs inference and writes a report with:
  - generated caption
  - actual caption
  - actual image (resolved from original dataset via DuckDB query)
- Performs numerical stability checks during training.

## Main script
- `run_overfit_100.py`

## DuckDB SQL templates
- `queries/duckdb_queries.yaml`

## Typical run
```powershell
python v1_design/overfit/run_overfit_100.py \
  --encoded-dataset-id LastTransformer/BLIP3o-Encoded-Features \
  --original-dataset-id BLIP3o/BLIP3o-Pretrain-Long-Caption \
  --tokenizer-repo-id LastTransformer/BLIP3o-SentencePiece-Unigram-Tokenizer \
  --steps 500 \
  --batch-size 2 \
  --num-infer 12
```

## Run with explicit parquet URL JSON
If you already have parquet URL JSON (for example `{"default": {"train": ["..."]}}`), pass it directly:

```powershell
python v1_design/overfit/run_overfit_100.py \
  --encoded-dataset-id LastTransformer/BLIP3o-Encoded-Features \
  --original-dataset-id BLIP3o/BLIP3o-Pretrain-Long-Caption \
  --tokenizer-repo-id LastTransformer/BLIP3o-SentencePiece-Unigram-Tokenizer \
  --original-parquet-urls-json "{\"default\":{\"train\":[\"https://huggingface.co/api/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption/parquet/default/train/0.parquet\"]}}"
```

## Outputs
- `v1_design/overfit/artifacts/loss_curve.csv`
- `v1_design/overfit/artifacts/inference_predictions.parquet`
- `v1_design/overfit/artifacts/image_metadata.parquet`
- `v1_design/overfit/artifacts/inference_joined.parquet`
- `v1_design/overfit/artifacts/inference_report.md`
- `v1_design/overfit/artifacts/inference_report.html`

Open the HTML report in a browser to visually inspect captions against images.
