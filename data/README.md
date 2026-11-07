# Data

Store datasets locally in `raw`, `interim`, and `processed`. Dataset files are ignored by Git to avoid publishing restricted or large data.

Every dataset used in this project must have documented provenance, usage rights, preprocessing steps, and split strategy. Clean documents must be split before synthetic errors are generated.

## Clean input schema

The preparation command accepts UTF-8 JSONL with one sentence per line:

```json
{"document_id":"doc-001","sentence_id":"doc-001-s001","text":"Tôi đang học xử lý ngôn ngữ tự nhiên.","source":"corpus-name"}
```

- `document_id`: required identifier used for leakage-safe splitting.
- `sentence_id`: required unique sentence identifier.
- `text`: required clean Vietnamese sentence.
- `source`: optional corpus or collection identifier.

Keep source files in `data/raw`. They are ignored by Git. Record the corpus URL, license or permission, download date, and preprocessing decisions in a separate data card before using the corpus for experiments.

## Prepare data

From the repository root:

```bash
python -m scripts.prepare_data --input data/raw/corpus.jsonl
```

The command writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, `vocabulary.json`, and `statistics.json` to `data/processed`. Vocabulary and frequency data are fitted only on the generated training split.

## Audit before preparation

Document the selected corpus in `DATA_CARD.md`, then run:

```bash
python -m scripts.audit_data \
  --input data/raw/corpus.jsonl \
  --output reports/data_audit.json
```

Review Unicode normalization, duplicate IDs, exact duplicates, near-duplicate examples, sentence lengths, URLs, emails, and unusual-character ratios before generating final splits.

