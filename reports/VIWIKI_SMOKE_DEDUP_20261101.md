# Vietnamese Wikipedia Near-Deduplicated Retraining — 2026-11-01 Dump

## Scope

This experiment retrains all learned artifacts after exact and near-duplicate
filtering. It uses validation for model selection and inference tuning. The test
split remains untouched.

## Dataset

- Input sentences: 2,803.
- Exact duplicates removed: 75.
- Near duplicates removed at similarity 0.95: 53.
- Remaining sentences: 2,675.
- Train: 101 documents, 2,013 sentences, 49,845 tokens.
- Validation: 13 documents, 234 sentences, 6,136 tokens.
- Test: 13 documents, 428 sentences, 10,564 tokens.
- Train-only vocabulary size: 5,367.
- Validation OOV rate: 10.61%.

## Candidate generation

- Validation spelling errors: 135.
- Target OOV rate: 14.07%.
- Recall@1: 56.30%.
- Recall@3: 70.37%.
- Recall@5: 77.04%.
- In-vocabulary Recall@5: 89.66%.
- Mean latency before cache warm-up: 17.64 ms.
- p95 latency before cache warm-up: 31.92 ms.

The candidate generator now caches immutable candidate tuples internally and
returns an independent list to callers. This reduces repeated edit-distance
work during inference tuning without changing ranking results.

## Trigram ranker

- Edit-distance baseline Accuracy@1: 56.30%.
- Context ranker Accuracy@1: 65.19%.
- Absolute Accuracy@1 improvement: 8.89 percentage points.
- Context ranker Accuracy@3: 74.07%.

## BiLSTM detector

- Random seed: 42.
- Best epoch: 7.
- Epochs completed: 11 with early stopping.
- Validation ERROR precision at threshold 0.5: 16.81%.
- Validation ERROR recall at threshold 0.5: 87.41%.
- Validation ERROR-F1 at threshold 0.5: 28.20%.

The detector still prioritizes recall and produces many token-level false
positives. The end-to-end keep-original and rank-margin guards prevent most of
these detections from becoming applied false corrections.

## Inference tuning

`scripts/tune_inference.py` evaluated detector thresholds
`[0.5, 0.6, 0.7, 0.8, 0.9]` and rank margins `[0.25, 0.5, 0.75]` on validation.
The primary selection metric was correction F1, subject to a maximum 2% false
correction rate. Ties prefer lower false-correction rate, higher sentence
accuracy, a higher detector threshold, and a higher rank margin.

Selected configuration:

- Detector threshold: 0.60.
- Minimum rank margin: 0.75.
- Title-case diacritic rank margin: 0.50.
- Keep-original bonus: 0.75.

The title-case margin applies only when the leading candidate preserves the
case pattern and differs only in Vietnamese diacritics. This retains safe
corrections such as `Hộm` to `Hôm` without lowering the global margin.

## Tuned end-to-end validation

- Detection ERROR precision: 17.03%.
- Detection ERROR recall: 87.41%.
- Detection ERROR-F1: 28.50%.
- Correction precision: 51.35%.
- Correction recall / Accuracy@1: 42.22%.
- Correction F1: 46.34%.
- Accuracy@3: 62.22%.
- Sentence accuracy: 53.85%.
- False-correction rate: 0.87%.

Compared with the untuned retrained configuration (`threshold=0.5`,
`margin=0.5`), correction F1 increased from 42.38% to 46.34% while the
false-correction rate decreased from 1.18% to 0.87%.

## Local artifacts

Generated model artifacts remain local and are excluded from Git:

- `artifacts/viwiki-20261101-smoke-dedup-detector.pt`.
- `artifacts/viwiki-20261101-smoke-dedup-trigram.json`.
- `data/processed/viwiki-20261101-smoke-dedup/vocabulary.json`.

Tracked JSON reports contain training history, validation metrics, candidate
coverage, ranker results, inference tuning results, and end-to-end metrics.

## Remaining limitations and next step

- The corpus is still a small Wikipedia shard with a 14.07% target OOV rate.
- Synthetic errors do not fully represent real user mistakes.
- Detector precision remains low because correct tokens greatly outnumber error
  tokens and the dataset is small.
- The next experiment should expand to additional approved Wikipedia shards,
  repeat deduplication and retraining, then freeze the design before evaluating
  the untouched test split once.
