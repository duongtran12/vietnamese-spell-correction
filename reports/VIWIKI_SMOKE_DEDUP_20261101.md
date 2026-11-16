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
- Context ranker Accuracy@1: 65.93%.
- Absolute Accuracy@1 improvement: 9.63 percentage points.
- Context ranker Accuracy@3: 74.07%.

The selected ranker uses context and edit scores with frequency weight zero.

## Ablation study

All variants use the same 135 validation error examples and candidate pool:

- Edit-distance baseline: Accuracy@1 56.30%, Accuracy@3 70.37%.
- Edit + frequency without context: Accuracy@1 54.81%, Accuracy@3 71.85%.
- Interpolated context + edit without frequency: Accuracy@1 65.93%, Accuracy@3 74.07%.
- Full interpolated ranker with frequency: Accuracy@1 65.19%, Accuracy@3 74.07%.
- Unigram context: Accuracy@1 48.89%, Accuracy@3 70.37%.
- Bigram context: Accuracy@1 65.93%, Accuracy@3 75.56%.
- Trigram context: Accuracy@1 62.22%, Accuracy@3 74.07%.

Frequency did not improve Top-1 ranking on this validation split. Removing it
also improved end-to-end Correction F1 from 46.34% to 47.58% and reduced the
false-correction rate from 0.87% to 0.60%. Bigram-only context ranked known
error positions well but reduced end-to-end Correction F1 to 41.95%, so the
interpolated language model remains selected.

Full selected-ranker Accuracy@1 by synthetic error type:

- Delete: 54.29%.
- Diacritic: 76.92%.
- Insert: 70.00%.
- Substitute: 52.38%.
- Transpose: 80.77%.

Generated figures:

- `reports/figures/viwiki-20261101-smoke-dedup-ranker-ablation.png`.
- `reports/figures/viwiki-20261101-smoke-dedup-error-types.png`.

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
`[0.5, 0.6, 0.7, 0.8, 0.9]` and rank margins
`[0.25, 0.5, 0.75, 1.0]` on validation.
The primary selection metric was correction F1, subject to a maximum 2% false
correction rate. Ties prefer lower false-correction rate, higher sentence
accuracy, a higher detector threshold, and a higher rank margin.

Selected configuration:

- Detector threshold: 0.60.
- Minimum rank margin: 0.75.
- Diacritic-only rank margin: 0.50.
- Title-case diacritic rank margin: 0.30.
- Keep-original bonus: 0.75.
- Frequency weight: 0.00.

The title-case margin applies only when the leading candidate preserves the
case pattern and differs only in Vietnamese diacritics. This retains safe
corrections such as `Hộm` to `Hôm` without lowering the global margin.

## Tuned end-to-end validation

- Detection ERROR precision: 17.03%.
- Detection ERROR recall: 87.41%.
- Detection ERROR-F1: 28.50%.
- Correction precision: 58.70%.
- Correction recall / Accuracy@1: 40.00%.
- Correction F1: 47.58%.
- Accuracy@3: 64.44%.
- Sentence accuracy: 55.98%.
- False-correction rate: 0.60%.

Compared with the previous selected configuration with frequency scoring,
correction F1 increased from 46.34% to 47.58% while the false-correction rate
decreased from 0.87% to 0.60%.

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
