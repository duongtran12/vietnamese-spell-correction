# Vietnamese Wikipedia Smoke Experiment — 2026-11-01 Dump

## Purpose

This experiment validates the complete project pipeline on a small, real Vietnamese Wikipedia shard. It is a development baseline, not the final reported test result.

## Provenance

- Source: official Wikimedia Vietnamese Wikipedia dump.
- Dump date: `20261101`.
- File: `viwiki-20261101-pages-articles4.xml-p4565247p6065246.bz2`.
- Compressed size: 377,498 bytes.
- SHA-256: `dd3c5ff2502607839c2864fd62e562135089e32b437be663f2a533983335dcf8`.
- Raw data and trained artifacts remain local and are excluded from Git.

## Conversion and audit

- Pages scanned: 596.
- Main-namespace non-redirect pages processed: 127.
- Sentences extracted: 2,803.
- Tokens: 68,106.
- Exact duplicate records: 75.
- Near-duplicate pairs reported at similarity 0.9: 545.
- Sentences with over 10% unusual characters: 23.

The shard contains many similarly structured articles. Its near-duplicate rate is too high for final experiments without additional filtering.

## Prepared splits

- Train: 101 documents, 2,051 sentences, 50,571 tokens.
- Validation: 13 documents, 234 sentences, 6,136 tokens.
- Test: 13 documents, 443 sentences, 10,679 tokens.
- Train-only vocabulary size: 5,371.
- Validation OOV rate: 10.59%.
- Test OOV rate: 9.03%.

The test split has not been evaluated. All metrics below use validation only.

## Candidate baseline

- Validation error examples: 135.
- Target OOV rate: 14.07%.
- Recall@1: 57.04%.
- Recall@3: 70.37%.
- Recall@5: 77.78%.
- In-vocabulary Recall@5: 90.52%.
- Mean candidate latency: 17.96 ms.
- p95 candidate latency: 32.58 ms.

## Trigram ranker

- Edit-distance baseline Accuracy@1: 57.04%.
- Context ranker Accuracy@1: 65.19%.
- Absolute Accuracy@1 improvement: 8.15 percentage points.
- Context ranker Accuracy@3: 74.07%.

The current interpolation and ranking weights are initial values and have not been tuned.

## BiLSTM detector

- Best epoch: 5.
- Training stopped after epoch 9 through early stopping.
- ERROR precision: 17.79%.
- ERROR recall: 85.93%.
- ERROR F1: 29.48%.

The detector prioritizes recall and produces many false positives at threshold 0.5. Threshold selection must be performed on validation before the design is locked.

## End-to-end validation

- Correction precision: 17.38%.
- Correction recall / Accuracy@1: 57.04%.
- Correction F1: 26.64%.
- Accuracy@3: 64.44%.
- Sentence accuracy: 20.09%.
- False-correction rate: 5.65%.

## Decisions before the final experiment

1. Use a broader collection of Wikipedia shards to reduce topic and template concentration.
2. Remove or group near-duplicate documents before splitting.
3. Tune detector threshold and rank margin on validation.
4. Tune Trigram interpolation and ranker weights on validation.
5. Freeze the configuration before evaluating the untouched test split.

