# Expanded Vietnamese Wikipedia Validation Experiment

## Scope

This experiment replaces the 127-document smoke corpus with 2,000
main-namespace, non-redirect articles from the official Vietnamese Wikipedia
dump dated 2026-11-01. It remains a validation experiment: the expanded test
split was generated but was not read by training, tuning or evaluation.

## Source and conversion

- Dump shard: `viwiki-20261101-pages-articles3.xml-p3452086p4565246.bz2`.
- Compressed size: 38,103,742 bytes (36.34 MB).
- SHA-256: `fe0cb61aa00ac4b2a28293dd97f721f2e776f4a282ee7893d986130955414f86`.
- Pages scanned: 13,332.
- Main-namespace pages retained: 2,000.
- Clean sentences written: 56,428.
- Clean tokens before deduplication: 1,400,404.

The converter used Unicode NFC normalization, transparent punctuation-based
sentence splitting, a 5–80 token length filter and a minimum letter ratio of
0.60. Full machine-readable provenance is in
`viwiki-20261101-expanded-2000-provenance.json`.

## Quality audit and split

- Exact duplicate records removed: 1,755.
- Near-duplicate records removed at threshold 0.95: 720.
- Remaining sentences: 53,953.
- Near-duplicate comparisons: 113,683; comparison limit was not exhausted.
- Train: 1,588 documents, 43,312 sentences, 1,085,839 tokens.
- Validation: 199 documents, 4,779 sentences, 124,790 tokens.
- Sealed test: 198 documents, 5,862 sentences, 155,041 tokens.
- Train-only vocabulary: 40,127 entries.
- Validation OOV rate: 3.22%.

Documents were split before synthetic noise generation. Vocabulary, candidate
dictionary and n-gram counts were fit on train only.

## Detector training

The BiLSTM was randomly initialized and trained from scratch on CPU with the
same declared architecture and optimizer settings as the smoke experiment.
Early stopping completed eight epochs and selected epoch 4.

- Validation ERROR precision: 40.21%.
- Validation ERROR recall: 82.78%.
- Validation ERROR-F1: 54.13%.

The smoke detector ERROR-F1 was 28.50%, so corpus expansion produced a 25.63
percentage-point absolute improvement.

## Candidate generator and ranker

Candidate generator over 2,898 validation spelling errors:

- Recall@1: 57.63%.
- Recall@3: 77.36%.
- Recall@5: 83.06%.
- Candidate dictionary size: 37,528.

Ranker results:

- Edit-distance baseline Accuracy@1: 57.63%.
- Context + Edit Accuracy@1: 75.12%.
- Context + Edit Accuracy@3: 81.61%.
- Context improves Top-1 by 17.49 percentage points.

## End-to-end validation

A grid over detector thresholds `(0.5, 0.6, 0.7, 0.8, 0.9)` and minimum rank
margins `(0.25, 0.50, 0.75, 1.00)` was evaluated on validation. Selection
maximized Correction F1 subject to false-correction rate at or below 2%.

Selected configuration:

- Detector threshold: 0.50.
- Minimum rank margin: 1.00.
- Correction precision: 69.88%.
- Correction recall: 54.45%.
- Correction F1: 61.21%.
- Accuracy@3: 66.49%.
- Sentence Accuracy: 65.24%.
- False Correction Rate: 0.40%.

Compared with the deduplicated smoke experiment, Correction F1 improves from
47.79% to 61.21%, Sentence Accuracy from 55.56% to 65.24%, and false correction
decreases from 0.58% to 0.40%.

## Reproducibility artifacts

- `viwiki-20261101-expanded-2000-provenance.json`.
- `viwiki-20261101-expanded-2000-audit.json`.
- `viwiki-20261101-expanded-2000-statistics.json`.
- `viwiki-20261101-expanded-2000-detector-training.json`.
- `viwiki-20261101-expanded-2000-candidates.json`.
- `viwiki-20261101-expanded-2000-ranker.json`.
- `viwiki-20261101-expanded-2000-inference-tuning.json`.

Raw data, processed splits, n-gram artifact and detector checkpoint remain local
and are excluded from Git.
