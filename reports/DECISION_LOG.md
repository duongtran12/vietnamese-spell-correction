# Decision Log

## 2026-11-01 — Use an official Vietnamese Wikipedia shard for smoke testing

- Decision: use a dated official Wikimedia `pages-articles` shard.
- Reason: clear provenance, document boundaries, reproducible download URL.
- Consequence: the shard is suitable for pipeline validation but too small and
  template-heavy for the final experiment.

## 2026-11-07 — Train all learned components from scratch

- Decision: randomly initialize token embeddings and BiLSTM weights.
- Decision: count all unigram, bigram and trigram statistics from train only.
- Reason: comply with the course requirement and preserve explainability.

## 2026-11-07 — Split by document before synthetic noise generation

- Decision: keep each `document_id` in exactly one split.
- Decision: generate noise independently after splitting.
- Reason: prevent clean or corrupted variants of the same sentence from leaking
  across train, validation and test.

## 2026-11-07 — Remove exact and near duplicates before splitting

- Decision: remove exact duplicates, then near duplicates at threshold 0.95.
- Reason: Wikipedia template families created highly similar sentences that
  could inflate validation results.
- Result: 75 exact and 53 near-duplicate records removed.

## 2026-11-08 — Add conservative inference guards

- Decision: always include keep-original as a candidate.
- Decision: require a minimum rank margin before applying Top-1.
- Decision: normalize case during candidate search and restore it for display.
- Reason: the detector intentionally favors recall and produces many false
  positives; forced correction caused visibly incorrect changes.

## 2026-11-16 — Retrain after near-duplicate removal

- Decision: rebuild vocabulary, Trigram LM and BiLSTM using the deduplicated
  train split.
- Reason: metrics from the earlier split cannot be attributed to the new data
  preparation protocol.
- Test policy: validation only; test remains untouched.

## 2026-11-16 — Select detector threshold and rank margin on validation

- Decision: select detector threshold 0.60 and global rank margin 0.75.
- Constraint: false-correction rate must remain below 2%.
- Selection metric: Correction F1, with safety and sentence accuracy as
  tie-breakers.

## 2026-11-16 — Remove frequency score from the selected ranker

- Decision: set `frequency_weight=0.0`.
- Evidence: Context + Edit reached Accuracy@1 65.93%, compared with 65.19% when
  the explicit frequency score was included.
- End-to-end effect: Correction F1 improved and false correction decreased.

## 2026-11-16 — Retain the interpolated language model

- Decision: keep interpolation lambdas `(0.1, 0.3, 0.6)`.
- Evidence: bigram-only context achieved strong known-position ranking but
  reduced end-to-end Correction F1 to 42.11%.

## 2026-11-16 — Lower diacritic margin only for OOV originals

- Decision: use margin 0.50 for diacritic-only OOV corrections and 0.30 for
  title-case diacritic-only OOV corrections.
- Decision: known train words retain global margin 0.75.
- Reason: preserve useful corrections such as `Hộm → Hôm` without weakening
  protection for valid known words such as `sợi`, `lai` or `Ngay`.

## 2026-11-16 — Keep test split sealed

- Decision: do not run final test while the corpus size, configuration or report
  protocol may still change.
- Next gate: confirm final corpus and design, record the commit, then execute the
  final test exactly once without further tuning.

## 2026-11-17 — Expand the corpus to 2,000 Wikipedia articles

- Decision: use 2,000 main-namespace articles from the official 2026-11-01
  Vietnamese Wikipedia shard
  `pages-articles3.xml-p3452086p4565246.bz2`.
- Reason: the 127-document smoke corpus left validation OOV at 10.61% and was
  too small for reliable correction of unseen sentences.
- Result: 53,953 sentences remain after exact and near-duplicate removal;
  train-only vocabulary grows to 40,127 entries and validation OOV falls to
  3.22%.
- Test policy: the expanded test split was generated but remains unread.

## 2026-11-17 — Retune safe inference on expanded validation

- Decision: set detector threshold to 0.50 and global rank margin to 1.00.
- Evidence: this combination achieved Correction F1 61.21% with a 0.40%
  false-correction rate on expanded validation.
- Selection protocol: maximize Correction F1 subject to false-correction rate
  at or below 2%, using only validation.
