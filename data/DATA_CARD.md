# Data Card

## Status

A small official Vietnamese Wikipedia shard from the `20261101` dump has been downloaded and used for a smoke experiment. The full production corpus and final protocol still require instructor approval. Raw text and trained artifacts remain local and are excluded from Git.

Smoke shard:

- File: `viwiki-20261101-pages-articles4.xml-p4565247p6065246.bz2`.
- Size: 377,498 bytes.
- SHA-256: `dd3c5ff2502607839c2864fd62e562135089e32b437be663f2a533983335dcf8`.
- Source URL: https://dumps.wikimedia.org/viwiki/20261101/viwiki-20261101-pages-articles4.xml-p4565247p6065246.bz2
- Result summary: `reports/VIWIKI_SMOKE_20261101.md`.

## Proposed primary clean corpus

### Vietnamese Wikipedia

- Role: clean text used to create document-level train, validation, and test splits.
- Official project identifier: `viwiki`.
- Preferred source: the official Wikimedia `pages-articles` dump or an official Wikimedia snapshot.
- Content license: Wikimedia states that original textual content is generally available under CC BY-SA 4.0 and GFDL, with possible page-specific exceptions.
- Attribution: retain the dump date, source URL, article title or page identifier, and applicable license notice.
- Repository policy: do not commit the raw dump or extracted corpus.

Official references:

- https://dumps.wikimedia.org/viwiki/
- https://dumps.wikimedia.org/legal.html
- https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use

### Intended preprocessing

1. Keep main-namespace article text and a stable article/page ID.
2. Remove markup, tables, references, templates, URLs, and control characters using documented rules.
3. Split article text into sentences while preserving `document_id`.
4. Normalize Unicode to NFC.
5. Audit exact and near duplicates before document-level splitting.
6. Generate synthetic errors independently after splitting.

The group must confirm whether a rule-based Vietnamese tokenizer is permitted. Until confirmed, the project uses its transparent regex tokenizer.

## Proposed real-error supplementary corpus

### ViLexNorm

- Role: optional supplemental evaluation of naturally occurring social-media normalization errors.
- Size reported by the authors: 10,467 original/normalized comment pairs.
- License: CC BY-NC-SA 4.0 according to the official dataset repository.
- Use restriction: research and non-commercial use.
- Scope warning: the dataset includes slang, abbreviations, and lexical normalization phenomena beyond character-level spelling errors.
- Evaluation policy: preserve the official train/development/test partitions. Do not tune on its test partition.
- Compatibility policy: report separately from synthetic spelling-error metrics. Token pairs that require merge/split handling must be excluded or evaluated in a dedicated extension.

Official references:

- https://github.com/ngxtnhi/ViLexNorm
- https://aclanthology.org/2024.eacl-long.85/

## Required local metadata

Before a corpus is used, record:

- Exact source URL and dataset version or dump date.
- Download date and file checksum.
- License and attribution requirements.
- Raw file names and sizes.
- Extraction and filtering commands.
- Number of documents, sentences, and tokens before and after filtering.
- Removed exact duplicates and reported near duplicates.
- Train, validation, and test seeds and document counts.

## Leakage policy

- Split by `document_id` before generating synthetic errors.
- Fit vocabulary, word frequencies, candidate dictionary, and all n-gram counts only on train.
- Select detector threshold, interpolation weights, and ranker weights only on validation.
- Run the final test after the design and hyperparameters are locked.
- Keep real-error test data separate from synthetic training data unless an instructor explicitly approves another protocol.

## Approval checklist

- [ ] Instructor approves Vietnamese Wikipedia as the clean training source.
- [ ] Instructor confirms whether rule-based tokenization libraries are allowed.
- [ ] Instructor approves synthetic noise as the primary detector training method.
- [ ] Instructor approves AI-assisted coding when AI is absent from the runtime pipeline.
- [ ] Group records the selected dump date and SHA-256 checksum.
- [ ] Group reviews the audit report before preparing the final splits.

