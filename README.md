# Vietnamese Spell Correction

A from-scratch Vietnamese spelling detection and correction system using a BiLSTM token detector, edit-distance candidate generation, and a statistical trigram language model.

## Overview

This project is developed as a final assignment for a Natural Language Processing course. It focuses on detecting misspelled Vietnamese tokens and suggesting context-aware corrections while keeping every learned component explainable and reproducible.

The minimum viable system consists of:

1. A BiLSTM token classifier trained from scratch to label tokens as `CORRECT` or `ERROR`.
2. An edit-distance candidate generator built from the training vocabulary.
3. A statistical trigram language model trained on clean training text.
4. A ranker combining context, edit-distance, and word-frequency scores.
5. A Streamlit interface for interactive demonstrations.

## Core constraints

- All model weights and embeddings are initialized and trained from scratch.
- No pretrained model, embedding, tokenizer, or checkpoint is used.
- No external AI or spelling-correction API is part of the data pipeline or application.
- Vocabulary, dictionary, frequency statistics, and n-gram counts are fitted only on the training split.
- Documents are split before synthetic errors are generated to prevent data leakage.
- Validation data is used for model selection. Test data is reserved for final evaluation.

## Planned pipeline

```text
Input sentence
    -> normalization and tokenization
    -> BiLSTM error detection
    -> edit-distance candidate generation
    -> trigram context scoring
    -> candidate ranking
    -> Top-1 correction and Top-3 suggestions
```

## Repository structure

```text
vietnamese-spell-correction/
├── app/                 # Streamlit application
├── artifacts/           # Locally trained model artifacts
├── configs/             # Data, model, and ranking configuration
├── data/                # Data documentation and local datasets
├── notebooks/           # Exploration and experiment notebooks
├── reports/             # Evaluation results and figures
├── scripts/             # Reproducible command-line workflows
├── src/                 # Core data, model, and correction modules
├── tests/               # Automated tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Development status

The repository currently contains the initial project structure. Data preparation, baselines, model training, evaluation, and the demo will be implemented incrementally.

## Evaluation plan

- Detector: precision, recall, and F1 score for the `ERROR` class.
- Candidate generator: Recall@1, Recall@3, Recall@5, and latency.
- Correction: Accuracy@1, Accuracy@3, end-to-end F1, sentence accuracy, and false-correction rate.
- Experiments: an edit-distance baseline and at least one ablation study.

## Academic scope

The MVP targets token-level Vietnamese typographical and spelling errors, including character insertion, deletion, substitution, transposition, and diacritic errors. General grammar correction and deep semantic rewriting are outside the MVP scope.

