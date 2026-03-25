# CNN-GRU + XAI Intrusion Detection Pipeline

This repository contains the cleaned FYP implementation for intrusion detection and forensic analysis using a CNN-GRU model with XAI outputs.

## Pipeline Steps

Current active scripts in order:

1. `01_data_preprocessing.py`
2. `02_pseudo_label_generation.py`
3. `03_sequence_dataset_builder.py`
4. `04_model_training.py`
5. `05_model_evaluation_and_xai.py`
6. `06_forensic_reporting.py`

## Folder Structure

```text
Implementation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── pseudo_labels/
│   └── sequences/
│
├── models/
│   ├── preprocessing/
│   ├── pseudo_labels/
│   ├── trained_models/
│   └── training_history/
│
├── outputs/
│   ├── evaluation/
│   ├── pseudo_labels/
│   ├── xai/
│   │   ├── shap/
│   │   ├── lime/
│   │   └── comparison/
│   ├── forensic_reports/
│   │   └── cases/
│   ├── forensic_audit/
│   └── plots/
│       ├── preprocessing/
│       ├── pseudo_labels/
│       ├── sequences/
│       ├── training/
│       ├── evaluation/
│       ├── xai/
│       └── forensic/
│
├── scripts/
├── archive/
├── run_pipeline.bat
└── README.md
```

## Required Input Data

Place UNSW-NB15 source files in `data/raw/`:

- `UNSW-NB15_1.csv`
- `UNSW-NB15_2.csv`
- `UNSW-NB15_3.csv`
- `UNSW-NB15_4.csv`
- `NUSW-NB15_features.csv`

Additional support files can remain in `data/raw/`.

## How To Run

From project root:

```bat
cd D:\Studies\IIT\FYP\Implementation
```

Run step-by-step:

```bat
python scripts\01_data_preprocessing.py
python scripts\02_pseudo_label_generation.py
python scripts\03_sequence_dataset_builder.py
python scripts\04_model_training.py
python scripts\05_model_evaluation_and_xai.py
python scripts\06_forensic_reporting.py
```

Or run the full pipeline:

```bat
run_pipeline.bat
```

## Notes

- `archive/` is intentionally kept for historical versions.
- Generated data, model artifacts, and outputs are excluded from GitHub using `.gitignore`.
- If paths change later, update path handling in `scripts/project_paths.py`.
