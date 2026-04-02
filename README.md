# CNN-GRU + XAI Intrusion Detection Pipeline

This repository contains the cleaned FYP implementation for intrusion detection and forensic analysis using a CNN-GRU model with XAI outputs.

## Prerequisites

- Windows environment (commands below use Windows paths and `.bat` execution)
- Python 3.10 or newer
- Required Python packages installed in your active environment (virtual environment recommended)

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

Dataset download (not stored in this GitHub repo):

- [UNSW-NB15 CSV download folder (SharePoint)](https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS)

Open the link, go to the `CSV Files` folder, and download the required CSV files. Then place them in `data/raw/` in this project.

Required files in `data/raw/`:

- `UNSW-NB15_1.csv`
- `UNSW-NB15_2.csv`
- `UNSW-NB15_3.csv`
- `UNSW-NB15_4.csv`
- `NUSW-NB15_features.csv`

Additional support files can remain in `data/raw/`.

Use the exact filenames expected by the scripts (`UNSW-...` and `NUSW-...` as listed above).

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

`run_pipeline.bat` automatically activates `.venv\Scripts\activate.bat` if that virtual environment exists.

## Troubleshooting

- If a script fails with missing file errors, confirm all required CSV files are in `data/raw/`.
- If `python` is not recognized, install Python and/or activate your virtual environment before running scripts.

## Notes

- `archive/` is intentionally kept for historical versions.
- Generated data, model artifacts, and outputs are excluded from GitHub using `.gitignore`.
- If paths change later, update the path constants directly inside the active pipeline scripts.