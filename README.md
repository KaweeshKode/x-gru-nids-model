# CNN-GRU + XAI Intrusion Detection Pipeline

This project is the cleaned implementation of your FYP pipeline for network intrusion detection and forensic analysis.

## Main flow

The active pipeline in this folder is:

1. `01_prepare_unsw_dataset.py`
2. `04_generate_calibrated_pseudo_labels.py`
3. `05_build_sequence_datasets.py`
4. `06_train_cnn_gru_model.py`
5. `07_evaluate_cnn_gru_model.py`
6. `08_tune_final_decision_rule.py`
7. `09_generate_shap_explanations.py`
8. `10_generate_lime_explanations.py`
9. `11_generate_forensic_reports.py`

The older experimental steps `02` and `03` were intentionally left out of this clean implementation folder.

## Folder layout

```text
Implementation/
│   README.md
│   run_pipeline.bat
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── final/
│
├── logs/
├── models/
├── outputs/
└── scripts/
```

## Data locations

Place the source files here:

- `data/raw/UNSW-NB15_1.csv`
- `data/raw/UNSW-NB15_2.csv`
- `data/raw/UNSW-NB15_3.csv`
- `data/raw/UNSW-NB15_4.csv`
- `data/raw/Training and Testing Sets/UNSW_NB15_training-set.csv`
- `data/raw/Training and Testing Sets/UNSW_NB15_testing-set.csv`

Supporting CSV files such as feature descriptions can stay in `data/raw/`.

## Script outputs

### Intermediate data
- `data/interim/prepared/`
- `data/interim/calibrated_labels/`
- `data/interim/sequences/`

### Models
- `models/cnn_gru/`

### Reports and analysis
- `outputs/evaluation/`
- `outputs/plots/`
- `outputs/xai/shap/`
- `outputs/xai/lime/`
- `outputs/forensic_reports/`

## How to run

Open a terminal in the project root:

```bat
cd D:\Studies\IIT\FYP\Implementation
```

Run step by step:

```bat
python scripts\01_prepare_unsw_dataset.py
python scripts\04_generate_calibrated_pseudo_labels.py
python scripts\05_build_sequence_datasets.py
python scripts\06_train_cnn_gru_model.py
python scripts\07_evaluate_cnn_gru_model.py
python scripts\08_tune_final_decision_rule.py
python scripts\09_generate_shap_explanations.py
python scripts\10_generate_lime_explanations.py
python scripts\11_generate_forensic_reports.py
```

Or run the full pipeline:

```bat
run_pipeline.bat
```

## Shared path helper

The file `project_paths.py` contains the common directory structure used by all scripts. Keep it inside the `scripts/` folder with the pipeline files.

## Notes

- This cleaned version focuses on the final FYP flow: dataset preparation, pseudo-labeling, CNN-GRU training, evaluation, XAI, and forensic reporting.
- Keep old experimental scripts in your archive or in `new-pipeline-test` so the main implementation folder stays clean.
- If you rename folders again later, update `project_paths.py` first instead of changing paths inside every script.
