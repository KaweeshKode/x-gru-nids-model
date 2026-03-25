import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PREPROCESSING_MODEL_DIR = BASE_DIR / "models" / "preprocessing"
PREPROCESSING_PLOT_DIR = BASE_DIR / "outputs" / "plots" / "preprocessing"

RAW_DATA_FILES = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]
FEATURE_NAME_FILE = "NUSW-NB15_features.csv"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
PRIMARY_TIME_COLUMN = "ltime"
SECONDARY_TIME_COLUMN = "stime"
CATEGORICAL_FEATURE_COLUMNS = ["proto", "state", "service"]
METADATA_COLUMNS = {"srcip", "dstip", "attack_cat", "label", "source_file", "row_id"}


def print_step(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_folders() -> None:
    for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREPROCESSING_MODEL_DIR, PREPROCESSING_PLOT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan) + 1e-9
    return numerator / safe_denominator


def load_feature_names() -> list[str]:
    features_path = RAW_DATA_DIR / FEATURE_NAME_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    feature_table = pd.read_csv(features_path, encoding="latin1")
    if "Name" not in feature_table.columns:
        raise ValueError("Expected column 'Name' not found in the features file.")
    feature_names = [normalize_column_name(name) for name in feature_table["Name"].tolist()]
    if len(feature_names) != 49:
        raise ValueError(f"Expected 49 feature names, but found {len(feature_names)}.")
    return feature_names


def load_raw_dataset() -> pd.DataFrame:
    feature_names = load_feature_names()
    dataset_parts = []
    for file_name in RAW_DATA_FILES:
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        print(f"[INFO] Loading raw file: {file_name}")
        file_table = pd.read_csv(file_path, header=None, low_memory=False)
        if file_table.shape[1] != len(feature_names):
            raise ValueError(f"{file_name} has {file_table.shape[1]} columns, expected {len(feature_names)}.")
        file_table.columns = feature_names
        file_table["source_file"] = file_name
        dataset_parts.append(file_table)
    dataset = pd.concat(dataset_parts, axis=0, ignore_index=True)
    dataset["row_id"] = np.arange(len(dataset))
    print(f"[INFO] Combined raw dataset shape: {dataset.shape}")
    return dataset


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    cleaned_dataset = dataset.copy()
    if "attack_cat" in cleaned_dataset.columns:
        cleaned_dataset["attack_cat"] = cleaned_dataset["attack_cat"].fillna("normal").astype(str).str.strip().str.lower()
    if "label" in cleaned_dataset.columns:
        cleaned_dataset["label"] = pd.to_numeric(cleaned_dataset["label"], errors="coerce").fillna(0).astype(int)
    skip_numeric_cast = {"srcip", "dstip", "proto", "state", "service", "attack_cat", "source_file"}
    candidate_numeric_columns = [column for column in cleaned_dataset.columns if column not in skip_numeric_cast]
    for column in candidate_numeric_columns:
        cleaned_dataset[column] = pd.to_numeric(cleaned_dataset[column], errors="coerce")
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in cleaned_dataset.columns:
            cleaned_dataset[column] = cleaned_dataset[column].fillna("unknown").astype(str).str.strip().str.lower()
    numeric_columns = cleaned_dataset.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        cleaned_dataset[numeric_columns] = cleaned_dataset[numeric_columns].fillna(0)
    cleaned_dataset = cleaned_dataset.replace([np.inf, -np.inf], 0)
    return cleaned_dataset


def engineer_features(dataset: pd.DataFrame) -> pd.DataFrame:
    engineered_dataset = dataset.copy()
    if {"sbytes", "dbytes"}.issubset(engineered_dataset.columns):
        engineered_dataset["bytes_total"] = engineered_dataset["sbytes"] + engineered_dataset["dbytes"]
        engineered_dataset["byte_ratio"] = safe_divide(engineered_dataset["sbytes"], engineered_dataset["dbytes"] + 1)
    if {"spkts", "dpkts"}.issubset(engineered_dataset.columns):
        engineered_dataset["pkts_total"] = engineered_dataset["spkts"] + engineered_dataset["dpkts"]
        engineered_dataset["pkt_ratio"] = safe_divide(engineered_dataset["spkts"], engineered_dataset["dpkts"] + 1)
    if {"sttl", "dttl"}.issubset(engineered_dataset.columns):
        engineered_dataset["ttl_gap"] = (engineered_dataset["sttl"] - engineered_dataset["dttl"]).abs()
    if {"sload", "dload"}.issubset(engineered_dataset.columns):
        engineered_dataset["load_total"] = engineered_dataset["sload"] + engineered_dataset["dload"]
    numeric_columns = engineered_dataset.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        engineered_dataset[numeric_columns] = engineered_dataset[numeric_columns].fillna(0)
    return engineered_dataset


def sort_dataset_by_time(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.sort_values(by=[PRIMARY_TIME_COLUMN, SECONDARY_TIME_COLUMN], kind="mergesort").reset_index(drop=True)


def split_dataset_by_time(dataset: pd.DataFrame):
    total_rows = len(dataset)
    train_end = int(total_rows * TRAIN_RATIO)
    validation_end = int(total_rows * (TRAIN_RATIO + VALIDATION_RATIO))
    return dataset.iloc[:train_end].copy(), dataset.iloc[train_end:validation_end].copy(), dataset.iloc[validation_end:].copy()


def separate_features_and_metadata(train_dataset, validation_dataset, test_dataset):
    def split_one(dataset: pd.DataFrame):
        metadata_columns = [column for column in dataset.columns if column in METADATA_COLUMNS]
        for time_column in [SECONDARY_TIME_COLUMN, PRIMARY_TIME_COLUMN]:
            if time_column in dataset.columns and time_column not in metadata_columns:
                metadata_columns.append(time_column)
        metadata_table = dataset[metadata_columns].copy()
        feature_columns = [column for column in dataset.columns if column not in metadata_columns]
        feature_table = dataset[feature_columns].copy()
        return feature_table, metadata_table
    train_features, train_metadata = split_one(train_dataset)
    validation_features, validation_metadata = split_one(validation_dataset)
    test_features, test_metadata = split_one(test_dataset)
    return train_features, validation_features, test_features, train_metadata, validation_metadata, test_metadata


def fit_categorical_encoders(train_features: pd.DataFrame) -> dict[str, dict[str, int]]:
    category_mappings = {}
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in train_features.columns:
            unique_values = sorted(train_features[column].astype(str).unique().tolist())
            category_mappings[column] = {value: index for index, value in enumerate(unique_values)}
    return category_mappings


def encode_categorical_features(train_features, validation_features, test_features, category_mappings):
    def encode_single(feature_table: pd.DataFrame):
        encoded_table = feature_table.copy()
        for column, mapping in category_mappings.items():
            if column in encoded_table.columns:
                encoded_table[column] = encoded_table[column].astype(str).map(mapping).fillna(-1).astype(int)
        return encoded_table
    return encode_single(train_features), encode_single(validation_features), encode_single(test_features)


def fit_feature_scaler(train_features: pd.DataFrame):
    numeric_feature_columns = train_features.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    scaler.fit(train_features[numeric_feature_columns])
    return scaler, numeric_feature_columns


def scale_numeric_features(train_features, validation_features, test_features, scaler, numeric_feature_columns):
    def scale_single(feature_table: pd.DataFrame):
        scaled_table = feature_table.copy()
        scaled_table[numeric_feature_columns] = scaler.transform(scaled_table[numeric_feature_columns])
        return scaled_table
    return scale_single(train_features), scale_single(validation_features), scale_single(test_features)


def save_processed_datasets(train_features, validation_features, test_features, train_metadata, validation_metadata, test_metadata):
    train_features.to_csv(PROCESSED_DATA_DIR / "train_processed.csv", index=False)
    validation_features.to_csv(PROCESSED_DATA_DIR / "validation_processed.csv", index=False)
    test_features.to_csv(PROCESSED_DATA_DIR / "test_processed.csv", index=False)
    train_metadata.to_csv(PROCESSED_DATA_DIR / "train_metadata.csv", index=False)
    validation_metadata.to_csv(PROCESSED_DATA_DIR / "validation_metadata.csv", index=False)
    test_metadata.to_csv(PROCESSED_DATA_DIR / "test_metadata.csv", index=False)


def save_preprocessing_artifacts(scaler, category_mappings, numeric_feature_columns):
    with open(PREPROCESSING_MODEL_DIR / "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)
    save_json(category_mappings, PREPROCESSING_MODEL_DIR / "category_mappings.json")
    save_json({"scaled_numeric_feature_columns": numeric_feature_columns}, PREPROCESSING_MODEL_DIR / "scaler_columns.json")


def create_preprocessing_plots(train_metadata, validation_metadata, test_metadata, train_features):
    split_sizes = {"train": len(train_metadata), "validation": len(validation_metadata), "test": len(test_metadata)}
    plt.figure(figsize=(7, 5))
    plt.bar(list(split_sizes.keys()), list(split_sizes.values()))
    plt.title("Dataset split sizes")
    plt.tight_layout()
    plt.savefig(PREPROCESSING_PLOT_DIR / "dataset_split_sizes.png", dpi=150)
    plt.close()


def main() -> None:
    create_folders()
    print_step("[STEP 1] Load raw UNSW dataset")
    raw_dataset = load_raw_dataset()
    print_step("[STEP 2] Clean raw dataset")
    cleaned_dataset = clean_dataset(raw_dataset)
    print_step("[STEP 3] Engineer additional features")
    featured_dataset = engineer_features(cleaned_dataset)
    print_step("[STEP 4] Sort dataset by time")
    ordered_dataset = sort_dataset_by_time(featured_dataset)
    print_step("[STEP 5] Split dataset by time")
    train_dataset, validation_dataset, test_dataset = split_dataset_by_time(ordered_dataset)
    print_step("[STEP 6] Separate features and metadata")
    train_features, validation_features, test_features, train_metadata, validation_metadata, test_metadata = separate_features_and_metadata(train_dataset, validation_dataset, test_dataset)
    print_step("[STEP 7] Encode categorical feature columns")
    category_mappings = fit_categorical_encoders(train_features)
    train_features, validation_features, test_features = encode_categorical_features(train_features, validation_features, test_features, category_mappings)
    print_step("[STEP 8] Scale numeric feature columns")
    scaler, numeric_feature_columns = fit_feature_scaler(train_features)
    train_features, validation_features, test_features = scale_numeric_features(train_features, validation_features, test_features, scaler, numeric_feature_columns)
    print_step("[STEP 9] Save processed datasets and artifacts")
    save_processed_datasets(train_features, validation_features, test_features, train_metadata, validation_metadata, test_metadata)
    save_preprocessing_artifacts(scaler, category_mappings, numeric_feature_columns)
    create_preprocessing_plots(train_metadata, validation_metadata, test_metadata, train_features)
    print_step("[DONE]")
    print("[INFO] Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
