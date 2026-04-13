import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Base directory for the project.
BASE_DIR = Path(__file__).resolve().parents[1]

# Data directories.
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PSEUDO_LABEL_DATA_DIR = BASE_DIR / "data" / "pseudo_labels"
SEQUENCE_DATA_DIR = BASE_DIR / "data" / "sequences"

SEQUENCE_PLOT_DIR = BASE_DIR / "outputs" / "plots" / "sequences"

# Configure how tabular traffic rows are converted into 
# fixed-length temporal windows.
SEQUENCE_LENGTH = 10
SEQUENCE_STRIDE = 1
REQUIRE_FULLY_STABLE_WINDOW_LABEL = False

# Map sequence label identifiers to readable class names.
LABEL_ID_TO_NAME = {
    0: "normal",
    1: "suspicious",
    2: "attack",
}


# Print a formatted step header for better readability in logs.
def print_step(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# Create necessary folders for storing sequence datasets and plots.
def create_folders() -> None:
    for folder in [SEQUENCE_DATA_DIR, SEQUENCE_PLOT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


# Save structured sequence configuration data in JSON format.
def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# Load processed features, metadata, and pseudo labels for a given data split.
def load_split_data(split_name: str):
    feature_table = pd.read_csv(PROCESSED_DATA_DIR / f"{split_name}_processed.csv")
    metadata_table = pd.read_csv(PROCESSED_DATA_DIR / f"{split_name}_metadata.csv")
    pseudo_label_table = pd.read_csv(PSEUDO_LABEL_DATA_DIR / f"{split_name}_pseudo_labeled.csv")
    
    return feature_table, metadata_table, pseudo_label_table


# Merge processed feature rows with pseudo-label information so sequence windows
# can be built from one aligned, chronologically ordered table.
def merge_processed_features_with_labels(
    feature_table: pd.DataFrame,
    metadata_table: pd.DataFrame,
    pseudo_label_table: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    if ("row_id" not in metadata_table.columns or "row_id" not in pseudo_label_table.columns):
        raise ValueError(f"{split_name}: row_id missing in metadata or pseudo-label table.")
    
    merged_table = feature_table.copy()
    merged_table["row_id"] = metadata_table["row_id"].values

    join_columns = [
        "row_id",
        "pseudo_label",
        "pseudo_label_id",
        "label",
        "attack_cat",
        "stime",
        "ltime",
    ]
    available_columns = [column for column in join_columns if column in pseudo_label_table.columns]

    merged_table = merged_table.merge(
        pseudo_label_table[available_columns],
        on="row_id",
        how="inner",
    )

    if "ltime" in merged_table.columns and "stime" in merged_table.columns:
        # Reapply chronological ordering after the merge.
        merged_table = merged_table.sort_values(
            ["ltime", "stime", "row_id"],
            kind="mergesort",
        ).reset_index(drop=True)

    return merged_table


# Select only the feature columns that should be fed into the sequence model,
# excluding identifiers, labels, and preserved metadata fields.
def get_model_feature_columns(merged_table: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "row_id",
        "pseudo_label",
        "pseudo_label_id",
        "label",
        "attack_cat",
        "stime",
        "ltime",
    }
    return [column for column in merged_table.columns if column not in excluded_columns]


# Convert a chronologically ordered table of traffic rows into fixed-length sequences
# with corresponding labels, based on the pseudo-label stability within each window.
def create_sequences(
    merged_table: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = SEQUENCE_STRIDE,
    require_stable_window_label: bool = REQUIRE_FULLY_STABLE_WINDOW_LABEL,
):
    # Extract the model input features and sequence labels as NumPy arrays for efficient slicing.
    feature_values = merged_table[feature_columns].values.astype(np.float32)
    label_values = merged_table["pseudo_label_id"].values.astype(np.int64)
    
    # Preserve row-level identifiers and timing fields for later sequence metadata tracking.
    row_ids = merged_table["row_id"].values
    ltimes = (
        merged_table["ltime"].values
        if "ltime" in merged_table.columns
        else np.zeros(len(merged_table))
    )
    stimes = (
        merged_table["stime"].values
        if "stime" in merged_table.columns
        else np.zeros(len(merged_table))
    )
    true_binary_values = (
        merged_table["label"].values.astype(np.int64)
        if "label" in merged_table.columns
        else np.full(len(merged_table), -1)
    )
    
    sequence_features = []
    sequence_labels = []
    sequence_metadata_rows = []
    
    # Slide a fixed-length window across the ordered records to build overlapping sequences.
    for start_index in range(0, len(merged_table) - sequence_length + 1, stride):
        end_index = start_index + sequence_length
        
        # Extract one candidate feature window and its row-level pseudo labels.
        window_features = feature_values[start_index:end_index]
        window_labels = label_values[start_index:end_index]
        
        # Optionally keep only windows whose labels are stable across the full sequence.
        if require_stable_window_label and len(np.unique(window_labels)) != 1:
            continue
        
        # Use the final time step label as the sequence target.
        final_label = int(window_labels[-1])
        
        sequence_features.append(window_features)
        sequence_labels.append(final_label)
        
        # Preserve metadata from the final row so predictions can be traced back later.
        sequence_metadata_rows.append(
            {
                "last_row_id": int(row_ids[end_index - 1]),
                "last_ltime": float(ltimes[end_index - 1]),
                "last_stime": float(stimes[end_index - 1]),
                "last_true_binary": int(true_binary_values[end_index - 1]),
                "sequence_label_id": final_label,
                "sequence_label_text": LABEL_ID_TO_NAME[final_label],
            }
        )

    return (
        np.asarray(sequence_features, dtype=np.float32),
        np.asarray(sequence_labels, dtype=np.int64),
        pd.DataFrame(sequence_metadata_rows),
    )


# Save the generated sequence tensors and label arrays.
def save_sequence_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    np.save(SEQUENCE_DATA_DIR / "X_train_sequences.npy", X_train)
    np.save(SEQUENCE_DATA_DIR / "y_train_labels.npy", y_train)

    np.save(SEQUENCE_DATA_DIR / "X_validation_sequences.npy", X_validation)
    np.save(SEQUENCE_DATA_DIR / "y_validation_labels.npy", y_validation)

    np.save(SEQUENCE_DATA_DIR / "X_test_sequences.npy", X_test)
    np.save(SEQUENCE_DATA_DIR / "y_test_labels.npy", y_test)


# Save sequence-level metadata.
def save_sequence_metadata(
    train_sequence_metadata: pd.DataFrame,
    validation_sequence_metadata: pd.DataFrame,
    test_sequence_metadata: pd.DataFrame,
) -> None:
    train_sequence_metadata.to_csv(SEQUENCE_DATA_DIR / "train_sequence_metadata.csv", index=False)
    validation_sequence_metadata.to_csv(SEQUENCE_DATA_DIR / "validation_sequence_metadata.csv", index=False)
    test_sequence_metadata.to_csv(SEQUENCE_DATA_DIR / "test_sequence_metadata.csv", index=False)


# Save the selected feature columns and sequence-building settings.
def save_sequence_config(feature_columns: list[str], build_info: dict) -> None:
    save_json({"feature_columns": feature_columns}, SEQUENCE_DATA_DIR / "sequence_feature_columns.json")
    save_json(build_info, SEQUENCE_DATA_DIR / "sequence_build_config.json")


# Create simple label-distribution plots for the generated train, validation,
# and test sequence sets.
def create_sequence_plots(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> None:
    for split_name, labels in [
        ("train", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ]:
        label_counts = pd.Series(labels).value_counts().sort_index()
        display_names = [
            LABEL_ID_TO_NAME.get(int(label_id), str(label_id))
            for label_id in label_counts.index
        ]

        plt.figure(figsize=(7, 5))
        plt.bar(display_names, label_counts.values)
        plt.title(f"{split_name.title()} sequence label distribution")
        plt.tight_layout()
        plt.savefig(SEQUENCE_PLOT_DIR / f"{split_name}_sequence_label_distribution.png", dpi=150)
        plt.close()


# Print the generated sequence tensor and label-array shapes for quick
# verification during execution.
def print_sequence_summary(split_name: str, X_sequences: np.ndarray, y_sequences: np.ndarray) -> None:
    print(f"[INFO] {split_name} X shape: {X_sequences.shape}")
    print(f"[INFO] {split_name} y shape: {y_sequences.shape}")


# Define the main execution flow.
def main() -> None:
    create_folders()

    print_step("[STEP 1] Load processed features and pseudo labels")
    train_features, train_metadata, train_labels = load_split_data("train")
    validation_features, validation_metadata, validation_labels = load_split_data("validation")
    test_features, test_metadata, test_labels = load_split_data("test")

    print_step("[STEP 2] Merge processed features with pseudo labels")
    train_table = merge_processed_features_with_labels(train_features, train_metadata, train_labels, "train")
    validation_table = merge_processed_features_with_labels(validation_features, validation_metadata, validation_labels, "validation")
    test_table = merge_processed_features_with_labels(test_features, test_metadata, test_labels, "test")

    feature_columns = get_model_feature_columns(train_table)

    print_step("[STEP 3] Create sequence datasets")
    X_train, y_train, train_sequence_metadata = create_sequences(train_table, feature_columns, "train")
    X_validation, y_validation, validation_sequence_metadata = create_sequences(validation_table, feature_columns, "validation")
    X_test, y_test, test_sequence_metadata = create_sequences(test_table, feature_columns, "test")

    print_sequence_summary("Train", X_train, y_train)
    print_sequence_summary("Validation", X_validation, y_validation)
    print_sequence_summary("Test", X_test, y_test)

    print_step("[STEP 4] Save sequence datasets and plots")
    save_sequence_datasets(X_train, y_train, X_validation, y_validation, X_test, y_test)
    save_sequence_metadata(train_sequence_metadata, validation_sequence_metadata, test_sequence_metadata)
    save_sequence_config(
        feature_columns,
        {
            "sequence_length": SEQUENCE_LENGTH,
            "sequence_stride": SEQUENCE_STRIDE,
            "require_fully_stable_window_label": REQUIRE_FULLY_STABLE_WINDOW_LABEL,
        },
    )
    create_sequence_plots(y_train, y_validation, y_test)

    print_step("[DONE]")
    print("[INFO] Sequence dataset building completed successfully.")


# Execute the main function when the script is run directly.
if __name__ == "__main__":
    main()
