from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


LABEL_MAP = {
    0: "normal",
    1: "suspicious",
    2: "attack",
}


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        impl_dir = candidate / "Implementation"
        backend_artifacts = candidate / "MVP" / "backend" / "artifacts"
        if impl_dir.exists() and backend_artifacts.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find project root. Expected both 'Implementation' and 'MVP/backend/artifacts'."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)

BACKEND_ROOT = PROJECT_ROOT / "MVP" / "backend"
ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model" / "cnn_gru_intrusion_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "preprocessing" / "scaler.pkl"
CATEGORY_MAPPINGS_PATH = ARTIFACTS_DIR / "preprocessing" / "category_mappings.json"
SCALER_COLUMNS_PATH = ARTIFACTS_DIR / "preprocessing" / "scaler_columns.json"
SEQUENCE_BUILD_CONFIG_PATH = ARTIFACTS_DIR / "sequences" / "sequence_build_config.json"
SEQUENCE_FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "sequences" / "sequence_feature_columns.json"

OUTPUT_DIR = SCRIPT_DIR / "check_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_TIME_COLUMN = "ltime"
SECONDARY_TIME_COLUMN = "stime"
DROP_IF_PRESENT = ["label", "attack_cat"]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_string_list(data: Any, preferred_keys: list[str] | None = None) -> list[str]:
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in preferred_keys or []:
            value = data.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
        for value in data.values():
            if isinstance(value, list):
                return [str(x) for x in value]
    raise TypeError(
        f"Unsupported config format: {type(data).__name__}. Expected a list or dict containing a list."
    )


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    result = numerator / safe_denominator
    return result.replace([np.inf, -np.inf], 0).fillna(0)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_column_names(df.copy())
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    for col in cleaned.columns:
        if col in {"proto", "state", "service"}:
            cleaned[col] = cleaned[col].astype(str)
        else:
            try:
                cleaned[col] = pd.to_numeric(cleaned[col])
            except Exception:
                pass

    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        cleaned[numeric_columns] = cleaned[numeric_columns].fillna(0)

    return cleaned.replace([np.inf, -np.inf], 0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    if {"sbytes", "dbytes"}.issubset(engineered.columns):
        engineered["bytes_total"] = engineered["sbytes"] + engineered["dbytes"]
        engineered["byte_ratio"] = safe_divide(engineered["sbytes"], engineered["dbytes"] + 1)

    if {"spkts", "dpkts"}.issubset(engineered.columns):
        engineered["pkts_total"] = engineered["spkts"] + engineered["dpkts"]
        engineered["pkt_ratio"] = safe_divide(engineered["spkts"], engineered["dpkts"] + 1)

    if {"sttl", "dttl"}.issubset(engineered.columns):
        engineered["ttl_gap"] = (engineered["sttl"] - engineered["dttl"]).abs()

    if {"sload", "dload"}.issubset(engineered.columns):
        engineered["load_total"] = engineered["sload"] + engineered["dload"]

    numeric_columns = engineered.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        engineered[numeric_columns] = engineered[numeric_columns].fillna(0)

    return engineered


def sort_dataset_by_time(df: pd.DataFrame) -> pd.DataFrame:
    if PRIMARY_TIME_COLUMN in df.columns and SECONDARY_TIME_COLUMN in df.columns:
        return df.sort_values(
            by=[PRIMARY_TIME_COLUMN, SECONDARY_TIME_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)
    return df.reset_index(drop=True)


def ensure_required_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler_columns: list[str],
    category_mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    result = df.copy()
    for col in set(feature_columns) | set(scaler_columns) | set(category_mappings.keys()):
        if col not in result.columns:
            result[col] = "missing" if col in category_mappings else 0
    return result


def encode_categorical_features_for_inference(
    df: pd.DataFrame,
    category_mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    encoded = df.copy()
    for column, mapping in category_mappings.items():
        if column in encoded.columns:
            encoded[column] = encoded[column].astype(str).map(mapping).fillna(-1).astype(int)
    return encoded


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result


def resolve_scaler_columns(
    scaler: Any,
    scaler_columns_raw: list[str],
) -> list[str]:
    if hasattr(scaler, "feature_names_in_"):
        try:
            return [str(x) for x in scaler.feature_names_in_.tolist()]
        except Exception:
            return [str(x) for x in scaler.feature_names_in_]
    return [str(x) for x in scaler_columns_raw]


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    sequence_stride: int,
) -> tuple[np.ndarray, list[int]]:
    values = df[feature_columns].values.astype(np.float32)
    sequences = []
    end_indices = []

    for start_index in range(0, len(df) - sequence_length + 1, sequence_stride):
        end_index = start_index + sequence_length
        sequences.append(values[start_index:end_index])
        end_indices.append(end_index - 1)

    if not sequences:
        return np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32), []

    return np.asarray(sequences, dtype=np.float32), end_indices


def run_inference(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    required_paths = [
        MODEL_PATH,
        SCALER_PATH,
        CATEGORY_MAPPINGS_PATH,
        SCALER_COLUMNS_PATH,
        SEQUENCE_BUILD_CONFIG_PATH,
        SEQUENCE_FEATURE_COLUMNS_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    scaler = joblib.load(SCALER_PATH)
    category_mappings = load_json(CATEGORY_MAPPINGS_PATH)
    scaler_columns_raw = extract_string_list(load_json(SCALER_COLUMNS_PATH), ["scaler_columns", "columns"])
    sequence_config = load_json(SEQUENCE_BUILD_CONFIG_PATH)
    feature_columns = extract_string_list(load_json(SEQUENCE_FEATURE_COLUMNS_PATH), ["feature_columns", "columns"])

    sequence_length = int(sequence_config.get("sequence_length", sequence_config.get("window_size", 10)))
    sequence_stride = int(sequence_config.get("sequence_stride", sequence_config.get("stride", 1)))

    scaler_columns = resolve_scaler_columns(scaler, scaler_columns_raw)
    model = tf.keras.models.load_model(MODEL_PATH)

    raw_df = pd.read_csv(csv_path)
    working_df = raw_df.copy()
    working_df = normalize_column_names(working_df)

    drop_columns = [c for c in DROP_IF_PRESENT if c in working_df.columns]
    if drop_columns:
        working_df = working_df.drop(columns=drop_columns)

    working_df = clean_dataset(working_df)
    working_df = engineer_features(working_df)
    working_df = sort_dataset_by_time(working_df)
    working_df = ensure_required_columns(
        working_df,
        feature_columns=feature_columns,
        scaler_columns=scaler_columns,
        category_mappings=category_mappings,
    )

    # Encode categoricals BEFORE scaling because the fitted scaler expects proto/service/state too.
    working_df = encode_categorical_features_for_inference(working_df, category_mappings)

    # Make sure scaler input columns are numeric and present in exact fitted order.
    for col in scaler_columns:
        if col not in working_df.columns:
            working_df[col] = 0

    working_df = coerce_numeric_columns(working_df, scaler_columns)
    working_df[scaler_columns] = scaler.transform(working_df[scaler_columns])

    # Ensure final model feature columns exist and are numeric.
    for col in feature_columns:
        if col not in working_df.columns:
            working_df[col] = 0

    working_df = coerce_numeric_columns(working_df, feature_columns)
    working_df[feature_columns] = working_df[feature_columns].fillna(0)

    X_sequences, end_indices = build_sequences(
        working_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )

    if len(X_sequences) == 0:
        raise ValueError(
            f"Not enough rows to build sequences. Need at least sequence_length={sequence_length}, "
            f"but got {len(working_df)} rows."
        )

    probabilities = model.predict(X_sequences, verbose=0)
    predicted_label_ids = probabilities.argmax(axis=1)

    predictions_df = pd.DataFrame(
        {
            "window_index": np.arange(len(predicted_label_ids)),
            "source_row_index": end_indices,
            "predicted_label_id": predicted_label_ids,
            "predicted_label_name": [LABEL_MAP[int(x)] for x in predicted_label_ids],
            "probability_normal": probabilities[:, 0],
            "probability_suspicious": probabilities[:, 1],
            "probability_attack": probabilities[:, 2],
        }
    )

    counts = predictions_df["predicted_label_name"].value_counts().to_dict()

    summary = {
        "input_csv": str(csv_path),
        "project_root": str(PROJECT_ROOT),
        "artifacts_root": str(ARTIFACTS_DIR),
        "rows_in_csv": int(len(raw_df)),
        "rows_used_after_cleaning": int(len(working_df)),
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "windows_generated": int(len(predictions_df)),
        "scaler_columns_used": scaler_columns,
        "class_counts": {
            "normal": int(counts.get("normal", 0)),
            "suspicious": int(counts.get("suspicious", 0)),
            "attack": int(counts.get("attack", 0)),
        },
    }

    return predictions_df, summary


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python run_csv_inference_check.py "path/to/file.csv"')
        sys.exit(1)

    csv_path = Path(sys.argv[1]).expanduser().resolve()
    predictions_df, summary = run_inference(csv_path)

    base_name = csv_path.stem
    predictions_path = OUTPUT_DIR / f"{base_name}_predictions.csv"
    summary_path = OUTPUT_DIR / f"{base_name}_summary.json"

    predictions_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CSV Inference Check Complete ===")
    print(f"Input CSV        : {csv_path}")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Artifacts root   : {ARTIFACTS_DIR}")
    print(f"Rows in CSV      : {summary['rows_in_csv']}")
    print(f"Rows after clean : {summary['rows_used_after_cleaning']}")
    print(f"Sequence length  : {summary['sequence_length']}")
    print(f"Sequence stride  : {summary['sequence_stride']}")
    print(f"Windows generated: {summary['windows_generated']}")
    print(f"Scaler cols used : {len(summary['scaler_columns_used'])}")
    print("\nClass counts:")
    print(f"  Normal     : {summary['class_counts']['normal']}")
    print(f"  Suspicious : {summary['class_counts']['suspicious']}")
    print(f"  Attack     : {summary['class_counts']['attack']}")
    print("\nSaved files:")
    print(f"  Predictions CSV: {predictions_path}")
    print(f"  Summary JSON   : {summary_path}")


if __name__ == "__main__":
    main()
