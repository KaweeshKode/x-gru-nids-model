import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

SEQUENCE_DATA_DIR = BASE_DIR / "data" / "sequences"
TRAINED_MODEL_DIR = BASE_DIR / "models" / "trained_models"
TRAINING_HISTORY_DIR = BASE_DIR / "models" / "training_history"

EVALUATION_OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"
SHAP_OUTPUT_DIR = BASE_DIR / "outputs" / "xai" / "shap"
LIME_OUTPUT_DIR = BASE_DIR / "outputs" / "xai" / "lime"

FORENSIC_REPORT_DIR = BASE_DIR / "outputs" / "forensic_reports"
FORENSIC_CASE_DIR = FORENSIC_REPORT_DIR / "cases"
FORENSIC_AUDIT_DIR = BASE_DIR / "outputs" / "forensic_audit"
FORENSIC_PLOT_DIR = BASE_DIR / "outputs" / "plots" / "forensic"

CLASS_ID_TO_NAME = {
    0: "normal",
    1: "suspicious",
    2: "attack",
}


def print_step(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_folders() -> None:
    for folder in [
        FORENSIC_REPORT_DIR,
        FORENSIC_CASE_DIR,
        FORENSIC_AUDIT_DIR,
        FORENSIC_PLOT_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_reporting_inputs():
    prediction_table = pd.read_csv(EVALUATION_OUTPUT_DIR / "test_predictions.csv")
    sequence_metadata_table = pd.read_csv(SEQUENCE_DATA_DIR / "test_sequence_metadata.csv")

    shap_global_table = pd.read_csv(SHAP_OUTPUT_DIR / "shap_global_feature_importance.csv")
    shap_local_summary_table = pd.read_csv(SHAP_OUTPUT_DIR / "shap_local_summary.csv")

    lime_summary_path = LIME_OUTPUT_DIR / "lime_explanation_summary.csv"
    lime_summary_table = pd.read_csv(lime_summary_path) if lime_summary_path.exists() else pd.DataFrame()

    return (
        prediction_table,
        sequence_metadata_table,
        shap_global_table,
        shap_local_summary_table,
        lime_summary_table,
    )


def _normalize_lime_feature_name(condition_or_feature: str) -> str:
    cleaned = str(condition_or_feature)
    for operator in ["<=", ">=", "<", ">", "="]:
        if operator in cleaned:
            left_part = cleaned.split(operator)[0].strip()
            right_part = cleaned.split(operator)[-1].strip()
            if left_part.startswith("t") and "_" in left_part:
                return left_part
            if right_part.startswith("t") and "_" in right_part:
                return right_part
    return cleaned.strip()


def select_forensic_cases(prediction_table: pd.DataFrame) -> pd.DataFrame:
    return prediction_table[
        prediction_table["predicted_label_name"].isin(["suspicious", "attack"])
    ].copy().reset_index(drop=True)


def write_plain_language_explanation(predicted_label_name: str, shap_features: list[str], lime_features: list[str]) -> str:
    key_shap = ", ".join(shap_features[:3]) if shap_features else "no strong SHAP indicators"
    key_lime = ", ".join(lime_features[:3]) if lime_features else "no strong LIME indicators"
    return (
        f"The model marked this case as {predicted_label_name}. "
        f"SHAP highlighted {key_shap}. "
        f"LIME highlighted {key_lime}. "
        "These indicators suggest the sequence contains traffic behaviour that differs from normal baseline activity."
    )


def write_analyst_recommendation(predicted_label_name: str) -> str:
    if predicted_label_name == "attack":
        return "Escalate immediately for analyst review and confirm with packet- or log-level evidence."
    return "Mark for analyst triage and verify whether the behaviour matches known suspicious activity."


def build_forensic_case_record(
    case_number: int,
    case_row: pd.Series,
    shap_local_summary_table: pd.DataFrame,
    lime_summary_table: pd.DataFrame,
):
    sample_row_index = int(case_row.name)

    shap_case = shap_local_summary_table[
        shap_local_summary_table["sample_row_index"] == sample_row_index
    ].copy()

    lime_case = lime_summary_table[
        lime_summary_table["sample_row_index"] == sample_row_index
    ].copy()

    shap_top_features = shap_case["flat_feature"].head(10).astype(str).tolist() if not shap_case.empty else []
    lime_case["normalized_feature"] = (
        lime_case["condition_or_feature"].apply(_normalize_lime_feature_name)
        if not lime_case.empty
        else []
    )
    lime_top_features = lime_case["normalized_feature"].head(10).astype(str).tolist() if not lime_case.empty else []

    shared_features = sorted(set(shap_top_features).intersection(set(lime_top_features)))

    forensic_record = {
        "case_id": f"case_{case_number:06d}",
        "sample_row_index": sample_row_index,
        "last_row_id": int(case_row.get("last_row_id", -1)),
        "predicted_label_id": int(case_row["predicted_label_id"]),
        "predicted_label_name": str(case_row["predicted_label_name"]),
        "true_label_id": int(case_row["true_label_id"]),
        "true_label_name": str(case_row["true_label_name"]),
        "probability_normal": float(case_row["probability_normal"]),
        "probability_suspicious": float(case_row["probability_suspicious"]),
        "probability_attack": float(case_row["probability_attack"]),
        "shap_top_features": shap_top_features,
        "lime_top_features": lime_top_features,
        "shared_features": shared_features,
        "plain_language_explanation": write_plain_language_explanation(
            str(case_row["predicted_label_name"]),
            shap_top_features,
            lime_top_features,
        ),
        "analyst_recommendation": write_analyst_recommendation(str(case_row["predicted_label_name"])),
    }
    return forensic_record


def generate_forensic_case_reports(
    forensic_cases_table: pd.DataFrame,
    shap_local_summary_table: pd.DataFrame,
    lime_summary_table: pd.DataFrame,
):
    forensic_records = []

    for case_number, (_, case_row) in enumerate(forensic_cases_table.iterrows(), start=1):
        record = build_forensic_case_record(
            case_number,
            case_row,
            shap_local_summary_table,
            lime_summary_table,
        )
        forensic_records.append(record)

        save_json(record, FORENSIC_CASE_DIR / f"{record['case_id']}.json")

    return forensic_records


def save_forensic_summary(forensic_records: list[dict]) -> pd.DataFrame:
    summary_table = pd.DataFrame(forensic_records)
    summary_table.to_csv(FORENSIC_REPORT_DIR / "forensic_case_summary.csv", index=False)

    if not summary_table.empty:
        sample_record = summary_table.iloc[0].to_dict()
        with open(FORENSIC_REPORT_DIR / "sample_forensic_report.txt", "w", encoding="utf-8") as handle:
            handle.write("Sample forensic report\n")
            handle.write("======================\n")
            for key, value in sample_record.items():
                handle.write(f"{key}: {value}\n")

    return summary_table


def create_forensic_plots(summary_table: pd.DataFrame) -> None:
    if summary_table.empty:
        return

    label_counts = summary_table["predicted_label_name"].value_counts()

    plt.figure(figsize=(7, 5))
    plt.bar(label_counts.index.astype(str), label_counts.values)
    plt.title("Forensic case distribution")
    plt.tight_layout()
    plt.savefig(FORENSIC_PLOT_DIR / "forensic_case_distribution.png", dpi=150)
    plt.close()

    all_shared_features = []
    for shared_feature_list in summary_table["shared_features"]:
        if isinstance(shared_feature_list, list):
            all_shared_features.extend(shared_feature_list)

    if all_shared_features:
        feature_counts = pd.Series(all_shared_features).value_counts().head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_counts.index[::-1], feature_counts.values[::-1])
        plt.title("Top recurring shared indicators")
        plt.tight_layout()
        plt.savefig(FORENSIC_PLOT_DIR / "top_shared_indicators.png", dpi=150)
        plt.close()


def build_run_manifest() -> dict:
    model_path = TRAINED_MODEL_DIR / "cnn_gru_intrusion_model.keras"
    prediction_path = EVALUATION_OUTPUT_DIR / "test_predictions.csv"
    shap_global_path = SHAP_OUTPUT_DIR / "shap_global_feature_importance.csv"
    lime_summary_path = LIME_OUTPUT_DIR / "lime_explanation_summary.csv"

    return {
        "mode": "ultralight_summary_only",
        "model_file": str(model_path),
        "model_sha256": file_sha256(model_path),
        "prediction_file": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_path),
        "shap_global_file": str(shap_global_path),
        "shap_global_sha256": file_sha256(shap_global_path),
        "lime_summary_file": str(lime_summary_path),
        "lime_summary_sha256": file_sha256(lime_summary_path) if lime_summary_path.exists() else None,
    }


def build_evidence_provenance_summary(
    prediction_table: pd.DataFrame,
    summary_table: pd.DataFrame,
) -> pd.DataFrame:
    if summary_table.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "sample_row_index",
                "last_row_id",
                "predicted_label_name",
                "true_label_name",
            ]
        )

    prediction_copy = prediction_table.copy()

    # Make sure the merge key exists on the prediction side.
    if "sample_row_index" not in prediction_copy.columns:
        prediction_copy = prediction_copy.reset_index().rename(
            columns={"index": "sample_row_index"}
        )

    # Make sure the merge key exists on the summary side too.
    if "sample_row_index" not in summary_table.columns:
        raise ValueError("summary_table is missing 'sample_row_index'.")

    merged_table = summary_table.merge(
        prediction_copy,
        on="sample_row_index",
        how="left",
        suffixes=("_summary", "_prediction"),
    )

    output_table = pd.DataFrame({
        "case_id": merged_table["case_id"],
        "sample_row_index": merged_table["sample_row_index"],
        "last_row_id": merged_table.get("last_row_id_summary", merged_table.get("last_row_id")),
        "predicted_label_name": merged_table.get(
            "predicted_label_name_summary",
            merged_table.get("predicted_label_name"),
        ),
        "true_label_name": merged_table.get(
            "true_label_name_summary",
            merged_table.get("true_label_name"),
        ),
    })

    return output_table


def build_feature_integrity_report() -> dict:
    feature_columns_path = SEQUENCE_DATA_DIR / "sequence_feature_columns.json"
    training_config_path = TRAINING_HISTORY_DIR / "training_config.json"

    feature_payload = {}
    if feature_columns_path.exists():
        with open(feature_columns_path, "r", encoding="utf-8") as handle:
            feature_payload = json.load(handle)

    training_payload = {}
    if training_config_path.exists():
        with open(training_config_path, "r", encoding="utf-8") as handle:
            training_payload = json.load(handle)

    return {
        "feature_columns_file": str(feature_columns_path),
        "feature_columns_sha256": file_sha256(feature_columns_path),
        "feature_columns_payload": feature_payload,
        "training_config_file": str(training_config_path),
        "training_config_sha256": file_sha256(training_config_path),
        "training_config_payload": training_payload,
    }


def build_reproducibility_checklist() -> dict:
    checks = {
        "model_exists": (TRAINED_MODEL_DIR / "cnn_gru_intrusion_model.keras").exists(),
        "test_predictions_exist": (EVALUATION_OUTPUT_DIR / "test_predictions.csv").exists(),
        "sequence_metadata_exists": (SEQUENCE_DATA_DIR / "test_sequence_metadata.csv").exists(),
        "shap_output_exists": (SHAP_OUTPUT_DIR / "shap_global_feature_importance.csv").exists(),
        "lime_output_exists": (LIME_OUTPUT_DIR / "lime_explanation_summary.csv").exists(),
    }
    checks["all_required_files_present"] = all(checks.values())
    return checks


def build_decision_trace_summary(summary_table: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [
        "case_id",
        "sample_row_index",
        "predicted_label_name",
        "probability_normal",
        "probability_suspicious",
        "probability_attack",
    ]
    available_columns = [column for column in columns_to_keep if column in summary_table.columns]
    return summary_table[available_columns].copy()


def main() -> None:
    create_folders()

    print_step("[STEP 1] Load reporting inputs")
    (
        prediction_table,
        sequence_metadata_table,
        shap_global_table,
        shap_local_summary_table,
        lime_summary_table,
    ) = load_reporting_inputs()

    print_step("[STEP 2] Select suspicious and attack cases")
    forensic_cases_table = select_forensic_cases(prediction_table)

    print_step("[STEP 3] Generate forensic case reports")
    forensic_records = generate_forensic_case_reports(
        forensic_cases_table,
        shap_local_summary_table,
        lime_summary_table,
    )
    summary_table = save_forensic_summary(forensic_records)
    create_forensic_plots(summary_table)

    print_step("[STEP 4] Build forensic audit summary")
    run_manifest = build_run_manifest()
    evidence_provenance_table = build_evidence_provenance_summary(prediction_table, summary_table)
    feature_integrity_report = build_feature_integrity_report()
    reproducibility_checklist = build_reproducibility_checklist()
    decision_trace_summary = build_decision_trace_summary(summary_table)

    save_json(run_manifest, FORENSIC_AUDIT_DIR / "audit_run_manifest.json")
    evidence_provenance_table.to_csv(FORENSIC_AUDIT_DIR / "evidence_provenance_summary.csv", index=False)
    save_json(feature_integrity_report, FORENSIC_AUDIT_DIR / "feature_integrity_report.json")
    save_json(reproducibility_checklist, FORENSIC_AUDIT_DIR / "reproducibility_checklist.json")
    decision_trace_summary.to_csv(FORENSIC_AUDIT_DIR / "decision_trace_summary.csv", index=False)

    print_step("[DONE]")
    print("[INFO] Forensic reporting completed successfully.")
    print(f"[INFO] Forensic cases tracked: {len(summary_table)}")


if __name__ == "__main__":
    main()