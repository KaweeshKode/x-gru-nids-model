import json
from pathlib import Path

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)


BASE_DIR = Path(__file__).resolve().parents[1]

SEQUENCE_DATA_DIR = BASE_DIR / "data" / "sequences"
TRAINED_MODEL_DIR = BASE_DIR / "models" / "trained_models"

EVALUATION_OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"
SHAP_OUTPUT_DIR = BASE_DIR / "outputs" / "xai" / "shap"
LIME_OUTPUT_DIR = BASE_DIR / "outputs" / "xai" / "lime"
XAI_COMPARISON_OUTPUT_DIR = BASE_DIR / "outputs" / "xai" / "comparison"

EVALUATION_PLOT_DIR = BASE_DIR / "outputs" / "plots" / "evaluation"
XAI_PLOT_DIR = BASE_DIR / "outputs" / "plots" / "xai"

CLASS_ID_TO_NAME = {
    0: "normal",
    1: "suspicious",
    2: "attack",
}
CLASS_NAMES = ["normal", "suspicious", "attack"]

RANDOM_STATE = 42
SHAP_BACKGROUND_SIZE = 50
SHAP_EXPLAIN_SAMPLE_SIZE = 30
LIME_SAMPLE_SIZE = 10


def print_step(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_folders() -> None:
    for folder in [
        EVALUATION_OUTPUT_DIR,
        SHAP_OUTPUT_DIR,
        LIME_OUTPUT_DIR,
        XAI_COMPARISON_OUTPUT_DIR,
        EVALUATION_PLOT_DIR,
        XAI_PLOT_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_trained_model():
    model_path = TRAINED_MODEL_DIR / "cnn_gru_intrusion_model.keras"
    return tf.keras.models.load_model(model_path)


def load_test_artifacts():
    X_test = np.load(SEQUENCE_DATA_DIR / "X_test_sequences.npy")
    y_test = np.load(SEQUENCE_DATA_DIR / "y_test_labels.npy")
    test_sequence_metadata = pd.read_csv(SEQUENCE_DATA_DIR / "test_sequence_metadata.csv")

    with open(SEQUENCE_DATA_DIR / "sequence_feature_columns.json", "r", encoding="utf-8") as handle:
        feature_columns_payload = json.load(handle)

    feature_columns = feature_columns_payload["feature_columns"]
    return X_test, y_test, test_sequence_metadata, feature_columns


def predict_test_probabilities(model, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test, verbose=0)


def convert_probabilities_to_labels(predicted_probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(predicted_probabilities, axis=1)


def evaluate_three_class_performance(y_test: np.ndarray, predicted_labels: np.ndarray):
    report_dict = classification_report(
        y_test,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, predicted_labels, labels=[0, 1, 2])
    return report_dict, report_text, confusion


def evaluate_binary_performance(y_test: np.ndarray, predicted_labels: np.ndarray):
    true_binary = np.where(y_test == 0, 0, 1)
    predicted_binary = np.where(predicted_labels == 0, 0, 1)

    report_dict = classification_report(
        true_binary,
        predicted_binary,
        labels=[0, 1],
        target_names=["normal", "attack_like"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        true_binary,
        predicted_binary,
        labels=[0, 1],
        target_names=["normal", "attack_like"],
        digits=4,
        zero_division=0,
    )
    confusion = confusion_matrix(true_binary, predicted_binary, labels=[0, 1])
    return report_dict, report_text, confusion


def save_evaluation_outputs(
    prediction_table: pd.DataFrame,
    metrics_3class: dict,
    metrics_binary: dict,
    report_3class_text: str,
    report_binary_text: str,
) -> None:
    prediction_table.to_csv(EVALUATION_OUTPUT_DIR / "test_predictions.csv", index=False)

    save_json(metrics_3class, EVALUATION_OUTPUT_DIR / "metrics_3class.json")
    save_json(metrics_binary, EVALUATION_OUTPUT_DIR / "metrics_binary.json")

    with open(EVALUATION_OUTPUT_DIR / "classification_report_3class.txt", "w", encoding="utf-8") as handle:
        handle.write(report_3class_text)

    with open(EVALUATION_OUTPUT_DIR / "classification_report_binary.txt", "w", encoding="utf-8") as handle:
        handle.write(report_binary_text)


def create_confusion_matrix_plot(confusion: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_positions = np.arange(len(labels))
    plt.xticks(tick_positions, labels, rotation=45)
    plt.yticks(tick_positions, labels)

    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            plt.text(column_index, row_index, int(confusion[row_index, column_index]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_evaluation_plots(
    y_test: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    confusion_3class: np.ndarray,
    confusion_binary: np.ndarray,
) -> None:
    create_confusion_matrix_plot(
        confusion_3class,
        CLASS_NAMES,
        EVALUATION_PLOT_DIR / "confusion_matrix_3class.png",
        "3-class confusion matrix",
    )

    create_confusion_matrix_plot(
        confusion_binary,
        ["normal", "attack_like"],
        EVALUATION_PLOT_DIR / "confusion_matrix_binary.png",
        "Binary confusion matrix",
    )

    true_binary = np.where(y_test == 0, 0, 1)
    attack_like_score = predicted_probabilities[:, 1] + predicted_probabilities[:, 2]

    fpr, tpr, _ = roc_curve(true_binary, attack_like_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Binary ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EVALUATION_PLOT_DIR / "roc_curve_binary.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(true_binary, attack_like_score)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Binary precision-recall curve")
    plt.tight_layout()
    plt.savefig(EVALUATION_PLOT_DIR / "precision_recall_curve_binary.png", dpi=150)
    plt.close()


def flatten_sequences_for_xai(X_sequences: np.ndarray, feature_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    sequence_length = X_sequences.shape[1]
    flat_feature_names = []
    for time_index in range(sequence_length):
        for feature_name in feature_columns:
            flat_feature_names.append(f"t{time_index}_{feature_name}")
    flattened = X_sequences.reshape(X_sequences.shape[0], -1)
    return flattened, flat_feature_names


def generate_shap_explanations(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    prediction_table: pd.DataFrame,
):
    flattened_test, flat_feature_names = flatten_sequences_for_xai(X_test, feature_columns)
    input_shape = X_test.shape[1:]

    def predict_from_flattened_sequences(flattened_input: np.ndarray) -> np.ndarray:
        reshaped_input = flattened_input.reshape((-1, input_shape[0], input_shape[1]))
        return model.predict(reshaped_input, verbose=0)

    np.random.seed(RANDOM_STATE)

    background_size = min(SHAP_BACKGROUND_SIZE, len(flattened_test))
    explain_size = min(SHAP_EXPLAIN_SAMPLE_SIZE, len(flattened_test))

    background_indices = np.random.choice(len(flattened_test), size=background_size, replace=False)
    explain_indices = np.random.choice(len(flattened_test), size=explain_size, replace=False)

    background_data = flattened_test[background_indices]
    explain_data = flattened_test[explain_indices]

    explainer = shap.KernelExplainer(predict_from_flattened_sequences, background_data)
    shap_values = explainer.shap_values(explain_data, nsamples=100)

    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=0)  # (classes, samples, features)
        shap_array = np.transpose(shap_array, (1, 2, 0))  # (samples, features, classes)
    else:
        shap_array = shap_values  # expected (samples, features, classes)

    predicted_labels_for_explained = prediction_table.iloc[explain_indices]["predicted_label_id"].values

    global_rows = []
    for flat_feature_index, flat_feature_name in enumerate(flat_feature_names):
        mean_abs_value = float(np.mean(np.abs(shap_array[:, flat_feature_index, :])))
        global_rows.append(
            {
                "flat_feature": flat_feature_name,
                "mean_absolute_shap_value": mean_abs_value,
            }
        )

    global_importance_table = pd.DataFrame(global_rows).sort_values(
        "mean_absolute_shap_value",
        ascending=False,
    ).reset_index(drop=True)

    global_importance_table.to_csv(
        SHAP_OUTPUT_DIR / "shap_global_feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    top_global = global_importance_table.head(20).iloc[::-1]
    plt.barh(top_global["flat_feature"], top_global["mean_absolute_shap_value"])
    plt.title("Top 20 SHAP global features")
    plt.tight_layout()
    plt.savefig(XAI_PLOT_DIR / "shap_global_feature_importance.png", dpi=150)
    plt.close()

    local_rows = []
    local_json_payload = {}

    if explain_size > 0:
        first_local_index = 0
        sample_row_index = int(explain_indices[first_local_index])
        sample_predicted_class = int(predicted_labels_for_explained[first_local_index])
        local_feature_values = explain_data[first_local_index]
        local_shap_values = shap_array[first_local_index, :, sample_predicted_class]

        local_table = pd.DataFrame(
            {
                "flat_feature": flat_feature_names,
                "input_value": local_feature_values,
                "shap_value_for_predicted_class": local_shap_values,
                "absolute_shap_value": np.abs(local_shap_values),
            }
        ).sort_values("absolute_shap_value", ascending=False)

        local_table.to_csv(
            SHAP_OUTPUT_DIR / "shap_local_explanation_sample_001.csv",
            index=False,
        )

        local_json_payload = {
            "sample_row_index": sample_row_index,
            "predicted_label_id": sample_predicted_class,
            "predicted_label_name": CLASS_ID_TO_NAME[sample_predicted_class],
            "top_features": local_table.head(15).to_dict(orient="records"),
        }
        save_json(local_json_payload, SHAP_OUTPUT_DIR / "shap_local_explanation_sample_001.json")

        plt.figure(figsize=(10, 6))
        top_local = local_table.head(15).iloc[::-1]
        plt.barh(top_local["flat_feature"], top_local["shap_value_for_predicted_class"])
        plt.title("SHAP local explanation sample 001")
        plt.tight_layout()
        plt.savefig(XAI_PLOT_DIR / "shap_local_explanation_sample_001.png", dpi=150)
        plt.close()

        for rank_index, row in local_table.head(15).reset_index(drop=True).iterrows():
            local_rows.append(
                {
                    "sample_row_index": sample_row_index,
                    "predicted_label_id": sample_predicted_class,
                    "predicted_label_name": CLASS_ID_TO_NAME[sample_predicted_class],
                    "rank": rank_index + 1,
                    "flat_feature": row["flat_feature"],
                    "shap_value": float(row["shap_value_for_predicted_class"]),
                    "absolute_shap_value": float(row["absolute_shap_value"]),
                }
            )

    shap_local_summary_table = pd.DataFrame(local_rows)
    shap_local_summary_table.to_csv(SHAP_OUTPUT_DIR / "shap_local_summary.csv", index=False)

    explained_samples_table = pd.DataFrame(
        {
            "sample_row_index": explain_indices,
            "true_label_id": y_test[explain_indices],
            "predicted_label_id": prediction_table.iloc[explain_indices]["predicted_label_id"].values,
            "predicted_label_name": prediction_table.iloc[explain_indices]["predicted_label_name"].values,
        }
    )
    explained_samples_table.to_csv(SHAP_OUTPUT_DIR / "shap_explained_samples.csv", index=False)

    return {
        "global_importance_table": global_importance_table,
        "local_summary_table": shap_local_summary_table,
        "explained_samples_table": explained_samples_table,
    }


def generate_lime_explanations(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    prediction_table: pd.DataFrame,
):
    flattened_test, flat_feature_names = flatten_sequences_for_xai(X_test, feature_columns)
    input_shape = X_test.shape[1:]

    def predict_from_flattened_sequences(flattened_input: np.ndarray) -> np.ndarray:
        reshaped_input = flattened_input.reshape((-1, input_shape[0], input_shape[1]))
        return model.predict(reshaped_input, verbose=0)

    suspicious_or_attack_indices = prediction_table.index[
        prediction_table["predicted_label_name"].isin(["suspicious", "attack"])
    ].tolist()

    np.random.seed(RANDOM_STATE)
    if suspicious_or_attack_indices:
        selected_indices = np.random.choice(
            suspicious_or_attack_indices,
            size=min(LIME_SAMPLE_SIZE, len(suspicious_or_attack_indices)),
            replace=False,
        )
    else:
        selected_indices = np.array([], dtype=int)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=flattened_test,
        feature_names=flat_feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    summary_rows = []

    for case_number, sample_row_index in enumerate(selected_indices, start=1):
        local_instance = flattened_test[int(sample_row_index)]
        predicted_class = int(prediction_table.iloc[int(sample_row_index)]["predicted_label_id"])

        explanation = explainer.explain_instance(
            data_row=local_instance,
            predict_fn=predict_from_flattened_sequences,
            num_features=15,
            top_labels=3,
        )

        local_items = explanation.as_list(label=predicted_class)

        local_table = pd.DataFrame(
            {
                "condition_or_feature": [item[0] for item in local_items],
                "lime_weight": [float(item[1]) for item in local_items],
                "absolute_lime_weight": [abs(float(item[1])) for item in local_items],
            }
        ).sort_values("absolute_lime_weight", ascending=False)

        local_csv_path = LIME_OUTPUT_DIR / f"lime_local_explanation_case_{case_number:03d}.csv"
        local_json_path = LIME_OUTPUT_DIR / f"lime_local_explanation_case_{case_number:03d}.json"

        local_table.to_csv(local_csv_path, index=False)
        save_json(
            {
                "sample_row_index": int(sample_row_index),
                "predicted_label_id": predicted_class,
                "predicted_label_name": CLASS_ID_TO_NAME[predicted_class],
                "top_features": local_table.head(15).to_dict(orient="records"),
            },
            local_json_path,
        )

        plt.figure(figsize=(10, 6))
        top_local = local_table.head(15).iloc[::-1]
        plt.barh(top_local["condition_or_feature"], top_local["lime_weight"])
        plt.title(f"LIME local explanation case {case_number:03d}")
        plt.tight_layout()
        plt.savefig(XAI_PLOT_DIR / f"lime_local_explanation_case_{case_number:03d}.png", dpi=150)
        plt.close()

        for rank_index, row in local_table.head(15).reset_index(drop=True).iterrows():
            summary_rows.append(
                {
                    "sample_row_index": int(sample_row_index),
                    "predicted_label_id": predicted_class,
                    "predicted_label_name": CLASS_ID_TO_NAME[predicted_class],
                    "rank": rank_index + 1,
                    "condition_or_feature": row["condition_or_feature"],
                    "lime_weight": float(row["lime_weight"]),
                    "absolute_lime_weight": float(row["absolute_lime_weight"]),
                }
            )

    lime_summary_table = pd.DataFrame(summary_rows)
    lime_summary_table.to_csv(LIME_OUTPUT_DIR / "lime_explanation_summary.csv", index=False)

    return {
        "lime_summary_table": lime_summary_table,
    }


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


def compare_xai_methods(
    shap_results: dict,
    lime_results: dict,
    prediction_table: pd.DataFrame,
):
    shap_local_table = shap_results["local_summary_table"].copy()
    lime_summary_table = lime_results["lime_summary_table"].copy()

    if not lime_summary_table.empty:
        lime_summary_table["normalized_feature"] = lime_summary_table["condition_or_feature"].apply(_normalize_lime_feature_name)
    else:
        lime_summary_table["normalized_feature"] = []

    case_level_rows = []

    shap_case_ids = set(shap_local_table["sample_row_index"].unique()) if not shap_local_table.empty else set()
    lime_case_ids = set(lime_summary_table["sample_row_index"].unique()) if not lime_summary_table.empty else set()
    common_case_ids = sorted(shap_case_ids.intersection(lime_case_ids))

    print(f"[INFO] Common SHAP/LIME local cases available: {len(common_case_ids)}")

    for case_id in common_case_ids:
        shap_case = shap_local_table[shap_local_table["sample_row_index"] == case_id].copy()
        lime_case = lime_summary_table[lime_summary_table["sample_row_index"] == case_id].copy()

        shap_features = set(shap_case["flat_feature"].astype(str).tolist())
        lime_features = set(lime_case["normalized_feature"].astype(str).tolist())

        if not shap_features and not lime_features:
            continue

        intersection = shap_features.intersection(lime_features)
        union = shap_features.union(lime_features)
        jaccard_similarity = (len(intersection) / len(union)) if union else 0.0

        case_level_rows.append(
            {
                "sample_row_index": int(case_id),
                "predicted_label_name": str(prediction_table.iloc[int(case_id)]["predicted_label_name"]),
                "shap_feature_count": len(shap_features),
                "lime_feature_count": len(lime_features),
                "shared_feature_count": len(intersection),
                "jaccard_similarity": jaccard_similarity,
                "shared_features": ", ".join(sorted(intersection)),
            }
        )

    case_level_table = pd.DataFrame(case_level_rows)
    case_level_table.to_csv(XAI_COMPARISON_OUTPUT_DIR / "xai_case_level_comparison.csv", index=False)

    mean_jaccard_similarity = (
        float(case_level_table["jaccard_similarity"].mean())
        if not case_level_table.empty
        else None
    )

    method_summary_table = pd.DataFrame(
        [
            {
                "criterion": "global interpretability",
                "shap": "high",
                "lime": "low",
                "recommended_method": "SHAP",
            },
            {
                "criterion": "local explanation",
                "shap": "good",
                "lime": "high",
                "recommended_method": "LIME",
            },
            {
                "criterion": "same-case overlap available",
                "shap": len(shap_case_ids),
                "lime": len(lime_case_ids),
                "recommended_method": "compare when shared cases exist",
            },
            {
                "criterion": "mean jaccard similarity",
                "shap": mean_jaccard_similarity,
                "lime": mean_jaccard_similarity,
                "recommended_method": "higher is better",
            },
        ]
    )
    method_summary_table.to_csv(
        XAI_COMPARISON_OUTPUT_DIR / "xai_method_comparison_summary.csv",
        index=False,
    )

    with open(XAI_COMPARISON_OUTPUT_DIR / "xai_comparison_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("XAI comparison summary\n")
        handle.write("======================\n")
        handle.write(f"SHAP local cases: {len(shap_case_ids)}\n")
        handle.write(f"LIME local cases: {len(lime_case_ids)}\n")
        handle.write(f"Common local cases: {len(common_case_ids)}\n")
        handle.write(f"Mean Jaccard similarity: {mean_jaccard_similarity}\n")
        if len(common_case_ids) == 0:
            handle.write(
                "\nNote: direct same-case SHAP vs LIME local comparison was not possible in this run "
                "because both methods did not explain the same sample rows.\n"
            )

    if not case_level_table.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(case_level_table["jaccard_similarity"], bins=10)
        plt.title("Distribution of SHAP vs LIME Jaccard similarity")
        plt.tight_layout()
        plt.savefig(XAI_PLOT_DIR / "xai_jaccard_similarity_histogram.png", dpi=150)
        plt.close()

    return {
        "case_level_table": case_level_table,
        "method_summary_table": method_summary_table,
    }


def main() -> None:
    create_folders()

    print_step("[STEP 1] Load trained model and test artifacts")
    model = load_trained_model()
    X_test, y_test, test_sequence_metadata, feature_columns = load_test_artifacts()

    print_step("[STEP 2] Evaluate model on test data")
    predicted_probabilities = predict_test_probabilities(model, X_test)
    predicted_labels = convert_probabilities_to_labels(predicted_probabilities)

    metrics_3class, report_3class_text, confusion_3class = evaluate_three_class_performance(y_test, predicted_labels)
    metrics_binary, report_binary_text, confusion_binary = evaluate_binary_performance(y_test, predicted_labels)

    prediction_table = test_sequence_metadata.copy()
    prediction_table["true_label_id"] = y_test
    prediction_table["true_label_name"] = [CLASS_ID_TO_NAME[int(x)] for x in y_test]
    prediction_table["predicted_label_id"] = predicted_labels
    prediction_table["predicted_label_name"] = [CLASS_ID_TO_NAME[int(x)] for x in predicted_labels]
    prediction_table["probability_normal"] = predicted_probabilities[:, 0]
    prediction_table["probability_suspicious"] = predicted_probabilities[:, 1]
    prediction_table["probability_attack"] = predicted_probabilities[:, 2]

    save_evaluation_outputs(
        prediction_table,
        metrics_3class,
        metrics_binary,
        report_3class_text,
        report_binary_text,
    )
    create_evaluation_plots(
        y_test,
        predicted_labels,
        predicted_probabilities,
        confusion_3class,
        confusion_binary,
    )

    print_step("[STEP 3] Generate SHAP explanations")
    shap_results = generate_shap_explanations(
        model,
        X_test,
        y_test,
        feature_columns,
        prediction_table,
    )

    print_step("[STEP 4] Generate LIME explanations")
    lime_results = generate_lime_explanations(
        model,
        X_test,
        y_test,
        feature_columns,
        prediction_table,
    )

    print_step("[STEP 5] Compare SHAP and LIME")
    compare_xai_methods(
        shap_results,
        lime_results,
        prediction_table,
    )

    print_step("[DONE]")
    print("[INFO] Model evaluation and XAI generation completed successfully.")


if __name__ == "__main__":
    main()