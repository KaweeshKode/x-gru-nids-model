import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score


BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PSEUDO_LABEL_DATA_DIR = BASE_DIR / "data" / "pseudo_labels"

PSEUDO_LABEL_MODEL_DIR = BASE_DIR / "models" / "pseudo_labels"

PSEUDO_LABEL_OUTPUT_DIR = BASE_DIR / "outputs" / "pseudo_labels"
PSEUDO_LABEL_PLOT_DIR = BASE_DIR / "outputs" / "plots" / "pseudo_labels"

NUMBER_OF_CLUSTERS = 3
RANDOM_STATE = 42
NUMBER_OF_INITIALIZATIONS = 20
SILHOUETTE_SAMPLE_SIZE = 20000

RISK_SCORE_WEIGHTS = {
    "distance_norm": 0.50,
    "ttl_gap_norm": 0.20,
    "byte_ratio_norm": 0.10,
    "pkt_ratio_norm": 0.10,
    "load_total_norm": 0.10,
}

LOW_ATTACK_RATE = 0.10
HIGH_ATTACK_RATE = 0.60
NUMBER_OF_CALIBRATION_BINS = 100

PSEUDO_LABEL_TO_ID = {
    "normal": 0,
    "suspicious": 1,
    "attack": 2,
}


def print_step(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_folders() -> None:
    for folder in [
        PSEUDO_LABEL_DATA_DIR,
        PSEUDO_LABEL_MODEL_DIR,
        PSEUDO_LABEL_OUTPUT_DIR,
        PSEUDO_LABEL_PLOT_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_processed_datasets():
    train_features = pd.read_csv(PROCESSED_DATA_DIR / "train_processed.csv")
    validation_features = pd.read_csv(PROCESSED_DATA_DIR / "validation_processed.csv")
    test_features = pd.read_csv(PROCESSED_DATA_DIR / "test_processed.csv")

    train_metadata = pd.read_csv(PROCESSED_DATA_DIR / "train_metadata.csv")
    validation_metadata = pd.read_csv(PROCESSED_DATA_DIR / "validation_metadata.csv")
    test_metadata = pd.read_csv(PROCESSED_DATA_DIR / "test_metadata.csv")

    return (
        train_features,
        validation_features,
        test_features,
        train_metadata,
        validation_metadata,
        test_metadata,
    )


def train_kmeans_model(train_features: pd.DataFrame) -> KMeans:
    model = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        init="k-means++",
        n_init=NUMBER_OF_INITIALIZATIONS,
        random_state=RANDOM_STATE,
    )
    model.fit(train_features)
    return model


def calculate_cluster_features(model: KMeans, features: pd.DataFrame):
    all_distances = model.transform(features)
    assigned_clusters = np.argmin(all_distances, axis=1)
    centroid_distances = all_distances[np.arange(len(features)), assigned_clusters]
    return assigned_clusters, centroid_distances, all_distances


def clamp_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).fillna(0)


def normalize_score_component(
    train_series: pd.Series,
    other_series: pd.Series | None = None,
):
    train_series = clamp_series(train_series.astype(float))
    minimum = float(train_series.min())
    maximum = float(train_series.max())
    denominator = max(maximum - minimum, 1e-9)

    train_normalized = ((train_series - minimum) / denominator).clip(0, 1)

    if other_series is None:
        return train_normalized, {"min": minimum, "max": maximum}

    other_series = clamp_series(other_series.astype(float))
    other_normalized = ((other_series - minimum) / denominator).clip(0, 1)

    return train_normalized, other_normalized, {"min": minimum, "max": maximum}


def build_proxy_feature(
    feature_table: pd.DataFrame,
    column_name: str,
    fallback_value: float = 0.0,
) -> pd.Series:
    if column_name not in feature_table.columns:
        return pd.Series(np.full(len(feature_table), fallback_value), index=feature_table.index)
    return clamp_series(feature_table[column_name]).abs()


def build_risk_score_components(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_distances: np.ndarray,
    validation_distances: np.ndarray,
    test_distances: np.ndarray,
):
    train_components: dict[str, pd.Series] = {}
    validation_components: dict[str, pd.Series] = {}
    test_components: dict[str, pd.Series] = {}
    component_ranges: dict[str, dict[str, float]] = {}

    component_definitions = {
        "distance_norm": (
            pd.Series(train_distances, index=train_features.index),
            pd.Series(validation_distances, index=validation_features.index),
            pd.Series(test_distances, index=test_features.index),
        ),
        "ttl_gap_norm": (
            build_proxy_feature(train_features, "ttl_gap"),
            build_proxy_feature(validation_features, "ttl_gap"),
            build_proxy_feature(test_features, "ttl_gap"),
        ),
        "byte_ratio_norm": (
            build_proxy_feature(train_features, "byte_ratio"),
            build_proxy_feature(validation_features, "byte_ratio"),
            build_proxy_feature(test_features, "byte_ratio"),
        ),
        "pkt_ratio_norm": (
            build_proxy_feature(train_features, "pkt_ratio"),
            build_proxy_feature(validation_features, "pkt_ratio"),
            build_proxy_feature(test_features, "pkt_ratio"),
        ),
        "load_total_norm": (
            build_proxy_feature(train_features, "load_total"),
            build_proxy_feature(validation_features, "load_total"),
            build_proxy_feature(test_features, "load_total"),
        ),
    }

    for component_name, (train_series, validation_series, test_series) in component_definitions.items():
        (
            train_components[component_name],
            validation_components[component_name],
            component_range,
        ) = normalize_score_component(train_series, validation_series)

        _, test_components[component_name], _ = normalize_score_component(train_series, test_series)
        component_ranges[component_name] = component_range

    return train_components, validation_components, test_components, component_ranges


def calculate_hybrid_risk_score(score_components: dict[str, pd.Series]) -> np.ndarray:
    score = np.zeros(len(next(iter(score_components.values()))), dtype=float)
    for component_name, component_weight in RISK_SCORE_WEIGHTS.items():
        score += component_weight * score_components[component_name].values
    return score


def calibrate_label_thresholds(
    train_scores: np.ndarray,
    true_binary_labels: pd.Series,
):
    calibration_table = pd.DataFrame(
        {
            "score": train_scores,
            "label": true_binary_labels.astype(int).values,
        }
    ).sort_values("score").reset_index(drop=True)

    min_score = float(calibration_table["score"].min())
    max_score = float(calibration_table["score"].max())

    if max_score - min_score < 1e-12:
        lower_threshold = float(np.quantile(train_scores, 0.80))
        upper_threshold = float(np.quantile(train_scores, 0.95))
        return lower_threshold, upper_threshold, pd.DataFrame()

    bins = np.linspace(min_score, max_score, NUMBER_OF_CALIBRATION_BINS + 1)
    calibration_table["bin"] = pd.cut(
        calibration_table["score"],
        bins=bins,
        include_lowest=True,
        labels=False,
    )

    bin_statistics = (
        calibration_table.groupby("bin", observed=False)
        .agg(
            count=("score", "size"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            attack_rate=("label", "mean"),
        )
        .reset_index()
    )

    bin_statistics = bin_statistics[bin_statistics["count"] > 0].copy().reset_index(drop=True)

    normal_bins = bin_statistics[bin_statistics["attack_rate"] <= LOW_ATTACK_RATE]
    attack_bins = bin_statistics[bin_statistics["attack_rate"] >= HIGH_ATTACK_RATE]

    lower_threshold = (
        float(normal_bins["max_score"].max())
        if not normal_bins.empty
        else float(np.quantile(train_scores, 0.80))
    )
    upper_threshold = (
        float(attack_bins["min_score"].min())
        if not attack_bins.empty
        else float(np.quantile(train_scores, 0.95))
    )

    if not (lower_threshold < upper_threshold):
        lower_threshold = float(np.quantile(train_scores, 0.80))
        upper_threshold = float(np.quantile(train_scores, 0.95))

    return lower_threshold, upper_threshold, bin_statistics


def assign_pseudo_labels(
    scores: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
):
    label_names = np.empty(len(scores), dtype=object)
    label_names[scores < lower_threshold] = "normal"
    label_names[(scores >= lower_threshold) & (scores < upper_threshold)] = "suspicious"
    label_names[scores >= upper_threshold] = "attack"

    label_ids = np.array(
        [PSEUDO_LABEL_TO_ID[label_name] for label_name in label_names],
        dtype=int,
    )
    return label_names, label_ids


def evaluate_pseudo_labels(
    split_name: str,
    true_binary_labels: pd.Series,
    pseudo_label_names: np.ndarray,
) -> None:
    predicted_binary = np.where(pseudo_label_names == "normal", 0, 1)

    confusion = confusion_matrix(true_binary_labels, predicted_binary, labels=[0, 1])
    print(f"\n[CHECK] {split_name} binary sanity check")
    print(confusion)

    report = classification_report(
        true_binary_labels,
        predicted_binary,
        labels=[0, 1],
        target_names=["true_normal", "true_attack"],
        digits=4,
        zero_division=0,
    )
    print(report)


def build_pseudo_label_output(
    metadata_table: pd.DataFrame,
    assigned_clusters: np.ndarray,
    centroid_distances: np.ndarray,
    hybrid_risk_scores: np.ndarray,
    pseudo_label_names: np.ndarray,
    pseudo_label_ids: np.ndarray,
) -> pd.DataFrame:
    labeled_table = metadata_table.copy()
    labeled_table["cluster_id"] = assigned_clusters
    labeled_table["distance_to_centroid"] = centroid_distances
    labeled_table["hybrid_risk_score"] = hybrid_risk_scores
    labeled_table["pseudo_label"] = pseudo_label_names
    labeled_table["pseudo_label_id"] = pseudo_label_ids
    return labeled_table


def save_pseudo_label_outputs(
    train_labels: pd.DataFrame,
    validation_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    model: KMeans,
    threshold_config: dict,
    component_ranges: dict,
    calibration_bin_statistics: pd.DataFrame,
) -> None:
    train_labels.to_csv(PSEUDO_LABEL_DATA_DIR / "train_pseudo_labeled.csv", index=False)
    validation_labels.to_csv(PSEUDO_LABEL_DATA_DIR / "validation_pseudo_labeled.csv", index=False)
    test_labels.to_csv(PSEUDO_LABEL_DATA_DIR / "test_pseudo_labeled.csv", index=False)

    with open(PSEUDO_LABEL_MODEL_DIR / "kmeans_model.pkl", "wb") as handle:
        pickle.dump(model, handle)

    save_json(
        {
            "number_of_clusters": NUMBER_OF_CLUSTERS,
            "risk_score_weights": RISK_SCORE_WEIGHTS,
            "threshold_config": threshold_config,
            "component_ranges": component_ranges,
        },
        PSEUDO_LABEL_MODEL_DIR / "pseudo_label_config.json",
    )

    if not calibration_bin_statistics.empty:
        calibration_bin_statistics.to_csv(
            PSEUDO_LABEL_OUTPUT_DIR / "train_calibration_bins.csv",
            index=False,
        )


def create_pseudo_label_plots(
    train_labels: pd.DataFrame,
    validation_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    train_features: pd.DataFrame,
    train_clusters: np.ndarray,
) -> None:
    for split_name, labeled_table in [
        ("train", train_labels),
        ("validation", validation_labels),
        ("test", test_labels),
    ]:
        label_counts = labeled_table["pseudo_label"].value_counts()

        plt.figure(figsize=(7, 5))
        plt.bar(label_counts.index.astype(str), label_counts.values)
        plt.title(f"{split_name.title()} pseudo-label distribution")
        plt.tight_layout()
        plt.savefig(PSEUDO_LABEL_PLOT_DIR / f"{split_name}_pseudo_label_distribution.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.hist(labeled_table["hybrid_risk_score"], bins=50)
        plt.title(f"{split_name.title()} hybrid risk score distribution")
        plt.tight_layout()
        plt.savefig(PSEUDO_LABEL_PLOT_DIR / f"{split_name}_risk_score_histogram.png", dpi=150)
        plt.close()

    if len(train_features.columns) >= 2:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        reduced_values = pca.fit_transform(train_features)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_values[:, 0], reduced_values[:, 1], c=train_clusters, s=3)
        plt.title("Train clusters in PCA 2D space")
        plt.tight_layout()
        plt.savefig(PSEUDO_LABEL_PLOT_DIR / "train_cluster_pca_scatter.png", dpi=150)
        plt.close()


def print_pseudo_label_summary(
    split_name: str,
    assigned_clusters: np.ndarray,
    labels: np.ndarray,
) -> None:
    print(f"[INFO] {split_name} cluster counts: {pd.Series(assigned_clusters).value_counts().sort_index().to_dict()}")
    print(f"[INFO] {split_name} pseudo-label counts: {pd.Series(labels).value_counts().to_dict()}")


def main() -> None:
    create_folders()

    print_step("[STEP 1] Load processed datasets")
    (
        train_features,
        validation_features,
        test_features,
        train_metadata,
        validation_metadata,
        test_metadata,
    ) = load_processed_datasets()

    print_step("[STEP 2] Train KMeans model")
    kmeans_model = train_kmeans_model(train_features)

    print_step("[STEP 3] Calculate cluster assignments and distances")
    train_clusters, train_distances, _ = calculate_cluster_features(kmeans_model, train_features)
    validation_clusters, validation_distances, _ = calculate_cluster_features(kmeans_model, validation_features)
    test_clusters, test_distances, _ = calculate_cluster_features(kmeans_model, test_features)

    if len(np.unique(train_clusters)) > 1:
        sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(train_features))
        train_silhouette = silhouette_score(
            train_features,
            train_clusters,
            sample_size=sample_size,
            random_state=RANDOM_STATE,
        )
        print(f"[INFO] Train silhouette score: {train_silhouette:.4f}")

    print_step("[STEP 4] Build hybrid risk score")
    (
        train_components,
        validation_components,
        test_components,
        component_ranges,
    ) = build_risk_score_components(
        train_features,
        validation_features,
        test_features,
        train_distances,
        validation_distances,
        test_distances,
    )

    train_scores = calculate_hybrid_risk_score(train_components)
    validation_scores = calculate_hybrid_risk_score(validation_components)
    test_scores = calculate_hybrid_risk_score(test_components)

    print_step("[STEP 5] Calibrate thresholds on train split")
    lower_threshold, upper_threshold, calibration_bin_statistics = calibrate_label_thresholds(
        train_scores,
        train_metadata["label"],
    )

    print_step("[STEP 6] Generate pseudo labels")
    train_label_names, train_label_ids = assign_pseudo_labels(train_scores, lower_threshold, upper_threshold)
    validation_label_names, validation_label_ids = assign_pseudo_labels(validation_scores, lower_threshold, upper_threshold)
    test_label_names, test_label_ids = assign_pseudo_labels(test_scores, lower_threshold, upper_threshold)

    print_pseudo_label_summary("Train", train_clusters, train_label_names)
    print_pseudo_label_summary("Validation", validation_clusters, validation_label_names)
    print_pseudo_label_summary("Test", test_clusters, test_label_names)

    print_step("[STEP 7] Evaluate pseudo labels")
    evaluate_pseudo_labels("Train", train_metadata["label"], train_label_names)
    evaluate_pseudo_labels("Validation", validation_metadata["label"], validation_label_names)
    evaluate_pseudo_labels("Test", test_metadata["label"], test_label_names)

    train_pseudo_labels = build_pseudo_label_output(
        train_metadata,
        train_clusters,
        train_distances,
        train_scores,
        train_label_names,
        train_label_ids,
    )
    validation_pseudo_labels = build_pseudo_label_output(
        validation_metadata,
        validation_clusters,
        validation_distances,
        validation_scores,
        validation_label_names,
        validation_label_ids,
    )
    test_pseudo_labels = build_pseudo_label_output(
        test_metadata,
        test_clusters,
        test_distances,
        test_scores,
        test_label_names,
        test_label_ids,
    )

    print_step("[STEP 8] Save pseudo label outputs")
    threshold_config = {
        "lower_threshold_normal_to_suspicious": lower_threshold,
        "upper_threshold_suspicious_to_attack": upper_threshold,
    }

    save_pseudo_label_outputs(
        train_pseudo_labels,
        validation_pseudo_labels,
        test_pseudo_labels,
        kmeans_model,
        threshold_config,
        component_ranges,
        calibration_bin_statistics,
    )

    create_pseudo_label_plots(
        train_pseudo_labels,
        validation_pseudo_labels,
        test_pseudo_labels,
        train_features,
        train_clusters,
    )

    with open(PSEUDO_LABEL_OUTPUT_DIR / "pseudo_label_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Pseudo-label generation completed successfully.\n")
        handle.write(f"Lower threshold: {lower_threshold:.6f}\n")
        handle.write(f"Upper threshold: {upper_threshold:.6f}\n")

    print_step("[DONE]")
    print("[INFO] Pseudo label generation completed successfully.")


if __name__ == "__main__":
    main()