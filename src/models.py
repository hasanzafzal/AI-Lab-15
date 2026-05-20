import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

from utils import (
    PROCESSED_DATASET, MODELS_DIR, RESULTS_DIR,
    DECISION_TREE_MODEL, KMEANS_MODEL, LINEAR_REG_MODEL,
    ALL_FEATURES, SOIL_FEATURES, CLIMATE_FEATURES,
    TARGET_CLASSIFICATION, TARGET_REGRESSION,
    ensure_dirs, save_metrics, setup_logger
)

logger = setup_logger("models")

# Global plot style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# MODULE 1: Decision Tree Classifier
def train_decision_tree(df: pd.DataFrame, test_size: float = 0.2,
                        random_state: int = 42, max_depth: int = 10) -> dict:
    """
    Train a Decision Tree Classifier for crop recommendation.

    Parameters:
        df: Processed DataFrame with features and crop labels.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.
        max_depth: Maximum depth of the decision tree.

    Returns:
        Dictionary containing the model, metrics, and artifact paths.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Decision Tree Classifier — Crop Recommendation")
    logger.info("=" * 60)

    X = df[ALL_FEATURES].values
    y = df[TARGET_CLASSIFICATION].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")

    # Train the model
    dt_clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        criterion="gini"
    )
    dt_clf.fit(X_train, y_train)
    logger.info(f"Model trained (max_depth={max_depth}, "
                f"n_leaves={dt_clf.get_n_leaves()}, "
                f"tree_depth={dt_clf.get_depth()})")

    # Predictions
    y_pred = dt_clf.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")

    # Feature importance
    feature_importance = dict(zip(ALL_FEATURES, dt_clf.feature_importances_))
    logger.info(f"Feature Importance: {feature_importance}")

    # Save metrics
    metrics = {
        "model": "DecisionTreeClassifier",
        "task": "Crop Recommendation",
        "hyperparameters": {
            "max_depth": max_depth,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "criterion": "gini"
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(accuracy, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_weighted": round(f1, 4),
        "feature_importance": {k: round(v, 4) for k, v in feature_importance.items()},
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    save_metrics(metrics, "decision_tree_metrics.json")

    # ── Visualization 1: Feature Importance Bar Chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_features)
    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Decision Tree — Feature Importance for Crop Recommendation",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {fi_path}")

    # ── Visualization 2: Confusion Matrix (top-10 crops for readability) ──
    cm = confusion_matrix(y_test, y_pred, labels=dt_clf.classes_)
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=dt_clf.classes_, yticklabels=dt_clf.classes_, ax=ax2)
    ax2.set_xlabel("Predicted", fontsize=12)
    ax2.set_ylabel("Actual", fontsize=12)
    ax2.set_title("Decision Tree — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig2.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # Serialize model
    joblib.dump(dt_clf, DECISION_TREE_MODEL)
    logger.info(f"Model serialized to {DECISION_TREE_MODEL}")

    return {
        "model": dt_clf,
        "metrics": metrics,
        "plots": [fi_path, cm_path]
    }


# MODULE 2: K-Means Clustering (Soil Segmentation)
def train_kmeans_clustering(df: pd.DataFrame, n_clusters: int = 5,
                            random_state: int = 42) -> dict:
    """
    Train a K-Means clustering model for soil profile segmentation.

    Clusters are formed based on soil features (N, P, K, pH) to identify
    homogeneous farm zones with similar nutrient profiles.

    Parameters:
        df: Processed DataFrame with soil features.
        n_clusters: Number of clusters to form.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing the model, metrics, and artifact paths.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: K-Means Clustering — Soil Profile Segmentation")
    logger.info("=" * 60)

    X_soil = df[SOIL_FEATURES].values

    # Scale soil features independently for clustering
    soil_scaler = StandardScaler()
    X_scaled = soil_scaler.fit_transform(X_soil)

    # ── Elbow Method (find optimal k) ──
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        silhouette_scores.append(sil)
        logger.info(f"  k={k}: Inertia={km.inertia_:.2f}, Silhouette={sil:.4f}")

    # Train final model with chosen k
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    final_silhouette = silhouette_score(X_scaled, labels)

    logger.info(f"\nFinal model: k={n_clusters}, Silhouette Score={final_silhouette:.4f}")
    logger.info(f"Cluster distribution: {np.bincount(labels).tolist()}")

    # Add cluster labels to DataFrame
    df_clustered = df.copy()
    df_clustered["soil_cluster"] = labels

    # Cluster profiles (mean feature values per cluster)
    cluster_profiles = df_clustered.groupby("soil_cluster")[SOIL_FEATURES].mean().round(2)
    logger.info(f"\nCluster Profiles:\n{cluster_profiles}")

    # Agronomic guidance per cluster
    guidance = generate_cluster_guidance(cluster_profiles)

    # Save metrics
    metrics = {
        "model": "KMeans",
        "task": "Soil Profile Segmentation",
        "n_clusters": n_clusters,
        "silhouette_score": round(final_silhouette, 4),
        "inertia": round(kmeans.inertia_, 4),
        "cluster_sizes": np.bincount(labels).tolist(),
        "cluster_profiles": cluster_profiles.to_dict(),
        "agronomic_guidance": guidance,
        "elbow_data": {
            "k_values": list(K_range),
            "inertias": [round(i, 2) for i in inertias],
            "silhouette_scores": [round(s, 4) for s in silhouette_scores]
        }
    }
    save_metrics(metrics, "kmeans_metrics.json")

    # ── Visualization 1: Cluster Scatter (PCA 2D projection) ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis",
                         alpha=0.6, s=20, edgecolors="white", linewidth=0.3)

    # Plot centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c="red", marker="X", s=200, edgecolors="black", linewidth=1.5,
               label="Centroids", zorder=5)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Cluster ID", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("K-Means — Soil Profile Clusters (PCA Projection)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    cluster_path = os.path.join(RESULTS_DIR, "cluster_scatter.png")
    fig.savefig(cluster_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Cluster scatter plot saved to {cluster_path}")

    # ── Visualization 2: Elbow + Silhouette Plot ──
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    ax2a.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
    ax2a.axvline(x=n_clusters, color="red", linestyle="--", alpha=0.7, label=f"Chosen k={n_clusters}")
    ax2a.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2a.set_ylabel("Inertia (SSE)", fontsize=12)
    ax2a.set_title("Elbow Method", fontsize=13, fontweight="bold")
    ax2a.legend()

    ax2b.plot(list(K_range), silhouette_scores, "go-", linewidth=2, markersize=8)
    ax2b.axvline(x=n_clusters, color="red", linestyle="--", alpha=0.7, label=f"Chosen k={n_clusters}")
    ax2b.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2b.set_ylabel("Silhouette Score", fontsize=12)
    ax2b.set_title("Silhouette Analysis", fontsize=13, fontweight="bold")
    ax2b.legend()

    plt.tight_layout()
    elbow_path = os.path.join(RESULTS_DIR, "elbow_silhouette.png")
    fig2.savefig(elbow_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Elbow/silhouette plot saved to {elbow_path}")

    # ── Visualization 3: Cluster Profile Heatmap ──
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(cluster_profiles, annot=True, fmt=".1f", cmap="YlGn",
                linewidths=0.5, ax=ax3)
    ax3.set_title("Mean Feature Values per Soil Cluster",
                  fontsize=13, fontweight="bold")
    ax3.set_xlabel("Soil Feature", fontsize=12)
    ax3.set_ylabel("Cluster ID", fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, "cluster_heatmap.png")
    fig3.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # Serialize model + scaler
    model_bundle = {"kmeans": kmeans, "scaler": soil_scaler, "pca": pca}
    joblib.dump(model_bundle, KMEANS_MODEL)
    logger.info(f"Model serialized to {KMEANS_MODEL}")

    return {
        "model": kmeans,
        "scaler": soil_scaler,
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
        "df_clustered": df_clustered,
        "plots": [cluster_path, elbow_path, heatmap_path]
    }


def generate_cluster_guidance(profiles: pd.DataFrame) -> dict:
    """
    Generate agronomic guidance for each soil cluster based on nutrient levels.
    """
    guidance = {}
    for cluster_id, row in profiles.iterrows():
        notes = []
        # Nitrogen assessment
        if row["N"] < 30:
            notes.append("Low nitrogen — consider urea or ammonium-based fertilizers")
        elif row["N"] > 80:
            notes.append("High nitrogen — suitable for leafy crops; reduce N-fertilizer")
        else:
            notes.append("Moderate nitrogen — balanced for most crops")

        # Phosphorus assessment
        if row["P"] < 30:
            notes.append("Low phosphorus — apply DAP or superphosphate")
        elif row["P"] > 80:
            notes.append("High phosphorus — ideal for root crops and fruiting")
        else:
            notes.append("Moderate phosphorus — adequate for general agriculture")

        # Potassium assessment
        if row["K"] < 30:
            notes.append("Low potassium — apply MOP (Muriate of Potash)")
        elif row["K"] > 80:
            notes.append("High potassium — good for fruit quality and disease resistance")
        else:
            notes.append("Moderate potassium — suitable for most crops")

        # pH assessment
        if row["ph"] < 5.5:
            notes.append("Acidic soil — lime application recommended")
        elif row["ph"] > 7.5:
            notes.append("Alkaline soil — gypsum or sulfur amendment may help")
        else:
            notes.append("Near-neutral pH — ideal for most crops")

        guidance[int(cluster_id)] = notes

    return guidance


# MODULE 3: Linear Regression (Yield Prediction)
def train_linear_regression(df: pd.DataFrame, test_size: float = 0.2,
                            random_state: int = 42) -> dict:
    """
    Train a Linear Regression model for crop yield prediction.

    Parameters:
        df: Processed DataFrame with features and yield column.
        test_size: Fraction of data for testing.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing the model, metrics, and artifact paths.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Linear Regression — Crop Yield Prediction")
    logger.info("=" * 60)

    X = df[ALL_FEATURES].values
    y = df[TARGET_REGRESSION].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")

    # Train
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    logger.info(f"RMSE:     {rmse:.4f}")
    logger.info(f"MAE:      {mae:.4f}")
    logger.info(f"R² (test):  {r2:.4f}")
    logger.info(f"R² (train): {r2_train:.4f}")

    # Coefficients
    coefficients = dict(zip(ALL_FEATURES, lr_model.coef_))
    logger.info(f"Coefficients: {coefficients}")
    logger.info(f"Intercept: {lr_model.intercept_:.4f}")

    # Residuals
    residuals = y_test - y_pred_test

    # Save metrics
    metrics = {
        "model": "LinearRegression",
        "task": "Crop Yield Prediction",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2_test": round(r2, 4),
        "r2_train": round(r2_train, 4),
        "coefficients": {k: round(v, 6) for k, v in coefficients.items()},
        "intercept": round(lr_model.intercept_, 4),
        "residual_stats": {
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
            "min": round(float(np.min(residuals)), 4),
            "max": round(float(np.max(residuals)), 4)
        }
    }
    save_metrics(metrics, "linear_regression_metrics.json")

    # ── Visualization 1: Actual vs Predicted ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_test, alpha=0.4, s=15, color="steelblue", edgecolors="white", linewidth=0.3)

    # Perfect prediction line
    lims = [min(y_test.min(), y_pred_test.min()) - 1,
            max(y_test.max(), y_pred_test.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Yield (tons/ha)", fontsize=12)
    ax.set_ylabel("Predicted Yield (tons/ha)", fontsize=12)
    ax.set_title(f"Linear Regression — Actual vs Predicted Yield\n"
                 f"R² = {r2:.4f} | RMSE = {rmse:.4f}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    avp_path = os.path.join(RESULTS_DIR, "actual_vs_predicted.png")
    fig.savefig(avp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Actual vs Predicted plot saved to {avp_path}")

    # ── Visualization 2: Residual Plot ──
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    ax2a.scatter(y_pred_test, residuals, alpha=0.4, s=15, color="darkorange",
                 edgecolors="white", linewidth=0.3)
    ax2a.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax2a.set_xlabel("Predicted Yield (tons/ha)", fontsize=12)
    ax2a.set_ylabel("Residual", fontsize=12)
    ax2a.set_title("Residuals vs Predicted Values", fontsize=13, fontweight="bold")

    # Residual distribution
    ax2b.hist(residuals, bins=40, color="teal", edgecolor="white", alpha=0.8, density=True)
    ax2b.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax2b.set_xlabel("Residual", fontsize=12)
    ax2b.set_ylabel("Density", fontsize=12)
    ax2b.set_title(f"Residual Distribution (μ={np.mean(residuals):.2f}, σ={np.std(residuals):.2f})",
                   fontsize=13, fontweight="bold")

    plt.tight_layout()
    res_path = os.path.join(RESULTS_DIR, "residual_analysis.png")
    fig2.savefig(res_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Residual analysis plot saved to {res_path}")

    # ── Visualization 3: Coefficient Bar Chart ──
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    names, vals = zip(*sorted_coefs)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    ax3.barh(range(len(names)), vals, color=colors, edgecolor="white")
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=11)
    ax3.set_xlabel("Coefficient Value", fontsize=12)
    ax3.set_title("Linear Regression — Feature Coefficients",
                  fontsize=14, fontweight="bold")
    ax3.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
    ax3.invert_yaxis()
    plt.tight_layout()
    coef_path = os.path.join(RESULTS_DIR, "regression_coefficients.png")
    fig3.savefig(coef_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # Serialize model
    joblib.dump(lr_model, LINEAR_REG_MODEL)
    logger.info(f"Model serialized to {LINEAR_REG_MODEL}")

    return {
        "model": lr_model,
        "metrics": metrics,
        "plots": [avp_path, res_path, coef_path]
    }


# FULL PIPELINE
def run_model_pipeline():
    """
    Execute the complete model training pipeline:
        1. Load processed data
        2. Train Decision Tree Classifier
        3. Train K-Means Clustering
        4. Train Linear Regression
        5. Print consolidated summary
    """
    ensure_dirs()

    # Load processed data
    logger.info("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATASET)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    # Train all models
    dt_results = train_decision_tree(df)
    km_results = train_kmeans_clustering(df)
    lr_results = train_linear_regression(df)

    # ── Consolidated Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("CONSOLIDATED MODEL PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\n1. Decision Tree Classifier:")
    logger.info(f"   Accuracy:  {dt_results['metrics']['accuracy']:.4f}")
    logger.info(f"   Precision: {dt_results['metrics']['precision_weighted']:.4f}")
    logger.info(f"   Recall:    {dt_results['metrics']['recall_weighted']:.4f}")
    logger.info(f"   F1-Score:  {dt_results['metrics']['f1_weighted']:.4f}")

    logger.info(f"\n2. K-Means Clustering:")
    logger.info(f"   Clusters:        {km_results['metrics']['n_clusters']}")
    logger.info(f"   Silhouette:      {km_results['metrics']['silhouette_score']:.4f}")
    logger.info(f"   Cluster Sizes:   {km_results['metrics']['cluster_sizes']}")

    logger.info(f"\n3. Linear Regression:")
    logger.info(f"   RMSE:  {lr_results['metrics']['rmse']:.4f}")
    logger.info(f"   MAE:   {lr_results['metrics']['mae']:.4f}")
    logger.info(f"   R²:    {lr_results['metrics']['r2_test']:.4f}")

    logger.info("=" * 60)

    return dt_results, km_results, lr_results


# Entry Point
if __name__ == "__main__":
    dt_results, km_results, lr_results = run_model_pipeline()
    print("\n✅ All models trained, evaluated, and serialized successfully.")
    print(f"\nSerialized models:")
    print(f"  • Decision Tree:     {DECISION_TREE_MODEL}")
    print(f"  • K-Means Clustering: {KMEANS_MODEL}")
    print(f"  • Linear Regression:  {LINEAR_REG_MODEL}")
    print(f"\nVisualization plots saved to: {RESULTS_DIR}")