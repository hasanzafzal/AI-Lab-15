"""
preprocessing.py — Data Engineering Layer for the Smart Agriculture DSS.

Responsibilities:
    1. Load raw dataset from the Kaggle Crop Recommendation CSV.
    2. Perform exploratory data analysis (EDA) and document findings.
    3. Handle missing values via median imputation.
    4. Treat outliers using IQR-based capping.
    5. Generate realistic synthetic crop yield column for regression tasks.
    6. Apply feature scaling (StandardScaler) and categorical encoding (LabelEncoder).
    7. Save the processed dataset, scaler, and encoder artifacts for reuse.
"""

import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from utils import (
    RAW_DATASET, PROCESSED_DATASET, SCALER_ARTIFACT, LABEL_ENCODER_ARTIFACT,
    DATA_DIR, RESULTS_DIR, ALL_FEATURES, SOIL_FEATURES, CLIMATE_FEATURES,
    TARGET_CLASSIFICATION, TARGET_REGRESSION, CROP_YIELD_PARAMS,
    ensure_dirs, save_metrics, setup_logger
)

logger = setup_logger("preprocessing")


# 1. Data Loading
def load_raw_data(filepath: str = RAW_DATASET) -> pd.DataFrame:
    """Load the raw Crop Recommendation CSV dataset."""
    logger.info(f"Loading raw dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# 2. Exploratory Data Analysis
def generate_data_dictionary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive data dictionary documenting each column's
    type, range, distribution statistics, and description.
    """
    descriptions = {
        "N":           "Ratio of Nitrogen content in soil (kg/ha)",
        "P":           "Ratio of Phosphorus content in soil (kg/ha)",
        "K":           "Ratio of Potassium content in soil (kg/ha)",
        "temperature": "Average temperature during the growing season (°C)",
        "humidity":    "Relative humidity (%)",
        "ph":          "Soil pH value (acidity/alkalinity measure)",
        "rainfall":    "Annual rainfall (mm)",
        "label":       "Target crop type for recommendation (22 classes)",
    }

    data_dict = {}
    for col in df.columns:
        info = {
            "description": descriptions.get(col, "N/A"),
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }
        if df[col].dtype in ["int64", "float64"]:
            info.update({
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "median": round(float(df[col].median()), 4),
                "q25": round(float(df[col].quantile(0.25)), 4),
                "q75": round(float(df[col].quantile(0.75)), 4),
            })
        else:
            info["sample_values"] = df[col].value_counts().head(5).to_dict()

        data_dict[col] = info

    return data_dict


# 3. Missing Value Imputation
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median for numeric columns
    and mode for categorical columns.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    missing_before = df.isnull().sum().sum()

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Imputed {col} with median = {median_val:.4f}")

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"Imputed {col} with mode = {mode_val}")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} → {missing_after}")
    return df


# 4. Outlier Treatment (IQR Capping)
def treat_outliers_iqr(df: pd.DataFrame, columns: list = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Cap outliers using the Interquartile Range (IQR) method.
    Values beyond Q1 − factor*IQR and Q3 + factor*IQR are clipped.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_report = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"Capped {n_outliers} outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")

        outlier_report[col] = {
            "q1": round(q1, 4), "q3": round(q3, 4), "iqr": round(iqr, 4),
            "lower_bound": round(lower, 4), "upper_bound": round(upper, 4),
            "outliers_capped": int(n_outliers)
        }

    return df, outlier_report


# 5. Synthetic Yield Generation
def generate_yield_column(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic yield column (tons/hectare) based on
    crop-specific agronomic parameters and soil/climate feature interactions.

    The yield model considers:
        - Crop-specific base and maximum yield potentials
        - Temperature optimality (Gaussian decay from optimal temp)
        - Rainfall adequacy (Gaussian decay from optimal rainfall)
        - NPK nutrient contribution (normalized additive factor)
        - pH suitability (penalty for extreme values)
        - Random noise for realistic variation
    """
    df = df.copy()
    rng = np.random.RandomState(seed)

    yields = np.zeros(len(df))

    for idx, row in df.iterrows():
        crop = row[TARGET_CLASSIFICATION]
        params = CROP_YIELD_PARAMS.get(crop, {"base_yield": 2.0, "max_yield": 5.0,
                                                "optimal_temp": 25, "optimal_rain": 100})

        base = params["base_yield"]
        max_y = params["max_yield"]
        opt_temp = params["optimal_temp"]
        opt_rain = params["optimal_rain"]

        # Temperature suitability factor (Gaussian, σ = 5°C)
        temp_factor = np.exp(-0.5 * ((row["temperature"] - opt_temp) / 5.0) ** 2)

        # Rainfall suitability factor (Gaussian, σ = 50mm)
        rain_factor = np.exp(-0.5 * ((row["rainfall"] - opt_rain) / 50.0) ** 2)

        # NPK contribution (normalized to [0, 1] range)
        npk_factor = (
            min(row["N"] / 140.0, 1.0) * 0.4 +
            min(row["P"] / 145.0, 1.0) * 0.3 +
            min(row["K"] / 205.0, 1.0) * 0.3
        )

        # pH suitability (optimal range 6.0–7.0, penalty outside)
        ph = row["ph"]
        if 6.0 <= ph <= 7.0:
            ph_factor = 1.0
        else:
            ph_factor = max(0.5, 1.0 - 0.1 * abs(ph - 6.5))

        # Composite yield
        composite = temp_factor * rain_factor * npk_factor * ph_factor
        yield_val = base + (max_y - base) * composite

        # Add noise (±10%)
        noise = rng.normal(0, 0.10 * yield_val)
        yield_val = max(0.1, yield_val + noise)

        yields[idx] = round(yield_val, 2)

    df[TARGET_REGRESSION] = yields
    logger.info(f"Generated yield column: mean={np.mean(yields):.2f}, "
                f"std={np.std(yields):.2f}, range=[{np.min(yields):.2f}, {np.max(yields):.2f}]")
    return df


# 6. Feature Scaling & Encoding
def scale_features(df: pd.DataFrame, features: list = ALL_FEATURES) -> tuple:
    """
    Apply StandardScaler to numeric features.
    Returns the scaled DataFrame and the fitted scaler.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    logger.info(f"Scaled {len(features)} features with StandardScaler")
    return df_scaled, scaler


def encode_labels(df: pd.DataFrame, column: str = TARGET_CLASSIFICATION) -> tuple:
    """
    Encode categorical crop labels to numeric values using LabelEncoder.
    Returns the DataFrame with encoded column and the fitted encoder.
    """
    encoder = LabelEncoder()
    df_encoded = df.copy()
    df_encoded[f"{column}_encoded"] = encoder.fit_transform(df[column])
    logger.info(f"Encoded {column}: {len(encoder.classes_)} classes → "
                f"{list(encoder.classes_[:5])}...")
    return df_encoded, encoder


# 7. Full Pipeline
def run_preprocessing_pipeline():
    """
    Execute the complete preprocessing pipeline:
        1. Load raw data
        2. Generate data dictionary
        3. Impute missing values
        4. Treat outliers
        5. Generate synthetic yield
        6. Scale features
        7. Encode labels
        8. Save all artifacts
    """
    ensure_dirs()

    # Step 1: Load
    df = load_raw_data()

    # Step 2: Data dictionary (on raw data)
    data_dict = generate_data_dictionary(df)
    dict_path = save_metrics(data_dict, "data_dictionary.json")
    logger.info(f"Data dictionary saved to {dict_path}")

    # Step 3: Impute missing values
    df = impute_missing_values(df)

    # Step 4: Outlier treatment
    df, outlier_report = treat_outliers_iqr(df, columns=ALL_FEATURES)
    save_metrics(outlier_report, "outlier_report.json")

    # Step 5: Generate yield column
    df = generate_yield_column(df)

    # Step 6: Scale features
    df_scaled, scaler = scale_features(df, features=ALL_FEATURES)

    # Step 7: Encode labels
    df_final, encoder = encode_labels(df_scaled)

    # Step 8: Save artifacts
    # Save processed dataset (with BOTH original-scale and scaled features)
    # We save the unscaled version with yield for human readability
    df.to_csv(PROCESSED_DATASET, index=False)
    logger.info(f"Processed dataset saved to {PROCESSED_DATASET}")

    # Save scaler and encoder
    joblib.dump(scaler, SCALER_ARTIFACT)
    logger.info(f"Scaler saved to {SCALER_ARTIFACT}")

    joblib.dump(encoder, LABEL_ENCODER_ARTIFACT)
    logger.info(f"Label encoder saved to {LABEL_ENCODER_ARTIFACT}")

    # Summary statistics on processed data
    proc_summary = {
        "total_samples": len(df),
        "features": ALL_FEATURES,
        "num_crops": int(df[TARGET_CLASSIFICATION].nunique()),
        "crop_list": sorted(df[TARGET_CLASSIFICATION].unique().tolist()),
        "yield_stats": {
            "mean": round(float(df[TARGET_REGRESSION].mean()), 4),
            "std": round(float(df[TARGET_REGRESSION].std()), 4),
            "min": round(float(df[TARGET_REGRESSION].min()), 4),
            "max": round(float(df[TARGET_REGRESSION].max()), 4),
        },
        "preprocessing_steps": [
            "Missing value imputation (median/mode)",
            "Outlier treatment (IQR capping, factor=1.5)",
            "Synthetic yield generation (agronomic model)",
            "Feature scaling (StandardScaler)",
            "Label encoding (LabelEncoder)"
        ]
    }
    save_metrics(proc_summary, "preprocessing_summary.json")

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info(f"  Samples:  {proc_summary['total_samples']}")
    logger.info(f"  Features: {proc_summary['features']}")
    logger.info(f"  Crops:    {proc_summary['num_crops']}")
    logger.info(f"  Yield:    μ={proc_summary['yield_stats']['mean']:.2f}, "
                f"σ={proc_summary['yield_stats']['std']:.2f}")
    logger.info("=" * 60)

    return df, scaler, encoder


# Entry Point
if __name__ == "__main__":
    df, scaler, encoder = run_preprocessing_pipeline()
    print("\n✅ Preprocessing complete. Processed dataset and artifacts saved.")
    print(f"\nProcessed data shape: {df.shape}")
    print(f"\nSample output:\n{df.head()}")