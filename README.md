# AI Lab 15
Artificial Intelligence Lab (CSL 411) 

Student Name: Hasan Zahid 

Enrollment #: 01-131232-028 

Teacher: Engr. Saad Mazhar Khan 

Dept of SE, BUIC

---

# Smart Agriculture Decision Support System (DSS)
An integrated precision agriculture pipeline binding data engineering, modular multi-model machine learning architecture, and an interactive Tkinter graphical application.

## 1. System Architecture
The application adheres strictly to the **Separation of Concerns** paradigm, splitting the platform into three independent, robust layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER (src/gui.py)                 │
│  - Interactive Tkinter Dark Theme Dashboard                            │
│  - Dynamic Live Matplotlib Visualization Panels                        │
│  - Sequential Pipeline Inference Driver                                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Loaded serialized artifacts
┌───────────────────────────────────▼────────────────────────────────────┐
│                          MODEL LAYER (src/models.py)                   │
│  - Decision Tree Classifier (96.36% Accuracy Crop Recommendation)     │
│  - K-Means Clustering (Soil Profile Zoning, k=5, Silhouette 0.3571)   │
│  - Linear Regression Model (Crop Yield Prediction with Confidence)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Preprocessed Training Data
┌───────────────────────────────────▼────────────────────────────────────┐
│                    DATA ENGINEERING LAYER (src/preprocessing.py)       │
│  - Kaggle Crop Recommendation Dataset & Metadata loading               │
│  - Missing values (Median) & Outlier capping (IQR bounds)              │
│  - Synthetic Yield Generation via non-linear agronomic modeling        │
│  - StandardScaler & LabelEncoder generation & serialization            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Algorithmic Rationale

### A. Decision Tree Classifier (Crop Recommendation)
- **Task**: Multi-class classification recommending the optimal crop among 22 distinct crop varieties based on soil and climatic features.
- **Rationale**: Highly interpretable, non-parametric, and capable of capturing complex non-linear feature interactions (such as the threshold interactions between rainfall, temperature, and relative humidity). It represents the industry standard for lightweight, low-compute deployment.
- **Implementation**: Max depth bounded to `10` to avoid over-fitting while maintaining robust generalization.

### B. K-Means Clustering (Soil Profile Segmentation)
- **Task**: Unsupervised segmentation of soil chemistry profiles (`N`, `P`, `K`, `pH`) to identify homogeneous agricultural zones.
- **Rationale**: Groups zones with similar nutrient footprints, allowing farm managers to execute localized fertilizer plans. K-Means was selected for its extreme computational efficiency in embedded systems.
- **Implementation**: The optimal number of clusters ($k=5$) was determined using the Elbow Method and Silhouette Analysis. Features were scaled using an independent `StandardScaler` to prevent high-magnitude nutrients (like Nitrogen) from dominating distance metrics.

### C. Linear Regression (Crop Yield Prediction)
- **Task**: Quantitative yield forecasting (metric tons per hectare) for recommended crops.
- **Rationale**: Explains the directional influence of each nutrient and weather metric through direct feature coefficients.
- **Implementation**: Baseline continuous model evaluated with Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² coefficient. Confidence bounds are dynamically formulated using $1.96 \times \sigma_{\text{residuals}}$ from the training set.

---

## 3. Quantitative Performance Summary

### Classification (Decision Tree)
- **Accuracy**: `96.36%`
- **Weighted Precision**: `97.18%`
- **Weighted Recall**: `96.36%`
- **Weighted F1-Score**: `96.39%`

### Clustering (K-Means)
- **Clusters ($k$)**: `5`
- **Silhouette Coefficient**: `0.3571`
- **Soil Zones Profiles**:
  - *Cluster 0*: Low N (25.5), Med P (48.1), Low K (20.5), Neutral pH (7.15)
  - *Cluster 1*: High N (96.5), Med P (41.2), Med K (37.8), Neutral pH (6.5)
  - *Cluster 2*: Med N (47.0), Med P (62.8), Med K (63.1), Neutral pH (7.05)
  - *Cluster 3*: Low N (22.0), High P (126.6), High K (92.5), Slightly Acidic (5.98)
  - *Cluster 4*: Low N (22.7), Med P (41.8), Low K (25.6), Acidic pH (5.76)

### Regression (Linear Regression)
- **R² Score (Test)**: `0.5014`
- **R² Score (Train)**: `0.5003`
- **Root Mean Squared Error (RMSE)**: `5.1336` tons/ha
- **Mean Absolute Error (MAE)**: `3.9068` tons/ha

---

## 4. Installation & Execution

### Prerequisites
Make sure Python 3.8+ and pip are installed.

### Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd "AI Lab 15"
   ```
2. Install all strict dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution Steps
1. **Run the Data Engineering Layer**:
   ```bash
   PYTHONPATH=src python3 src/preprocessing.py
   ```
   This loads, caps, pre-processes, and generates the processed dataset `data/Crop_recommendation_processed.csv` along with scaling encoders.

2. **Run the Model Training Core**:
   ```bash
   PYTHONPATH=src python3 src/models.py
   ```
   Trains all 3 models sequentially, creates joblib serialization files inside `models/`, and generates high-resolution evaluation figures inside `results/`.

3. **Start the Interactive GUI Dashboard**:
   ```bash
   PYTHONPATH=src python3 src/gui.py
   ```
   Opens the dark-mode dashboard allowing interactive param submissions and dynamic visualization tab switches.

---

## 5. Future Work
1. **Non-Linear Yield Regression with Ensemble Methods**: Replace the baseline Linear Regression module with a Random Forest Regressor or Gradient Boosted Trees (XGBoost) to better capture complex non-linear climatic interactions, boosting the R² accuracy of the yield prediction component.
2. **Deep Representation Learning for Clustering**: Integrate a Deep Autoencoder pipeline to perform non-linear feature dimension reduction on heterogeneous soil parameters prior to clustering, resolving multi-collinearity issues and establishing more compact, robust zoning boundaries.
