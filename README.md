# 🚚 Indian Multi-Agency E-Commerce Logistics: EDD & SLA Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview & Two Core Motives

This repository implements an end-to-end Machine Learning pipeline from scratch on **210,850** pan-India multi-carrier shipments to solve two core operational logistics objectives:

1. **EDD (Estimated Delivery Date) Prediction — Continuous Regression:** Predict actual shipment transit duration (`Transit_Time_Hours`) using Batch Gradient Descent with Armijo Backtracking Line Search and Proximal Gradient Descent for Lasso ($L_1$).
2. **SLA (Service Level Agreement) Violation Prediction — Binary Classification:** Predict operational SLA delay risk (`Is_Delayed_SLA`) using Standard and Class-Weighted Binary Logistic Regression from scratch across fixed ($\tau = 0.30$, $\tau = 0.50$) and validation-tuned F1-optimal ($\tau^* = 0.1900$) probability thresholds under severe class imbalance (**3.49%** overall delay rate / **3.44%** in the holdout test split).

---

## 📊 Dataset & Leakage-Free Preprocessing

- **Dataset:** Indian Multi-Agency E-Commerce Logistics & EDD Dataset (v2) (**210,850 rows × 21 attributes**)
- **Logistics Providers:** Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, and Shadowfax
- **Routing & Modes:** Logistics Zones A–E across pan-India cities (`Surface` and `Air` shipment modes)
- **Data Partitioning:** Sequential 80/20 holdout split (**168,680** training rows / **42,170** holdout test rows)
- **Preprocessing & Multicollinearity Controls:**
  - **Post-Delivery & ID Purging:** Dropped non-causal outcome columns (`Actual_Delivery_Days`, `Rating_Stars`) and non-predictive identifiers (`Order_ID`, `Origin_Pincode`, `Destination_Pincode`, `Review_Text`).
  - **Exact Multicollinearity Removal:** Dropped `Total_Shipping_Cost` because `Total_Shipping_Cost = Freight_Cost + Insurance_Fee` exactly (`2069.28 + 103.46 = 2172.75`), ensuring a full-rank feature matrix.
  - **Missing Value Imputation:** Encoded **147,569** null records in `Festival_Peak` as `"Normal_Period"`. Exploratory conditional median imputation on `Rating_Stars` (**6,345** nulls, grouped by `Delivery_Partner` and `Is_Delayed_SLA`) was isolated from downstream feature matrices.
  - **One-Hot Encoding & In-Sample Scaling:** Applied dummy encoding (`drop_first=True`) to expand categorical variables into $X \in \mathbb{R}^{m \times 40}$ (encoded dataframe shape: `210,850 × 49` prior to target/ID dropping). Continuous features (`Distance_km`, `Weight_kg`, `Warehouse_Handling_Fee`, `Fuel_Surcharge`, `Freight_Cost`, `Insurance_Fee`) were standardized using mean and standard deviation computed strictly on the training set before transforming $X_{\text{test}}$.

---

## 🎯 Motive 1: EDD Prediction (Regression)

### Optimization Formulation

- **Armijo Backtracking Line Search:** Dynamically contracts step size $\eta$ ($\beta = 0.5$, $c = 0.005$) over `3,500` epochs (`learning_rt = 0.1`) to guarantee monotonic Mean Squared Error reduction:
  $$f(w - \eta \nabla f(w)) \le f(w) - c \, \eta \, \|\nabla f(w)\|_2^2$$
- **Proximal Gradient Descent (Lasso $L_1$):** Applies element-wise soft-thresholding over `1,500` epochs (`learning_rt = 0.1`):
  $$S_{\eta\alpha}(v_j) = \text{sign}(v_j)\max(0, |v_j| - \eta\alpha)$$
  An 80/20 inner validation sweep across $\alpha \in [10^{-5}, 10^{1}]$ selected **$\alpha = 0.031623$** (Inner Validation RMSE = **8.7486 hours**).

### EDD Benchmark Results (`Transit_Time_Hours`)

| Model Architecture | Test RMSE (Hours) | Test $R^2$ Score | Notes |
| :--- | :---: | :---: | :--- |
| **Custom Proximal Lasso (Scratch)** | **8.7004** | **0.9280** | $\alpha = 0.031623$, prunes **30/40** redundant features to zero |
| **Scikit-Learn Lasso ($L_1$)** | 8.7012 | 0.9280 | $\alpha = 0.001$ |
| **Custom GD (Scratch)** | 8.7013 | 0.9280 | `epochs=3500`, Armijo backtracking line search |
| **Bagging Regressor (Linear)** | 8.7013 | 0.9280 | `n_estimators=50` |
| **Ridge Regression ($L_2$)** | 8.7014 | 0.9280 | $\alpha = 0.01$ |
| **ElasticNet ($L_1 + L_2$)** | 8.7378 | 0.9274 | $\alpha = 0.01$, `l1_ratio=0.5` |
| **Random Forest Regressor** | 8.7625 | 0.9270 | `n_estimators=50`, `max_depth=12` |

- **Selected EDD Model:** **Custom Proximal Lasso ($L_1$)** achieves the lowest holdout error (**8.7004 hours RMSE**) while eliminating **75%** (`30/40`) of sparse dummy features for fast, lightweight deployment.

---

## 🚨 Motive 2: SLA Violation Prediction (Classification)

### Optimization Formulation

- **Standard Binary Logistic Regression (Scratch):** Models SLA delay probability $\hat{p} = \sigma(w^T x + b)$ via Batch Gradient Descent (`learning_rate = 0.1`, `epochs = 1500`) minimizing Binary Cross-Entropy loss:
  $$L_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$
- **Class-Weighted Binary Logistic Regression (Scratch):** Addresses severe class imbalance (`n_neg = 162,774`, `n_pos = 5,906` in training) by incorporating an inverse-frequency positive-class weight $w_{\text{pos}} = \frac{n_{\text{neg}}}{n_{\text{pos}}} = \mathbf{27.56}$ directly into the BCE objective (`learning_rate = 0.1`, `epochs = 1500`):
  $$L_{\text{WBCE}} = -\frac{1}{m} \sum_{i=1}^{m} \left[ w_{\text{pos}} \, y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

### Threshold Tuning & Ranking-Quality Diagnostics

- **Validation Precision-Recall Threshold Sweep:** Sweeping $\tau \in [0.05, 0.60]$ across a 10% internal validation slice carved strictly from `X_train` identified an F1-optimal operating threshold of **$\tau^* = 0.1900$** (Validation F1 = **0.5885**, outperforming the default $\tau = 0.3000$ operating point).
- **Threshold-Independent Ranking Diagnostics (Holdout Test Set):**
  - **ROC-AUC Score:** **0.9736**
  - **Average Precision (PR-AUC):** **0.5847** (vs. `0.0344` no-skill prevalence baseline)

### SLA Benchmark Results (`Is_Delayed_SLA`)

Because only **3.44%** (`1,451 / 42,170`) of holdout shipments violate SLA, a majority-class dummy classifier achieves **96.56%** accuracy with `0.0000` recall. Models are therefore evaluated primarily on **Precision**, **Recall**, and **F1-Score** for Class 1 (`Is_Delayed_SLA = 1`):

| Model Architecture | Threshold ($\tau$) | Precision (C1) | Recall (C1) | F1-Score (C1) | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bagging Classifier (Logistic)** | 0.30 | 0.5561 | **0.6871** | **0.6147** | 97.04% |
| **Random Forest Classifier** | 0.30 | 0.5527 | **0.6871** | 0.6126 | 97.01% |
| **Custom Logistic Regression (Scratch — F1-Tuned)** | **0.19** | 0.5692 | 0.6547 | **0.6090** | 97.11% |
| **Custom Logistic Regression (Scratch)** | 0.30 | **0.6332** | 0.5341 | 0.5794 | **97.33%** |
| **Bagging Classifier (Logistic)** | 0.50 | 0.6632 | 0.4900 | 0.5636 | 97.39% |
| **Random Forest Classifier** | 0.50 | 0.6667 | 0.4645 | 0.5475 | 97.36% |
| **Custom Logistic Regression (Scratch)** | 0.50 | 0.7426 | 0.2764 | 0.4028 | 97.18% |
| **Custom Weighted Logistic (`pos_weight=27.56`)** | **0.19** | 0.1527 | **0.9986** | 0.2649 | 80.93% |

- **Operating Point Trade-Offs:**
  - **Custom Logistic Regression ($\tau^* = 0.1900$):** Maximizes F1-score (**0.6090**) among scratch models while maintaining **97.11%** overall accuracy and catching **65.47%** of SLA breaches.
  - **Custom Logistic Regression ($\tau = 0.3000$):** Delivers higher alert precision (**0.6332**) and **97.33%** accuracy when minimizing false delay alerts is prioritized.
  - **Custom Weighted Logistic Regression (`pos_weight = 27.56`, $\tau^* = 0.1900$):** Captures **99.86%** (`1,449 / 1,451`) of all SLA breaches (only **2** missed delays) for high-sensitivity early-warning triage.

### Holdout Confusion Matrices ($m_{\text{test}} = 42,170$)

- **Standard Custom Logistic Regression ($\tau = 0.3000$):**
  - **True Positives (TP):** `775` | **False Positives (FP):** `449`
  - **True Negatives (TN):** `40,270` | **False Negatives (FN):** `676`
- **Standard Custom Logistic Regression ($\tau^* = 0.1900$, F1-Optimal):**
  - **True Positives (TP):** `950` | **False Positives (FP):** `719`
  - **True Negatives (TN):** `40,000` | **False Negatives (FN):** `501`
- **Class-Weighted Custom Logistic Regression (`pos_weight = 27.56`, $\tau^* = 0.1900$):**
  - **True Positives (TP):** `1,449` | **False Positives (FP):** `8,041`
  - **True Negatives (TN):** `32,678` | **False Negatives (FN):** `2`

---

## ⚙️ Learned Parameters of Chosen Models

- **EDD Custom Proximal Lasso Intercept ($b_{\text{EDD}}$):** `85.136029` (10 non-zero weights out of 40 features)
  - `Distance_km`: `+39.003655` | `Festival_Peak_Normal_Period`: `-39.022109`
  - `Freight_Cost`: `-10.599448` | `Insurance_Fee`: `-10.599162`
  - `Fuel_Surcharge`: `+1.253778` | `Shipment_Mode_Surface`: `+0.283790`
  - `Warehouse_Handling_Fee`: `+0.144645` | `Logistics_Zone_Zone B`: `+0.037001`
  - `Weight_kg`: `+0.026705` | `Logistics_Zone_Zone C`: `-0.008649`
- **SLA Custom Logistic Regression Top Drivers:** `Festival_Peak_Normal_Period`, `Shipment_Mode_Surface`, `Freight_Cost`, `Insurance_Fee`, and `Distance_km`

---

## 🔍 Unseen Holdout Data Audit (50-Shipment Verification)

Both chosen models were audited on **50 unseen holdout shipments** (`np.random.seed(42)` from the test split) against actual delivery logs:

- **Sample Mean Absolute Error (EDD):** **6.29 hours** (~0.26 days)
- **Sample SLA Classification Match ($\tau = 0.30$):** **98.0%** (`49/50` exact matches, catching `5/6` actual SLA violations with `0` false positives)

| Order ID | Partner | Mode | Actual Hours | Pred Hours | Error (Hrs) | Actual SLA | SLA Prob | Pred SLA |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `IND202600171970` | Ekart | Surface | 56.96 | 59.17 | +2.21 | 0 | 0.0075 | 0 |
| `IND202600168687` | Blue Dart | Air | 106.13 | 85.12 | -21.01 | 1 | 0.6640 | 1 |
| `IND202600184075` | Delhivery | Surface | 136.66 | 132.62 | -4.04 | 0 | 0.0815 | 0 |
| `IND202600196807` | Ekart | Air | 20.29 | 19.76 | -0.53 | 0 | 0.0086 | 0 |
| `IND202600198447` | Ekart | Air | 92.85 | 82.94 | -9.91 | 1 | 0.5715 | 1 |
| `IND202600194376` | Delhivery | Surface | 94.06 | 92.99 | -1.07 | 0 | 0.0219 | 0 |
| `IND202600183947` | Blue Dart | Air | 31.06 | 32.04 | +0.98 | 0 | 0.0416 | 0 |
| `IND202600172963` | XpressBees | Air | 95.79 | 83.99 | -11.80 | 1 | 0.6220 | 1 |
| `IND202600177632` | Blue Dart | Air | 72.74 | 69.60 | -3.14 | 1 | 0.3249 | 1 |
| `IND202600172308` | Ecom Express | Air | 88.74 | 76.45 | -12.29 | 1 | 0.3947 | 1 |

---

## 🚀 Repository Structure

```text
├── app.py                                    # Interactive Streamlit prediction & analytics dashboard
├── india_multi_agency_edd_dataset_v2.csv     # Source logistics dataset (210,850 rows × 21 columns)
├── indian-ecommerce-logistics-edd-v2.ipynb   # Complete Jupyter Notebook (EDA, Scratch Models, Tuning & Audit)
├── trained_cache_v2.joblib                   # Serialized model weights & preprocessing artifacts
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation & benchmark report
```
