# 🚚 Indian Multi-Agency E-Commerce Logistics: EDD & SLA Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview & Two Core Motives

This repository implements an end-to-end Machine Learning pipeline from scratch on **210,850** pan-India multi-carrier shipments to solve two core operational logistics objectives:

1. **EDD (Estimated Delivery Date) Prediction - Continuous Regression:** Predict actual shipment transit duration (`Transit_Time_Hours`) using Batch Gradient Descent with Armijo Backtracking Line Search and Proximal Gradient Descent for Lasso ($L_1$).
2. **SLA (Service Level Agreement) Violation Prediction - Binary Classification:** Predict operational SLA delay risk (`Is_Delayed_SLA`) using Standard and Class-Weighted Binary Logistic Regression from scratch across fixed ($\tau = 0.30$, $\tau = 0.50$) and validation-tuned F1-optimal ($\tau^* = 0.1900$) probability thresholds under severe class imbalance (**3.49%** overall delay rate / **3.44%** in the holdout test split).

---

## 📊 Dataset & Leakage-Free Preprocessing

- **Dataset:** [Indian Multi-Agency E-Commerce Logistics & EDD Dataset (v2)](https://www.kaggle.com/datasets/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd?select=india_multi_agency_edd_dataset_v2.csv) (**210,850 rows × 21 attributes**)
- **Kaggle Dataset Link:** [https://www.kaggle.com/datasets/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd?select=india_multi_agency_edd_dataset_v2.csv](https://www.kaggle.com/datasets/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd?select=india_multi_agency_edd_dataset_v2.csv)
- **Logistics Providers:** Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, and Shadowfax
- **Routing & Modes:** Logistics Zones A-E across pan-India cities (`Surface` and `Air` shipment modes)
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
  $$f(w - \eta \nabla f(w)) \le f(w) - c \ \eta \ \|\nabla f(w)\|_2^2$$
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
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$
- **Class-Weighted Binary Logistic Regression (Scratch):** Addresses severe class imbalance (`n_neg = 162,774`, `n_pos = 5,906` in training, `pos_weight = 27.56`) by incorporating an inverse-frequency positive-class weight $w_{\text{pos}} = \frac{n_{\text{neg}}}{n_{\text{pos}}} = \mathbf{27.56}$ directly into the BCE objective (`learning_rate = 0.1`, `epochs = 1500`):
  $$\mathcal{L}_{\text{WBCE}} = -\frac{1}{m} \sum_{i=1}^{m} \left[ w_{\text{pos}} \ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

### Threshold Tuning & Ranking-Quality Diagnostics

- **Validation Precision-Recall Threshold Sweep:** Sweeping $\tau \in [0.05, 0.60]$ with 56 candidates across a 10% internal validation slice carved strictly from `X_train` identified an F1-optimal operating threshold of **$\tau^* = 0.1900$** (Validation F1 = **0.5885**, outperforming the default $\tau = 0.3000$ operating point).
- **Threshold-Independent Ranking Diagnostics (Holdout Test Set):**
  - **ROC-AUC Score:** **0.9736**
  - **Average Precision (PR-AUC):** **0.5847** (vs. `0.0344` no-skill prevalence baseline)

### SLA Benchmark Results (`Is_Delayed_SLA`)

Because only **3.44%** (`1,451 / 42,170`) of holdout shipments violate SLA, a majority-class dummy classifier achieves **96.56%** accuracy with `0.0000` recall. Models are therefore evaluated primarily on **Precision**, **Recall**, and **F1-Score** for Class 1 (`Is_Delayed_SLA = 1`):

| Classification Architecture | Operating Threshold ($\tau$) | Class 1 Precision | Class 1 Recall | Class 1 F1-Score | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom Weighted Logistic (`pos_weight=27.56`)** | **0.19** | **0.1527** | **0.9986** | **0.2649** | **80.93%** |
| **Custom Standard Logistic (Scratch)** | 0.19 | 0.6384 | 0.5479 | 0.5897 | 97.43% |
| **Custom Standard Logistic (Scratch)** | 0.19 | 0.6384 | 0.5479 | 0.5897 | 97.43% |
| **Random Forest Classifier** | 0.19 | 0.6410 | 0.5362 | 0.5839 | 97.44% |
| **Custom Standard Logistic (Scratch)** | 0.30 | 0.6974 | 0.4480 | 0.5455 | 97.47% |
| **Custom Standard Logistic (Scratch)** | 0.50 | 0.7719 | 0.3343 | 0.4663 | 97.40% |

- **Operational SLA Triage Strategy:**
  - **Custom Weighted Logistic Regression (`pos_weight = 27.56`, $\tau^* = 0.1900$):** Captures **99.86%** (`1,449 / 1,451`) of all SLA breaches (only **2** missed delays) for high-sensitivity early-warning triage, yielding a weighted logistic regression report with **80.93%** accuracy, **0.1527** precision, **0.9986** recall, and **0.2649** F1-score.
  - **Custom Standard Logistic ($\tau^* = 0.1900$):** Balances detection accuracy (**0.5897** F1-score) for automated notification workflows.

---

## 📈 Confusion Matrices & Error Diagnostics

```
[Holdout Test Set Evaluation: 42,170 Shipments]

1. Custom Weighted Logistic Regression (pos_weight = 27.56, tau = 0.1900):
   TN: 32,674  |  FP: 8,045
   FN:      2  |  TP: 1,449
   Recall: 99.86% | Precision: 15.27% | Accuracy: 80.93% | F1: 0.2649

2. Custom Standard Logistic Regression (tau = 0.1900):
   TN: 40,269  |  FP:   450
   FN:    656  |  TP:   795
   Recall: 54.79% | Precision: 63.84% | Accuracy: 97.43% | F1: 0.5897
```

---

## 🔬 Learned Parameters & Sensitivity Insights

- **Key Delay Escalators ($w_j > 0$):** `Distance_km` ($+3.42$), `Weather_Condition_Monsoon` ($+1.85$), `Carrier_Shadowfax` ($+1.12$), `Logistics_Zone_E` ($+0.98$).
- **Key Delay Mitigators ($w_j < 0$):** `Shipment_Mode_Air` ($-2.14$), `Carrier_Blue_Dart` ($-1.45$).

---

## 🛡️ Unseen Holdout Audit

- **Audit Strategy:** Evaluated on an independent, non-overlapping sequential 20% holdout split (**42,170** unseen shipments).
- **Leakage Zero-Tolerance:** Continuous feature standard scaling parameters ($\mu, \sigma$) and categorical dummy structures were generated exclusively from `X_train`.

---

## 📁 Repository Structure

```
.
├── README.md                                  # Comprehensive technical report and benchmarks
├── indian-ecommerce-logistics-edd-v2.ipynb    # End-to-end Python notebook (Scratch ML & Benchmarks)
└── requirements.txt                           # Core dependencies (numpy, pandas, scikit-learn, etc.)
```

---

## 📜 License & Citation

Distributed under the MIT License. See `LICENSE` for details.
