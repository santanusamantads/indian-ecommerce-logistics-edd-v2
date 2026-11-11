# Indian E-Commerce Logistics: EDD & SLA Breach Prediction (Version 2)

A leak-free, end-to-end Machine Learning and Optimization study on **210,850 Indian e-commerce shipment records**. This project implements core regression, regularization, and binary classification algorithms **from scratch in NumPy** alongside Scikit-Learn industry benchmarks.

---

## 📌 Project Overview

Modern Indian e-commerce logistics spans diverse transit corridors—from intra-city hubs to remote territories—across multiple third-party logistics (3PL) partners and seasonal festive surges. This repository solves two core predictive tasks:

1. **Expected Delivery Date (EDD) Regression (`Transit_Time_Hours`):** Continuous prediction of parcel transit duration using pre-dispatch operational and financial attributes.
2. **SLA Breach Classification (`Is_Delayed_SLA`):** Binary detection of high-risk deliveries (`1` = breached SLA, `0` = on-time) under severe class imbalance (`3.49%` positive breach rate).

---

## 📦 Dataset Summary

* **Dataset:** [Indian Multi-Agency E-Commerce Logistics & EDD Dataset (Version 2)](https://www.kaggle.com/datasets/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd?select=india_multi_agency_edd_dataset_v2.csv) (Created by Santanu Samanta)
* **Scale:** `210,850` shipment rows × `21` attributes
* **Logistics Carriers:** Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, Shadowfax
* **Spatial Routing:** Zones A to E across pan-India 6-digit pincodes
* **Transport Modes:** `Surface` and `Air`
* **Festive Surges:** Diwali Peak, Durga Puja / Dussehra, Eid-ul-Fitr, Holi, Independence Day Sale, New Year Carnival, and Normal Period

---

## 🛠️ Data Engineering & Leakage Prevention

To ensure unbiased out-of-sample generalization on the **42,170 held-out test rows** (`80/20` split):
* **Target Leakage Prevention:** Post-delivery variables (`Rating_Stars`, `Actual_Delivery_Days`) and non-predictive identifiers (`Order_ID`, `Origin_Pincode`, `Destination_Pincode`, `Review_Text`) were strictly excluded from the feature matrix.
* **Missing Value Handling:** `Festival_Peak` nulls (`147,569` rows) were imputed as `"Normal_Period"`. Exploratory conditional median imputation on `Rating_Stars` was kept isolated from model training.
* **Encoding & Scaling:** Six categorical features were one-hot encoded (`drop_first=True`), yielding **41 predictive features**. Z-score standardization (\(\mu, \sigma\)) was fitted **strictly on the training set** (`168,680` rows) before transforming test data.

---

## 🧮 Algorithms Implemented from Scratch

* **Batch Gradient Descent with Armijo Backtracking Line Search:** Dynamically contracts step size (\(\beta = 0.5\), \(c = 0.005\)) at every epoch to guarantee monotonic MSE reduction without overshooting.
* **Proximal Gradient Descent Lasso (\(L_1\) Regularization):** Uses Iterative Soft-Thresholding (ISTA) with backtracking line search to induce exact feature sparsity, tuned via an internal validation split (`33,736` rows) across \(\alpha \in [10^{-5}, 10^{1}]\).
* **Batch Logistic Regression:** Optimizes numerically clipped Sigmoid (\(z \in [-500, 500]\)) and Binary Cross-Entropy Log-Loss (\(\hat{p} \in [10^{-15}, 1 - 10^{-15}]\)), paired with a custom confusion matrix and threshold tuning (\(\tau = 0.30\)) for imbalanced classification.

---

## 📊 Experimental Results

### 1. EDD Transit Time Regression (`Transit_Time_Hours`)

| Model Architecture | Test RMSE (Hours) | Test \(R^2\) Score | Notes |
| :--- | :---: | :---: | :--- |
| **Custom Proximal Lasso (Scratch)** | **8.7003** | **0.9280** | \(\alpha = 0.0316\) (selected via validation), prunes **30/41** features to zero |
| **Scikit-Learn Lasso (\(L_1\))** | 8.7012 | 0.9280 | \(\alpha = 0.001\) |
| **Bagging Regressor (Linear Base)** | 8.7013 | 0.9280 | 50 estimators |
| **Ridge Regression (\(L_2\))** | 8.7014 | 0.9280 | \(\alpha = 0.01\) |
| **Custom Batch GD (Scratch)** | **8.7068** | **0.9279** | 1,600 epochs with Armijo backtracking line search |
| **ElasticNet (\(L_1 + L_2\))** | 8.7345 | 0.9274 | \(\alpha = 0.01\), \(l_1\text{\_ratio} = 0.5\) |
| **Random Forest Regressor** | 8.7643 | 0.9269 | 50 trees, `max_depth=12` |

### 2. SLA Breach Classification (`Is_Delayed_SLA` — Minority Class 1)

| Model Architecture | Accuracy | Precision (C1) | Recall (C1) | F1-Score (C1) |
| :--- | :---: | :---: | :---: | :---: |
| **Custom Logistic Regression (Scratch, \(\tau = 0.30\))** | **0.9738** | **0.6514** | 0.5114 | **0.5730** |
| **Random Forest Classifier (`balanced`)** | 0.9403 | 0.3611 | 0.9566 | 0.5243 |
| **Bagging Classifier (Logistic `balanced`)** | 0.9205 | 0.3010 | **0.9904** | 0.4617 |

* **Confusion Matrix (Custom Logistic, \(\tau = 0.30\)):** `TN = 40,322` | `FP = 397` | `FN = 709` | `TP = 742`

---

## 🚀 Repository Structure & Setup

```text
├── india_multi_agency_edd_dataset_v2.csv   # Dataset (210,850 records)
├── notebook.ipynb                          # Complete EDA, from-scratch models & benchmarks
├── report.text                              # LaTeX academic project report
└── README.md                               # Project documentation
```

### Quick Start
1. Clone the repository and install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
2. Place `india_multi_agency_edd_dataset_v2.csv` in the root directory and run all cells in the Jupyter Notebook.

---

## 📜 License & Acknowledgments

* **Dataset:** Public Domain (`CC0`), authored by Santanu Samanta.
* **Academic Context:** Developed as part of the 1st Semester M.Sc. Data Science & Artificial Intelligence coursework at Ramakrishna Mission Vivekananda Educational and Research Institute (RKMVERI).
