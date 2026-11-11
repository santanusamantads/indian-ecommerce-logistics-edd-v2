## 📦 Indian E-Commerce Logistics: EDD & SLA Breach Prediction (Version 2)

**M.Sc. Data Science & Artificial Intelligence Project**  
*Ramakrishna Mission Vivekananda Educational and Research Institute (RKMVERI)*  
*Author:* Santanu Samanta  

---

## 📌 Project Overview

Modern Indian e-commerce logistics spans diverse transit corridors—from intra-city hubs to remote territories—across multiple third-party logistics (3PL) partners and seasonal festive surges. This repository solves two core predictive tasks:

1. **Expected Delivery Date (EDD) Regression (`Transit_Time_Hours`):** Continuous prediction of parcel transit duration using pre-dispatch operational and financial attributes.
2. **SLA Breach Classification (`Is_Delayed_SLA`):** Binary detection of high-risk deliveries (`1` = breached SLA, `0` = on-time) under severe class imbalance (`3.49%` positive breach rate).

---


## 📊 Dataset Summary

* **Dataset:** [Indian Multi-Agency E-Commerce Logistics & EDD Dataset (Version 2)](https://www.kaggle.com/datasets/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd?select=india_multi_agency_edd_dataset_v2.csv) (Created by Santanu Samanta)
* **Scale:** `210,850` shipment rows × `21` attributes
* **Logistics Carriers:** Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, Shadowfax
* **Spatial Routing:** Zones A to E across pan-India 6-digit pincodes
* **Transport Modes:** `Surface` and `Air`
* **Festive Surges:** Diwali Peak, Durga Puja / Dussehra, Eid-ul-Fitr, Holi, Independence Day Sale, New Year Carnival, and Normal Period

---

### 🛡️ Preprocessing Pipeline & Data Leakage Prevention

To ensure unbiased out-of-sample generalization on the holdout test partition ($m_{\text{test}} = 42{,}170$ rows, $80/20$ sequential split):

* **Target Leakage & Identifier Exclusion**  
  Post-delivery realization variables ($\text{Rating\_Stars}$, $\text{Actual\_Delivery\_Days}$) and high-cardinality/non-predictive identifiers ($\text{Order\_ID}$, $\text{Origin\_Pincode}$, $\text{Destination\_Pincode}$, $\text{Review\_Text}$) were strictly purged prior to constructing the design matrix $\mathbf{X}$.

---

* **Missing Value Imputation Protocol**  
  Structural missingness in seasonal attributes ($\text{Festival\_Peak}$, $147{,}569$ null records) was categorically encoded as `"Normal_Period"`. Exploratory conditional median imputation conducted on user feedback metrics was strictly isolated from downstream training manifolds to prevent distribution bleed.

---

* **Categorical Expansion & Strict In-Sample Feature Scaling**  
  Categorical streams were one-hot encoded with reference category elimination (`drop_first=True`) to avoid the dummy variable trap and matrix singularity, expanding the feature space to $d = 41$ predictive dimensions:
  $$\mathbf{X} \in \mathbb{R}^{m \times 41}$$
  Continuous feature vectors were standardized via explicit Z-score transformations parameterized **strictly on the training partition** ($m_{\text{train}} = 168{,}680$ instances) to prevent data snooping:
  $$\mu_j = \frac{1}{m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} x_{i,j}, \qquad \sigma_j = \sqrt{\frac{1}{m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} (x_{i,j} - \mu_j)^2}$$
  $$x_{i,j}^{(\text{scaled})} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$
  The precomputed training statistics $(\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}})$ were then deterministically mapped across the test slice $\mathbf{X}_{\text{test}}$.

---

## 🧮 Algorithms Implemented from Scratch

* **Batch Gradient Descent with Armijo Backtracking Line Search**  
  Dynamically contracts the step size $\eta$ using parameters $\beta = 0.5$ and $c = 0.005$ to guarantee monotonic Mean Squared Error reduction without overshooting:
  $$f(\mathbf{w} - \eta \nabla f(\mathbf{w})) \le f(\mathbf{w}) - c \, \eta \, \|\nabla f(\mathbf{w})\|_2^2$$
  where the learning rate updates iteratively via $\eta \leftarrow \beta \eta$ until the Armijo condition is satisfied.

---

* **Proximal Gradient Descent Lasso ($L_1$ Regularization)**  
  Applies Iterative Soft-Thresholding Algorithm (ISTA) combined with line search to induce exact parameter sparsity over the objective:
  $$\min_{\mathbf{w}} \frac{1}{2m} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_1$$
  Parameter coordinates are updated through the soft-thresholding operator:
  $$\mathcal{S}_{\eta \alpha}(v_j) = \operatorname{sign}(v_j) \cdot \max\left(0,\, |v_j| - \eta \alpha\right)$$
  Tuned across an internal validation holdout (33,736 instances) spanning $\alpha \in [10^{-5}, 10^{1}]$.

---

* **Batch Logistic Regression (SLA Delay Classification)**  
  Trained on imbalanced outcomes using numerically stable, boundary-clipped Sigmoid and Binary Cross-Entropy Log-Loss:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \operatorname{clip}(\mathbf{w}^T \mathbf{x} + b, -500, 500)$$
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \Big]$$
  Probabilities are clipped to $\hat{p} \in [10^{-15}, 1 - 10^{-15}]$ to prevent numerical divergence, evaluated with an operational decision boundary threshold of $\tau = 0.30$.

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
