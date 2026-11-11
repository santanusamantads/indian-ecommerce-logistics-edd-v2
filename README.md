# 🚚 Indian E-Commerce Logistics & Estimated Delivery Date (EDD) Prediction (v2)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Overview

This repository contains an end-to-end Machine Learning pipeline designed to predict Estimated Delivery Dates (EDD) and analyze shipment delay risks across pan-India e-commerce logistics networks. Built upon real-world multi-carrier logistics telemetry (`210,850` shipments across `6` major logistics providers), the project models complex supply chain dynamics, regional routing bottlenecks, and seasonal festive demand surges.

### 🎯 Key Objectives & Core Features
* **Robust Regression:** Custom implementation of Proximal Gradient Descent for Regularized Linear Models (Lasso $L_1$, Ridge $L_2$, and ElasticNet).
* **Classification & Risk Modeling:** Logistic Regression classifier with custom decision thresholds ($\tau = 0.30$) to flag high-probability logistics delays.
* **Data Leakage Mitigation:** Strict split-before-scaling protocols preventing distribution bleed across temporal folds.
* **Multi-Carrier Telemetry:** Comparative benchmark across Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, and Shadowfax.

---

## 📊 Dataset Summary

* **Dataset:** [Indian Multi-Agency E-Commerce Logistics & EDD Dataset (Version 2)](https://www.kaggle.com/datasets/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd?select=india_multi_agency_edd_dataset_v2.csv) (Created by Santanu Samanta)
* **Scale:** `210,850` shipment rows $\times$ `21` attributes
* **Logistics Carriers:** Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, Shadowfax
* **Spatial Routing:** Zones A to E across pan-India 6-digit pincodes
* **Transport Modes:** `Surface` and `Air`
* **Festive Surges:** Diwali Peak, Durga Puja / Dussehra, Eid-ul-Fitr, Holi, Independence Day Sale, New Year Carnival, and Normal Period

---

### 🌐 Preprocessing Pipeline & Data Leakage Prevention

To ensure unbiased out-of-sample generalization on the holdout test partition ($m_{\text{test}} = 42{,}170$ rows, 80/20 sequential split):

* **Data Purging:** Non-causal post-delivery realization variables (`Rating_Stars`, `Actual_Delivery_Days`) and high-cardinality/non-predictive identifiers (`Order_ID`, `Origin_Pincode`, `Destination_Pincode`, `Review_Text`) were strictly purged prior to constructing the design matrix $\mathbf{X}$.

---

* **Missing Value Imputation Protocol**
  Structural missingness in seasonal attributes (`Festival_Peak`, $147{,}569$ null records) was categorically encoded as `"Normal_Period"`. Exploratory conditional median imputation conducted on user feedback metrics was strictly isolated from downstream training manifolds to prevent distribution bleed.

---

* **Categorical Expansion & Strict In-Sample Feature Scaling**
  Categorical streams were one-hot encoded with reference category elimination (`drop_first=True`) to avoid the dummy variable trap and matrix singularity, expanding the feature space to $d = 41$ predictive dimensions:

$$\mathbf{X} \in \mathbb{R}^{m \times 41}$$

  Continuous features were standard-scaled using parameters derived exclusively from the training set:

$$\mu_j = \frac{1}{m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} x_{i,j}, \qquad \sigma_j = \sqrt{\frac{1}{m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} (x_{i,j} - \mu_j)^2}$$

$$x_{i,j}^{(\text{scaled})} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$

  The precomputed training statistics $(\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}})$ were then deterministically mapped across the test slice $\mathbf{X}_{\text{test}}$.

---

## 📐 Mathematical Formulations & Algorithms

### 1. Backtracking Line Search (Armijo Condition)

Dynamically contracts the step size $\eta$ using parameters $\beta = 0.5$ and $c = 0.005$ to guarantee monotonic Mean Squared Error reduction without overshooting:

$$f(\mathbf{w} - \eta \nabla f(\mathbf{w})) \le f(\mathbf{w}) - c \, \eta \, \|\nabla f(\mathbf{w})\|_2^2$$

If violated, update: $\eta \leftarrow \beta \eta$.

### 2. Proximal Gradient Descent for Lasso ($L_1$ Regularization)

To handle the non-differentiable $L_1$ penalty in:

$$\min_{\mathbf{w}} \frac{1}{2m} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_1$$

We apply the Soft-Thresholding operator element-wise to each parameter $v_j$:

$$\mathcal{S}_{\eta \alpha}(v_j) = \operatorname{sign}(v_j) \cdot \max\left(0, |v_j| - \eta \alpha\right)$$

Hyperparameter grid search spanned $\alpha \in [10^{-5}, 10^{1}]$.

### 3. Binary Logistic Regression for Delay Risk Classification

Logistics delay risk ($y_i \in \{0, 1\}$) was modeled using the Sigmoid activation:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \operatorname{clip}(\mathbf{w}^T \mathbf{x} + b, -500, 500)$$

Optimized via Binary Cross-Entropy (BCE) Loss:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \Big]$$

---

## 📈 Benchmark Results & Performance

### 📌 Regression Performance (EDD Prediction in Hours)
| Model Architecture | Test RMSE (Hours) | Test $R^2$ Score | Notes |
| :--- | :---: | :---: | :--- |
| **Custom Proximal Lasso (Scratch)** | **8.7003** | **0.9280** | $\alpha = 0.0316$ (selected via validation), prunes **30/41** features to zero |
| **Scikit-Learn Lasso ($L_1$)** | 8.7012 | 0.9280 | $\alpha = 0.001$ |
| **Ridge Regression ($L_2$)** | 8.7014 | 0.9280 | $\alpha = 0.01$ |
| **ElasticNet ($L_1 + L_2$)** | 8.7345 | 0.9274 | $\alpha = 0.01$, $l_1\text{ ratio} = 0.5$ |

### 📌 Classification Performance (Delay Flag)
| Model / Metric | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Custom Logistic Regression (Scratch, $\tau = 0.30$)** | **0.7812** | **0.8645** | **0.8207** | **0.9140** |

* **Confusion Matrix (Custom Logistic, $\tau = 0.30$):**
  * **True Positives (TP):** 11,240
  * **False Positives (FP):** 3,148
  * **True Negatives (TN):** 24,910
  * **False Negatives (FN):** 2,872
