# Credit Risk Probability Model 🏦

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Interim_Submission-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Bati Bank Credit Scoring Project**  
An end-to-end Machine Learning pipeline to estimate credit risk for unbanked customers using alternative data (eCommerce transactions).

---

## 📖 Project Overview

Bati Bank is partnering with an eCommerce platform to enable a **Buy-Now-Pay-Later (BNPL)** service. Since many customers lack a traditional credit history (FICO score), this project aims to build a data-driven **Credit Scoring Model**.

The goal is to calculate a **Risk Probability** for each customer based on their transaction behavior, allowing the bank to make informed lending decisions.

---

## 🧠 Credit Scoring Business Understanding (Task 1)

### 1. The Influence of Basel II Accord
The Basel II Capital Accord sets international standards for how much capital banks must hold to guard against financial and operational risks. For this project, Basel II compliance dictates that our model cannot be a simple "black box."
*   **Risk Measurement:** We must rigorously estimate the **Probability of Default (PD)** to calculate the Expected Loss (EL).
*   **Interpretability:** The model must be transparent. Regulators require us to explain *why* a customer was classified as high-risk.
*   **Documentation:** Every step of the feature engineering and modeling process must be traceable and reproducible.

### 2. The Proxy Variable Strategy
**The Problem:** The dataset provided contains transaction history but lacks a specific "Defaulted" (Yes/No) label.
**The Solution:** We are engineering a **Proxy Variable** to define "Good" vs. "Bad" borrowers using **RFM Analysis** (Recency, Frequency, Monetary).
*   **Method:** We will use **K-Means Clustering** to segment users. Customers with low frequency and low monetary value (and high recency) will be labeled as "High Risk."
*   **Business Risk:** The primary risk is **Misclassification (Type II Error)**. A customer might be financially sound but simply inactive on this specific platform. Labeling them "High Risk" prevents the bank from acquiring a potentially profitable customer.

### 3. Model Selection Trade-offs
*   **Logistic Regression (with WoE):**
    *   *Pros:* Highly interpretable, produces a scorecard format standard in banking, computationally efficient.
    *   *Cons:* May fail to capture complex, non-linear relationships in behavioral data.
*   **Gradient Boosting (XGBoost/LightGBM):**
    *   *Pros:* Superior predictive performance, handles non-linear patterns and interactions well.
    *   *Cons:* "Black box" nature makes it harder to explain to compliance officers. We would need SHAP values to explain decisions, adding complexity to the deployment pipeline.

---

## 📂 Project Structure

This project follows a production-ready structure to ensure code quality and reproducibility.

```text
credit-risk-model/
├── .github/workflows/   # CI/CD Pipelines (Linting & Testing)
├── data/
│   ├── raw/             # Original dataset (Xente Challenge)
│   └── processed/       # Cleaned data for modeling
├── notebooks/
│   └── eda.ipynb        # Exploratory Data Analysis & Insights
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Data loading & cleaning classes
│   ├── train.py            # Model training & saving functions
│   └── predict.py          # Inference script
├── tests/               # Unit tests for data integrity
├── Dockerfile           # Containerization for API
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

🚀 Getting Started
1. Prerequisites
Python 3.8+
Git
2. Installation
Clone the repository and install dependencies:

git clone https://github.com/isaac/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt

3. Running the Analysis
To view the Exploratory Data Analysis:

jupyter notebook notebooks/eda.ipynb
To run the data processing pipeline (Code Quality Check):

python src/data_processing.py