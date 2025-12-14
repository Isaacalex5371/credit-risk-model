# Credit Risk Probability Model for Bati Bank

## Project Overview
Bati Bank is partnering with an eCommerce company to offer a Buy-Now-Pay-Later service. This project aims to build a Credit Scoring Model to estimate the likelihood of default for customers who lack traditional credit histories.

## Credit Scoring Business Understanding

### 1. The Influence of Basel II Accord
The Basel II Capital Accord emphasizes rigorous risk measurement to determine the minimum capital a bank must hold. For our project, this means our model cannot just be a "black box." It requires:
*   **Interpretability:** We must explain *why* a customer was rejected or approved.
*   **Documentation:** Every step of the feature engineering and modeling process must be traceable.
*   **Compliance:** The risk estimates (Probability of Default) must be robust enough to withstand regulatory scrutiny.

### 2. Necessity of a Proxy Variable
Since the dataset lacks a direct "Defaulted" (Yes/No) label, we must engineer a proxy variable. We will use RFM (Recency, Frequency, Monetary) analysis to classify users.
*   **Why:** We assume that customers with low frequency and low monetary value are "high risk" and less likely to repay.
*   **Business Risk:** The risk of this approach is **misclassification**. A low-spending customer might be financially responsible but just inactive. If we label them "High Risk" incorrectly, the bank loses a potential good customer (Type II error).

### 3. Model Selection Trade-offs
*   **Logistic Regression (with WoE):** 
    *   *Pros:* Highly interpretable, easy to explain to regulators, standard in banking.
    *   *Cons:* May miss complex, non-linear patterns in data.
*   **Gradient Boosting (XGBoost/LightGBM):**
    *   *Pros:* High predictive performance, handles complex relationships well.
    *   *Cons:* "Black box" nature makes it harder to explain individual decisions to stakeholders, requiring tools like SHAP for explainability.