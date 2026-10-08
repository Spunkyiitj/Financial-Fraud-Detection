# 💳 Financial Transaction Fraud Detection & Analytics

A machine learning-based financial fraud detection project that analyzes
millions of financial transactions, identifies fraud patterns, trains a
classification model, and provides an interactive Streamlit application
for real-time fraud prediction.

## 🚀 Project Overview

Financial institutions process millions of transactions every day.
Detecting fraudulent transactions manually is difficult and inefficient.

This project builds an end-to-end fraud detection workflow:

**Transaction Data → Data Analysis → Feature Analysis → Preprocessing →
Machine Learning → Evaluation → Streamlit Deployment**

The project focuses on identifying potentially fraudulent transactions
while handling the highly imbalanced nature of financial fraud data.

------------------------------------------------------------------------

## 🎯 Objectives

The main objectives of this project are to:

-   Analyze a large financial transaction dataset.
-   Understand transaction and fraud patterns.
-   Identify transaction types associated with higher fraud rates.
-   Analyze transaction amounts and account balances.
-   Build a machine learning classification pipeline.
-   Handle severe class imbalance.
-   Evaluate the model using fraud-focused metrics.
-   Deploy the trained model through a Streamlit application.
-   Provide real-time fraud predictions for new transactions.

------------------------------------------------------------------------

## 📊 Dataset

The dataset contains:

-   **6,362,620 transactions**
-   **11 columns/features**
-   **8,213 fraudulent transactions**
-   **6,354,407 legitimate transactions**
-   Approximately **0.13%** of transactions are fraudulent.

### Dataset Features

  Feature            Description
  ------------------ --------------------------------------------
  `step`             Time step associated with the transaction
  `type`             Transaction type
  `amount`           Transaction amount
  `nameOrig`         Sender account identifier
  `oldbalanceOrg`    Sender balance before the transaction
  `newbalanceOrig`   Sender balance after the transaction
  `nameDest`         Receiver account identifier
  `oldbalanceDest`   Receiver balance before the transaction
  `newbalanceDest`   Receiver balance after the transaction
  `isFraud`          Target variable: 1 = Fraud, 0 = Legitimate
  `isFlaggedFraud`   Existing fraud flag in the dataset

> The original dataset is not included in this GitHub repository because
> of its large file size.

------------------------------------------------------------------------

## 🔎 Exploratory Data Analysis

Several analyses were performed to understand the dataset and identify
potential fraud patterns.

### Fraud Distribution

Fraud represents only approximately **0.13%** of all transactions.

This creates a severe class imbalance problem, making accuracy alone an
unreliable evaluation metric.

### Fraud by Transaction Type

Observed fraud rates:

  Transaction Type     Fraud Rate
  ------------------ ------------
  TRANSFER                0.7688%
  CASH_OUT                0.1840%
  CASH_IN                      0%
  DEBIT                        0%
  PAYMENT                      0%

The analysis indicates that fraudulent transactions are concentrated
primarily in **TRANSFER** and **CASH_OUT** transactions.

### Transaction Amount Analysis

The transaction amount distribution is highly skewed.

Approximate statistics:

-   Mean: **179,861**
-   Median: **74,871**
-   Maximum: **92,445,516**

A logarithmic transformation using `log1p()` was used for visualization
of the highly skewed transaction amounts.

### Balance Analysis

Two balance-difference features were explored:

``` python
balancedifforig = oldbalanceOrg - newbalanceOrig
balancediffdest = newbalanceDest - oldbalanceDest
```

These features were used during analysis to investigate changes in
sender and receiver balances.

The analysis found:

-   **1,399,253** transactions with a negative sender balance
    difference.
-   **1,238,864** transactions with a negative receiver balance
    difference.
-   **1,188,074** TRANSFER/CASH_OUT transactions where the sender had a
    positive balance before the transaction and zero balance afterward.

### Time Analysis

The `step` feature was analyzed to understand how fraudulent activity
varies over time. The time variable was used for exploratory analysis
and removed before the final baseline model.

------------------------------------------------------------------------

## 🧹 Data Preparation

The following identifier/existing-flag columns were removed from the
baseline model:

``` text
nameOrig
nameDest
isFlaggedFraud
```

The target variable is:

``` text
isFraud
```

where:

``` text
0 = Legitimate
1 = Fraud
```

### Model Features

Numerical features:

``` text
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

Categorical feature:

``` text
type
```

------------------------------------------------------------------------

## ⚙️ Machine Learning Pipeline

The project uses a Scikit-learn preprocessing and classification
pipeline.

### Pipeline

``` text
Raw Transaction
       ↓
Feature Selection
       ↓
 ┌─────┴─────┐
 ↓           ↓
Numerical   Categorical
Features     Feature
 ↓           ↓
Standard    One-Hot
Scaler      Encoder
 └─────┬─────┘
       ↓
Logistic Regression
       ↓
Fraud Prediction
```

### Numerical Preprocessing

`StandardScaler` is used to standardize numerical features.

### Categorical Preprocessing

`OneHotEncoder(drop="first")` converts the transaction type into
numerical features.

------------------------------------------------------------------------

## 🤖 Machine Learning Model

The baseline model is **Logistic Regression**.

The model uses:

``` python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

### Why `class_weight="balanced"`?

Fraud represents only around **0.13%** of the transactions.

Without addressing class imbalance, a model could strongly favor the
majority legitimate class.

`class_weight="balanced"` gives greater importance to the minority fraud
class during model training.

------------------------------------------------------------------------

## 🧪 Train-Test Split

The dataset was divided using a stratified train-test split:

-   **70% training data**
-   **30% testing data**

Stratification was used to preserve the fraud/legitimate class
distribution between the training and testing sets.

------------------------------------------------------------------------

## 📈 Model Performance

### Overall Accuracy

**94.70%**

However, because the dataset is extremely imbalanced, accuracy is not
sufficient to judge fraud detection performance.

### Classification Performance

  Class          Precision   Recall   F1-Score
  ------------ ----------- -------- ----------
  Legitimate          1.00     0.95       0.97
  Fraud               0.02     0.93       0.04

### Key Result

The model achieves approximately **93% recall for fraudulent
transactions**.

This means the baseline model is able to identify most fraudulent
transactions in the test set.

However, fraud precision is approximately **2%**, which means the model
generates a large number of false-positive alerts.

------------------------------------------------------------------------

## 📊 Confusion Matrix

The model produced the following confusion matrix on the test set:

``` text
                 Predicted
                 Legitimate   Fraud

Actual Legitimate   1,805,224   101,098
Actual Fraud              163     2,301
```

### Interpretation

-   **True Negatives:** 1,805,224
-   **False Positives:** 101,098
-   **False Negatives:** 163
-   **True Positives:** 2,301

The model successfully detects a large proportion of fraudulent
transactions, but reducing false positives remains a major area for
improvement.

------------------------------------------------------------------------

## ⚠️ Current Limitations

This project is a baseline fraud detection implementation and is **not
production-ready**.

The main limitation is the low fraud precision and high number of false
positives.

In a real financial institution, excessive false alerts can:

-   Increase manual investigation workload.
-   Cause unnecessary transaction reviews.
-   Reduce customer experience.
-   Increase operational costs.

Therefore, the model needs further optimization before being considered
for real-world financial decision-making.

------------------------------------------------------------------------

## 🚀 Streamlit Application

The trained machine learning pipeline is saved as:

``` text
fraud_detection_pipeline.pkl
```

The Streamlit application loads this trained pipeline and allows users
to enter transaction details.

### User Inputs

-   Transaction type
-   Transaction amount
-   Sender's old balance
-   Sender's new balance
-   Receiver's old balance
-   Receiver's new balance

The application then generates a prediction.

### Prediction

``` text
0 → Transaction appears legitimate
1 → Potential fraud detected
```

The application provides a simple interface for testing new transactions
without retraining the model.

------------------------------------------------------------------------

## 🏗️ Project Architecture

``` text
                Financial Transaction Dataset
                           │
                           ▼
                    Data Cleaning
                           │
                           ▼
              Exploratory Data Analysis
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    Fraud Analysis   Amount Analysis   Balance Analysis
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    Feature Selection
                           │
                           ▼
                    Train/Test Split
                           │
                           ▼
                   Data Preprocessing
                  ┌────────┴────────┐
                  ▼                 ▼
           StandardScaler      OneHotEncoder
                  │                 │
                  └────────┬────────┘
                           ▼
                  Logistic Regression
                   class_weight=balanced
                           │
                           ▼
                       Evaluation
                           │
                           ▼
              Saved ML Pipeline (.pkl)
                           │
                           ▼
                  Streamlit Application
                           │
                           ▼
               Real-Time Fraud Prediction
```

------------------------------------------------------------------------

## 📁 Repository Structure

``` text
fraud-detection/
│
├── app.py
├── fraud_detection_pipeline.pkl
├── fraud_detection_analysis.ipynb
├── requirements.txt
└── README.md
```

### Files

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit frontend and prediction
                                      application

  `fraud_detection_pipeline.pkl`      Trained Scikit-learn machine
                                      learning pipeline

  `fraud_detection_analysis.ipynb`    Data analysis, EDA, feature
                                      analysis and model development

  `requirements.txt`                  Python dependencies

  `README.md`                         Project documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Technologies Used

### Programming & Data Analysis

-   Python
-   Pandas
-   NumPy

### Visualization

-   Matplotlib
-   Seaborn

### Machine Learning

-   Scikit-learn
-   Logistic Regression
-   StandardScaler
-   OneHotEncoder
-   Imbalanced classification

### Deployment

-   Streamlit
-   Joblib

------------------------------------------------------------------------

## ▶️ Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Start the Streamlit application

``` bash
streamlit run app.py
```

The application will open in your browser.

------------------------------------------------------------------------

## 🌐 Deployment

The Streamlit application can be deployed using **Streamlit Community
Cloud**.

The deployment requires:

``` text
app.py
fraud_detection_pipeline.pkl
requirements.txt
```

The original large dataset does not need to be uploaded to the deployed
application because the trained machine learning pipeline has already
been saved.

------------------------------------------------------------------------

## 🔮 Future Improvements

### 1. Advanced Machine Learning Models

Compare the baseline Logistic Regression model with:

-   Random Forest
-   XGBoost
-   LightGBM
-   Gradient Boosting

### 2. Better Feature Engineering

Potential additional features:

``` text
Sender balance change
Receiver balance change
Transaction-to-balance ratio
Zero-balance indicator
Transaction frequency
Account transaction frequency
Amount deviation from normal behavior
```

### 3. Threshold Optimization

Test different probability thresholds to find a better balance between:

-   Fraud recall
-   Fraud precision
-   False positives
-   False negatives

### 4. Precision-Recall Analysis

Because the dataset is highly imbalanced, future versions should
include:

-   Precision-Recall curve
-   PR-AUC
-   Threshold analysis

### 5. Time-Based Validation

A future version could train on historical transactions and test on
later transactions to better simulate a real-world fraud detection
system.

### 6. Interactive Analytics Dashboard

A future version could include a Power BI or Streamlit dashboard
containing:

-   Total transactions
-   Fraud rate
-   Transaction volume
-   Fraud by transaction type
-   Fraud over time
-   High-risk transaction segments
-   Model performance
-   False-positive rate

------------------------------------------------------------------------

## 💼 Business Value

A fraud detection system can help financial institutions:

-   Identify potentially fraudulent transactions.
-   Prioritize suspicious transactions for investigation.
-   Automate initial transaction screening.
-   Identify high-risk transaction patterns.
-   Support risk monitoring.
-   Process large transaction volumes efficiently.
-   Assist analysts in making data-driven decisions.

The project demonstrates an end-to-end workflow from large-scale
transaction analysis to machine learning prediction and application
deployment.

------------------------------------------------------------------------

## 📌 Key Project Highlights

-   **6.36M+ transactions analyzed**
-   **8,213 fraud cases**
-   **0.13% fraud rate**
-   **94.70% overall test accuracy**
-   **\~93% fraud recall**
-   Imbalanced classification handling
-   Exploratory fraud pattern analysis
-   Feature preprocessing pipeline
-   Logistic Regression baseline model
-   Saved machine learning pipeline
-   Streamlit real-time prediction application

------------------------------------------------------------------------

## 🧠 Skills Demonstrated

### Data Analytics

-   Exploratory Data Analysis
-   Data Cleaning
-   Statistical Analysis
-   Data Visualization
-   Transaction Analysis
-   Fraud Pattern Analysis

### Machine Learning

-   Classification
-   Logistic Regression
-   Imbalanced Classification
-   Feature Preprocessing
-   Standardization
-   One-Hot Encoding
-   Model Evaluation
-   Confusion Matrix

### Deployment

-   Streamlit
-   Joblib
-   Scikit-learn Pipelines

### Business Analytics

-   Financial Transaction Analysis
-   Fraud Risk Analysis
-   Pattern Identification
-   Data-Driven Decision Support

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is developed for educational and portfolio purposes.

The model should not be used as a production financial fraud detection
system without additional validation, monitoring, security controls,
threshold optimization, model governance, and testing on real-world
financial data.

------------------------------------------------------------------------

## 👨‍💻 Author

**Prince Raj**

B.Tech --- Chemical Engineering\
Indian Institute of Technology Jodhpur

------------------------------------------------------------------------

⭐ If you found this project interesting, feel free to explore the
repository and connect with me.
