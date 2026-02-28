# Bank Loan Default Prediction – Classification Model Comparison

![Python](https://img.shields.io/badge/Python-3.9-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.5-green)

This project applies and compares three popular classification algorithms—**K‑Nearest Neighbors (KNN)**, **Logistic Regression (LR)**, and **Naive Bayes (Gaussian, Multinomial, Bernoulli)**—to predict whether a bank loan applicant will default. The dataset contains 700 records with 8 features (age, education, employment years, address, income, debt ratio, credit card debt, other debt). Extensive feature discretization and hyperparameter tuning are performed to optimize model performance.

## 🔍 Project Overview
- **Goal**: Build a robust classifier to predict loan default (`违约` = 1) using customer demographic and financial information.
- **Dataset**: 700 rows, 9 columns (8 features + binary target). No missing values.
- **Key Steps**:
  - **Exploratory Data Analysis**: Correlation analysis to understand feature relationships.
  - **Feature Discretization**:
    - Equal‑frequency binning for `年龄` (age), `教育` (education), `工龄` (employment years) using quartiles.
    - Custom binning for `地址` (address), `收入` (income), `负债率` (debt ratio), `信用卡负债` (credit card debt), `其他负债` (other debt) based on percentiles and domain knowledge.
    - All features converted to categorical integers (0–5) for model compatibility.
  - **Model Training & Tuning**:
    - **KNN**: Grid search over `n_neighbors` (5–14) and `weights` ('uniform', 'distance').
    - **Logistic Regression**: Exhaustive search over `penalty` ('l1', 'l2', 'elasticnet', 'none'), `solver` ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), `multi_class` ('ovr', 'multinomial'), and `fit_intercept`.
    - **Naive Bayes**: Gaussian, Multinomial, and Bernoulli variants with default parameters.
  - **Evaluation**: Accuracy on a 20% hold‑out test set.

## 📁 Repository Structure
bank-loan-default-classification/
├── data/ # – bankloan.csv
├── notebooks/ # Jupyter notebook with the full analysis
│ └── classification_comparison.ipynb
├── requirements.txt # Python dependencies
├── README.md # This file
├── LICENSE # MIT License
└── .gitignore # Ignores data, checkpoints, etc.

text

## 🚀 How to Run
1. Clone this repository.
2. Install required packages:
   ```bash
   pip install -r requirements.txt
Place the dataset bankloan.csv in the data/ folder (or update the path in the notebook).

Run the notebook notebooks/classification_comparison.ipynb to reproduce the analysis and results.

📊 Results
After extensive tuning, the best performing model was Logistic Regression with the following configuration:

penalty='none' (no regularization)

solver='lbfgs'

multi_class='multinomial'

fit_intercept=True

Achieved test accuracy: 80.0%.

Model	Best Accuracy
Logistic Regression	80.0%
Multinomial Naive Bayes	77.1%
KNN (k=11, uniform)	74.3%
Gaussian Naive Bayes	75.7%
Bernoulli Naive Bayes	75.0%
🛠️ Technologies Used
Python 3.9

pandas, numpy

scikit-learn (KNN, Logistic Regression, Naive Bayes, model selection)

matplotlib (for histograms, though not extensively used)

📌 Key Takeaways
Discretizing continuous features can improve model interpretability and performance for certain algorithms.

Logistic Regression with no regularization and multinomial loss generalized best on this dataset.

KNN performance plateaued around 74% with optimal k ~11–14.

Naive Bayes variants showed competitive results (75–77%), with MultinomialNB slightly outperforming the others.

📄 License
MIT
