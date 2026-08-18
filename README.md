# Machine Learning Assignment 2

## A. Problem Statement

The objective of this assignment is to implement classification models
using a public dataset and compare their performance using different
evaluation metrics.

An interactive Streamlit application was developed to allow users to
upload test data, select a model, generate predictions, and view model
evaluation results.

---

## B. Dataset Description

The Breast Cancer Wisconsin Diagnostic Dataset was obtained from the
UCI Machine Learning Repository.

Dataset details:

- Number of instances: 569
- Number of features: 30
- Problem type: Binary classification
- Feature columns: 30 numerical features
- Target column: target
- Class 0: Benign
- Class 1: Malignant

The dataset satisfies the minimum requirements of at least 500 instances
and 12 features.

The dataset was divided into 80% training data and 20% test data using a
stratified split.

---

## C. GitHub Repository Link

GitHub Repository:

PASTE_YOUR_GITHUB_LINK_HERE

The repository contains:

- app.py
- train_all.py
- requirements.txt
- README.md
- test_data.csv
- Python files for each model in the model directory
- Saved model files in the model directory

---

## D. Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Ensemble
6. Support Vector Machine

The first five models are the models listed in the assignment. SVM was
added as the sixth model because the assignment states that six models
must be implemented.

---

## E. Evaluation Metrics

The following metrics were calculated for every model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient

---

## F. Model Comparison Table

Copy the actual values from:

```text
model/comparison_results.csv