# Customer Churn Prediction

## 📌 Project Overview

This project focuses on predicting customer churn using classical machine learning models and ensemble methods.
The main goal is to correctly identify customers who are likely to churn, which is a minority class, making recall and F1-score for the positive class especially important.

The project includes:

- Data preprocessing with a unified pipeline
- A baseline model for reference
- Ensemble models (Random Forest and LightGBM)
- Model evaluation, visualization, and experiment tracking with MLflow

## 📊 Data Preprocessing

The preprocessing step was designed as part of a pipeline to ensure reproducibility and avoid data leakage.

Key preprocessing decisions:
- Numerical features were not normally distributed, but this did not affect model performance as tree-based models were used.
- Binary categorical features with values "Yes" and "No" were converted into 1 and 0.
- Multi-class categorical features were encoded using One-Hot Encoding.
- No significant outliers were detected in the dataset.

All preprocessing steps were applied consistently using a pipeline.

## 🤖 Models

#### Baseline Model
- **Logistic Regression** was used as a baseline to establish a minimum performance reference.

#### Ensemble Models
- **Random Forest (Bagging-based)**: Provided better performance for churn detection, especially in terms of recall and ROC-AUC, making it a strong candidate for this problem.

- **LightGBM (Boosting-based)**: Used to compare bagging vs. boosting approaches. Boosting improved ROC-AUC and recall when properly tuned and class imbalance was taken into account.

Although LightGBM achieved strong overall separability, Random Forest proved to be more robust for detecting churn in this dataset.

## 📈 Evaluation Metrics

The following metrics were used to evaluate model performance:

- **Accuracy**: Percentage of total correct predictions. Indicates general classification performance.

- **ROC-AUC**: Area Under the ROC Curve. Measures the model’s ability to separate classes.

- **F1-macro**: Average F1-score across all classes. A high value indicates that the model does not focus only on the majority class.

- **Precision (Yes)**: Of all customers predicted as churn, how many actually churned.

- **Recall (Yes)**: Of all customers who actually churned, how many were correctly detected by the model.

- **F1-score (Yes)**: Harmonic mean of precision and recall for the churn class. Indicates whether the model is well-balanced for the most important class.

## 🎯 Metric Strategy

Since churn is the minority class, the evaluation focused primarily on:

- Recall (Yes) → minimizing false negatives

- F1-score (Yes) → balance between precision and recall

- ROC-AUC → overall class separability

Accuracy alone was not considered sufficient for model selection.

## 🔍 Model Comparison

The project compares:

- Bagging-based ensembles (Random Forest)

- Boosting-based models (LightGBM)

This comparison highlights how boosting can improve recall and ROC-AUC, while bagging can offer more stable performance depending on the data distribution.

## 🧪 Experiment Tracking

All experiments were tracked using MLflow, including:

- Model parameters

- Metrics

- Artifacts (models, plots)

This allows easy comparison between runs and ensures reproducibility.

## ✅ Conclusion

Logistic Regression serves as a useful baseline but is insufficient for churn detection.

Random Forest offers the best balance for this dataset, especially in recall and ROC-AUC.

LightGBM demonstrates the strengths of boosting and provides competitive results when properly tuned.

Metric selection is critical when dealing with imbalanced data.
