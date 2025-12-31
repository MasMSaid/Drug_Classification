# Drug Prescription Classification (Multiclass Machine Learning)

## 📌 Project Overview

This project applies supervised machine learning techniques to predict the most suitable drug for a patient based on medical attributes.

The dataset contains demographic and clinical features, and the target variable represents one of five prescribed drugs.

Note: This project is for educational purposes only and does not replace professional medical advice.

## 📂 Dataset

| Feature     | Type        | Description            |
| ----------- | ----------- | ---------------------- |
| Age         | Numeric     | Patient age            |
| Sex         | Categorical | M / F                  |
| BP          | Categorical | Blood pressure         |
| Cholesterol | Categorical | HIGH / NORMAL          |
| Na_to_K     | Numeric     | Sodium–Potassium ratio |
| Drug        | Target      | Prescribed drug        |

Source: Drug200.csv


## ⚙️ Models Implemented

- Gaussian Naive Bayes

- Multinomial Naive Bayes

- Bernoulli Naive Bayes

- Logistic Regression

- K-Nearest Neighbours (KNN)

- Support Vector Machine (SVM)

- Decision Tree

- Random Forest

- XGBoost (eXtreme Gradient Boosting)


## 🧪 Evaluation Strategy

- Train-test split (70/30)

- Stratified sampling to preserve class distribution

- Hyperparameter tuning with GridSearchCV

- Accuracy as primary metric


## 🏆 Key Findings

- Best-performing models: Decision Tree, Random Forest, and XGBoost achieved 100% test accuracy.

- Strong alternatives: SVM (95%) and Logistic Regression (93.33%) also performed well and offer better generalization.

- Naive Bayes insights: Gaussian NB (93.33%) is suitable for continuous features; Multinomial NB (60%) and Bernoulli NB (43.33%) are not appropriate for this dataset.

- KNN performance: 90% accuracy; sensitive to feature scaling and choice of neighbors.

- Dataset characteristics: High accuracy across multiple models suggests the classification problem is relatively straightforward.

- Caution: Perfect accuracy in tree-based models may indicate potential overfitting due to the small dataset size.


## 📊 Model Performance Comparison

| Model                           | Test Accuracy | Top Feature(s) / Notes               |
| ------------------------------- | ------------- | ------------------------------------ |
| Decision Tree                   | 100%          | Na_to_K ratio, Age                   |
| Random Forest                   | 100%          | Na_to_K ratio, Age, BP               |
| XGBoost                         | 100%          | Na_to_K ratio, Age, Cholesterol      |
| Support Vector Classifier (SVC) | 95%           | Na_to_K ratio, Age (linear kernel)   |
| Logistic Regression             | 93.33%        | Na_to_K ratio, Age                   |
| Gaussian Naive Bayes            | 93.33%        | Age, Na_to_K ratio                   |
| K-Nearest Neighbors (KNN)       | 90%           | Distance-based, all scaled features  |
| Multinomial Naive Bayes         | 60%           | Not suitable for continuous features |
| Bernoulli Naive Bayes           | 43.33%        | Not suitable for continuous features |



## 🛠️ Technologies Used

- Python

- Scikit-learn

- XGBoost

- Pandas, NumPy, Matplotlib, Seaborn


## 🚀 How to Run

pip install -r requirements.txt

jupyter notebook Drug_Prescription_Multiclass_Classification.ipynb


## ✅ Summary

This project demonstrates a full workflow for multiclass classification: data preprocessing, model comparison, hyperparameter tuning, evaluation, and feature importance analysis. It provides hands-on experience with both linear and non-linear models, ensemble methods, and distance-based classifiers—making it a strong portfolio piece for machine learning and data science roles.
