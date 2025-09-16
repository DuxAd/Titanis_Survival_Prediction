# Titanic Survival Prediction

## Description
This project uses machine learning techniques to predict the survival of Titanic passengers using the Kaggle dataset (which can be found here : https://www.kaggle.com/c/titanic/data). The goal is to identify key factors influencing survival and build high-performing models.

The project includes:
- **Data Exploration**: Visualizations of numerical variables and categorical variables using Seaborn.
- **Preprocessing**: Imputation of missing values, one-hot encoding and discretization.
- **Feature Engineering**: Creation of features and interactions. Multiple feature combinations were tested.
- **Statistical Tests**: Chi-square tests to validate the association between features.
- **Modeling**: Training of five models (DecisionTree, RandomForest, SVM, XGBoost, KNN) with hyperparameter optimization via `GridSearchCV`. Parameter ranges were progressively narrowed for precision.
- **Evaluation**: Metrics (accuracy, precision, recall, F1-score, ROC-AUC), cross-validation, and normalized confusion matrices.
