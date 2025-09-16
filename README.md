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


## Résultats

| Model                 | Accuracy (Test) | F1-Score | ROC-AUC | Precision | Recall | Std-Dev    CV | Accuracy (CV) |
|-----------------------|-----------------|----------|---------|-----------|--------|---------------|---------------|
| DecisionTree          | 0.822           | 0.769    | 0.813   | 0.807     | 0.733  | 0.053         | 0.813 ± 0.053 |
| RandomForest          | 0.822           | 0.756    | 0.897   | 0.845     | 0.683  | 0.055         | 0.832 ± 0.055 |
| SVM                   | 0.613           | 0.080    | 0.530   | 1.000     | 0.042  | 0.057         | 0.821 ± 0.057 |
| XGBoost               | 0.828           | 0.769    | 0.889   | 0.842     | 0.708  | 0.056         | 0.832 ± 0.056 |
| KNN                   | 0.801           | 0.757    | 0.857   | 0.748     | 0.767  | 0.053         | 0.818 ± 0.053 |

### **Quick Analysis of Results**
- **XGBoost** is the best model (test accuracy=0.828, CV=0.832), followed by **DecisionTree** and **RandomForest** (0.822 test).
- **SVM** performs poorly (accuracy=0.613, recall=0.042).
- **KNN** has good recall (0.767) but lower test accuracy (0.801).
- **RandomForest** has an excellent ROC-AUC (0.897) but limited recall (0.683), suggesting issues with detecting survivors due to class imbalance (~60% non-survivors, 40% survivors).
