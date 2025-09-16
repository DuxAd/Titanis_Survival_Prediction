import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import MyFunction
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

### Loading the data
df = pd.read_csv('train.csv')
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])
df['Embarked_encoded'] = le.fit_transform(df['Embarked']) # encoding for data visualisation
df['Group'] = df['SibSp']  + df['Parch']  + 1
df['Seul'] = df['Group'] <= 1

Key_numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Group', 'Fare', 'Embarked_encoded']
Key_categorical = ['Sex', 'Seul']
for i in Key_numeric:
    plt.figure()
    plt.hist([df[df['Survived'] == 1][i],df[df['Survived'] == 0][i]], label=['Survived','Died'], bins=10,stacked=True,density=True)
    plt.title('Survie en fonction de '+ i)
    plt.xlabel(i)
    plt.ylabel('Fréquence')
    plt.legend()

for i in Key_categorical:
    plt.figure()
    sns.countplot(x=i, hue='Survived', data=df)
    plt.title('Survie en fonction de ' + i)
    plt.xlabel(i)
    plt.ylabel('Nombre')
    plt.legend(title='Survived', labels=['Died', 'Survived'])
    plt.show()


# Number of Missing
print("########## Missing Values before preprocess ##########")
for i in df.keys() :
    print( i , " Missing : ", df[i].isna().sum())
print("Total :", len(df))
print()

##### Preprocess
df = MyFunction.Preprocess(df, 'Infer')

##### Correlation 
import seaborn as sns

df_num = df.select_dtypes(include=['int', 'float', 'bool'])
df_num = df_num.drop(['Embarked_C', 'Embarked_Q', 'Embarked_S',
'Title_Dr.', 'Title_Master', 'Title_Miss.', 'Title_Mr.', 'Title_Mrs.',
'Title_Rare', 'Group_age'], axis=1)
correlation_matrix = df_num.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, 
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5,
            mask = np.triu(np.ones_like(correlation_matrix)))      # Add lines between cells

#### Contingency test #### 
from scipy.stats import chi2_contingency

print("\n####################################")
for i in ['FareCat', 'AgeCat']:
    print("\nContingency table for ", i)
    contingency_table = pd.crosstab(df['Survived'], df[i])
    print(contingency_table)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-2 statistics : {chi2}")
    print(f"p-value : {p_value}")

##### train/test split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
############################################################################################
############################################################################################
X = df[['Pclass', 'Sex_encoded', 'Age', 'Seul', 'FareCat',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'Title_Dr.', 'Title_Master', 'Title_Miss.', 'Title_Mr.', 'Title_Mrs.',
        'Title_Rare', 'PClass_age', 'Sex_age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3333, random_state=42)

##### DecisionTree
print("\n####################################")
print("Results for the different models :")
print("####################################\n")
from sklearn.model_selection import GridSearchCV

X_train_DT = X_train.drop(['Sex_age'], axis = 1)
X_test_DT = X_test.drop(['Sex_age'], axis = 1)

param_grid = {
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 5, 10, 15],
    'max_leaf_nodes': [22,25,27],
    'class_weight': [None, 'balanced', {0:1, 1:1.2}, {0:1, 1:1.5}]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_DT, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)

# Using class_weight='balanced' lower the true_négatives and improve true_positives
# But Lower the overall performance
clf = DecisionTreeClassifier(max_leaf_nodes=25, min_samples_split=5, max_depth=None,  random_state=42)
clf.fit(X_train_DT, y_train)

MyFunction.AffichageRes(clf, X_train_DT, X_test_DT, y_train, y_test)


feature_importance = pd.DataFrame({
    'feature': X_train_DT.columns,
    'importance': clf.feature_importances_,
}).sort_values('importance', ascending=False)
print("\n####################################")
print("Feature importance for the DecisionTree")
print(feature_importance)
print("\n####################################")

# Tree representation
from sklearn.tree import DecisionTreeClassifier, export_text
text_representation = export_text(clf, feature_names=list(X_train_DT.columns))
#print(text_representation)


##### Random Forest ######
from sklearn.ensemble import RandomForestClassifier

X_train_RF = X_train.drop(['PClass_age', 'Sex_age'], axis = 1)
X_test_RF = X_test.drop(['PClass_age', 'Sex_age'], axis = 1)

param_grid = {
    'min_samples_split': [2, 3],
    'max_depth': [5,7, 9],
    'n_estimators': [10,35,75],
    'min_samples_leaf': [2,5,8],
   # 'class_weight': [None, 'balanced', {0:1, 1:1.2}, {0:1, 1:1.5}]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_RF, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)

# Same comment concerning class_weight='balanced', 
rfc = RandomForestClassifier(class_weight=None, min_samples_leaf = 5,n_estimators = 35, min_samples_split=2, max_depth=9, random_state=42)
rfc.fit(X_train_RF, y_train)

MyFunction.AffichageRes(rfc, X_train_RF, X_test_RF, y_train, y_test)


feature_importance = pd.DataFrame({
    'feature': X_train_RF.columns,
    'importance': rfc.feature_importances_
}).sort_values('importance', ascending=False)
print("\n####################################")
print("Feature importance for the RandomForestClassifier")
print(feature_importance)
print("\n####################################")


##### Support Vector Machine #####
from sklearn import svm

X_train_SVM = X_train.drop(['PClass_age', 'Sex_age'], axis = 1)
X_test_SVM = X_test.drop(['PClass_age', 'Sex_age'], axis = 1)
X_train_SVM['Age'] = X_train_SVM['Age']/X_train_SVM['Age'].max()
X_test_SVM['Age'] = X_test_SVM['Age']/X_train_SVM['Age'].max()
X_train_SVM['FareCat'] = X_train_SVM['FareCat']/X_train_SVM['FareCat'].max()
X_test_SVM['FareCat'] = X_test_SVM['FareCat']/X_train_SVM['FareCat'].max()

param_grid_SVC = {
    'kernel': ['linear'],
    'C': [1,10,15,20 ],
    #'gamma': ['scale', 0.1, 0.01],
    'class_weight': [None, 'balanced', {0:1, 1:1.2}]
}
grid_search = GridSearchCV(svm.SVC(random_state=42), param_grid_SVC, cv=15, scoring='accuracy')
grid_search.fit(X_train_SVM, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)

SVC = svm.SVC(probability=True, kernel='linear', C=1, random_state=42, class_weight='balanced')
SVC.fit(X_train_SVM, y_train)

MyFunction.AffichageRes(SVC, X_train_SVM, X_test_SVM, y_train, y_test)



#### XGBOOST 
import xgboost as xgb

X_train_XGB = X_train.drop(['PClass_age', 'Sex_age'], axis = 1)
X_test_XGB = X_test.drop(['PClass_age', 'Sex_age'], axis = 1)

param_grid = {
    'n_estimators': [90, 100, 110],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 6, 9]
}
grid_search = GridSearchCV(xgb.XGBClassifier(random_state=42), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_XGB, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)

# Same comment, adding scaling reduces the accuracy
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model_xgb.fit(X_train_XGB, y_train)

MyFunction.AffichageRes(model_xgb, X_train_XGB, X_test_XGB, y_train, y_test)

feature_importance = pd.DataFrame({
    'feature': X_train_XGB.columns,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)
print("\n####################################")
print("Feature importance for the RandomForestClassifier")
print(feature_importance)
print("\n####################################")

##### Knn #####
from sklearn.neighbors import KNeighborsClassifier

X_train_KNN = X_train.drop(['PClass_age'], axis = 1)
X_test_KNN = X_test.drop(['PClass_age'], axis = 1)
X_test_KNN['Age'] = X_test_KNN['Age']/X_train_KNN['Age'].max()
X_train_KNN['Age'] = X_train_KNN['Age']/X_train_KNN['Age'].max()
X_test_KNN['Sex_age'] = X_test_KNN['Sex_age']/X_train_KNN['Sex_age'].max()
X_train_KNN['Sex_age'] = X_train_KNN['Sex_age']/X_train_KNN['Sex_age'].max()

param_grid = {
    'n_neighbors': [1,3,5, 10],
    'weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_KNN, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)


knn = KNeighborsClassifier(n_neighbors =5, weights='uniform')
knn.fit(X_train_KNN, y_train)

MyFunction.AffichageRes(knn, X_train_KNN, X_test_KNN, y_train, y_test)
