from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Preprocess(df, method):
    ## Filling the missing data 
    titles = ['Miss.', 'Master', 'Mr.', 'Mrs.', 'Dr.']
    for i in titles:
        df.loc[df['Name'].str.contains(i), 'Title'] = i
    
    if method == "Knn":
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']])
    else :
        for i in titles:
            ## Determining the most common age by title 
            Most_common = df[df['Name'].str.contains(i)]['Age'].mode()[0]
            df.loc[df['Name'].str.contains(i) & df['Age'].isna(), 'Age'] = Most_common

    mode_embarked = df['Embarked_encoded'].mode()[0]
    fill_values = {"Title": 'Rare', "Embarked_encoded": mode_embarked}
    df = df.fillna(value=fill_values)
    
    #df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=False)

    df['FareCat'] = pd.qcut(df['Fare'], 4) #4 yeld the best results
    df['AgeCat'] = pd.qcut(df['Age'], 3)
    df = pd.get_dummies(df, columns=['Embarked', 'Title'], prefix=['Embarked','Title'], drop_first=False)
    
    le = LabelEncoder()
    df['FareCat'] = le.fit_transform(df['FareCat'])  
    df['AgeCat'] = le.fit_transform(df['AgeCat'])     
   
    
    df['PClass_age'] = df['Pclass'] * df['Age']
    df['Sex_age'] = df['Sex_encoded'] * df['Age']
    df['Group_age'] = df['Group'] * df['Age']
    df['Alone_sex'] = df['Sex_encoded'] * df['Seul']
    
    return df
    
def AffichageRes(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
        
    # Évaluer les performances du modèle
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    scores = cross_val_score(model, X_train, y_train, cv = 10)


    print('\n--------------------------------------------')
    print("\nRésultats du modele", model.__class__.__name__)
    print(f'Accuracy : {accuracy:.3f}, Precision : {precision:.3f}, recall :{recall:.3f}, F1 :{f1:.3f}')
    print(f'ROC_AUC : {roc_auc}')
    print('Score de validation Croisée')
    print(f'Score : {np.round(scores,3)}')
    print(f"Accuracy moyenne : {scores.mean():.3f}")
    print(f"Écart-type : {scores.std():.3}")
    print('\n--------------------------------------------\n')

    # cm = confusion_matrix(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                  normalize = 'true',
                                  #title = "Confusion Matrix "
                                  display_labels = ["Died", "Survived"])

    #cm_display.plot()
    ax = plt.gca() # GCA = Get Current Axes
    ax.set_title(f"Matrice de Confusion de {model.__class__.__name__}")

    plt.show()
