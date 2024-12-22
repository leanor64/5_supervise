import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

# Chemin vers le fichier CSV
file_features = './californie/alt_acsincome_ca_features_85(1).csv'
file_labels = './californie/alt_acsincome_ca_labels_85.csv'

df_feat = pd.read_csv(file_features)
size = len(df_feat['SEX'])
df_lab = pd.read_csv(file_labels)


# SPLIT TRAIN TEST part
X_train, X_test, y_train, y_test = train_test_split(df_feat, df_lab, test_size=0.4, random_state=13)

y_train = y_train['PINCP'].astype(int) 

params = {'n_estimators' : range(10,110,20),
          'learning_rate': [0.2, 0.6, 1.0, 1.4]}

X_inter = X_test.copy()
X_inter['PINCP'] = y_test['PINCP'].astype(int)
X1 = X_inter.copy()
X2 = X_inter.copy()
hommes_X_test = X1[X1['SEX'] == 1]
femmes_X_test = X2[X2['SEX'] == 2]

hommes_X_test = hommes_X_test.drop(columns=['SEX'])
femmes_X_test = femmes_X_test.drop(columns=['SEX'])

y_test_h = hommes_X_test['PINCP']
y_test_f = femmes_X_test['PINCP']
hommes_X_test = hommes_X_test.drop(columns=['PINCP'])
femmes_X_test = femmes_X_test.drop(columns=['PINCP'])

X_train = X_train.drop(columns=['SEX'])
X_test = X_test.drop(columns=['SEX'])

# Preparation des donnees
# sc = StandardScaler()
# X_train_scaled = sc.fit_transform(X_train)
sc = joblib.load('scaler_gender.joblib')
X_test_scaled = sc.transform(X_test)
hommes_X_test_scaled = sc.transform(hommes_X_test)
femmes_X_test_scaled = sc.transform(femmes_X_test)


# grid = GridSearchCV(AdaBoostClassifier(), params, cv=5)
# grid.fit(X_train_scaled, y_train)
# print(f"Le meilleur score est : {grid.best_score_}")
# print(f"Les meilleur paramètres sont : {grid.best_params_}")
# print(f"Le meilleur estimateur est : {grid.best_estimator_}")

# joblib.dump(grid.best_estimator_,'AdaBoost_Without_Gender_.joblib')

#AdaBoost test best model

abwg=joblib.load('AdaBoost_Without_Gender_08132.joblib')
y_pred = abwg.predict(X_test_scaled)
y_pred_int = y_pred.astype(int)

matrix=confusion_matrix(y_test,y_pred_int)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix for AdaBoost without gender column")
plt.show()

# Matrices par genre

predictions_h = abwg.predict(hommes_X_test_scaled)
predictionsh = predictions_h.astype(int)

matrix=confusion_matrix(predictionsh,y_test_h)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix AdaBoost without gender on Men")
plt.show()


predictions_f = abwg.predict(femmes_X_test_scaled)
predictionsf = predictions_f.astype(int)

matrix=confusion_matrix(predictionsf,y_test_f)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix AdaBoost without gender on Women")
plt.show()