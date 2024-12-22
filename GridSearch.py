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

# Preparation des donnees
# sc = StandardScaler()
# sc = joblib.load('scaler.joblib')
# X_train_scaled = sc.transform(X_train)
# X_test_scaled = sc.transform(X_test)

# labels = df_lab['PINCP'].map({'True':1.0, 'False':0.0})


# Amélioration des modèles (3.2)

# RandomForest

# params = {'n_estimators' : range(100,180,20),
#           'max_depth': (None, 10, 15, 20, 25),
#           'min_samples_leaf' : range(20,180, 40)}

# grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
# time_before = time.time()
# grid.fit(X_train_scaled, y_train)
# time_after = time.time()


# # Temps d'exécution pour RandomForest
# time_exec = time_after - time_before
# print(f"Temps d'exécution pour RandomForest : {time_exec}")
# print(f"Le meilleur score est : {grid.best_score_}")
# print(f"Les meilleur paramètres sont : {grid.best_params_}")
# print(f"Le meilleur estimateur est : {grid.best_estimator_}")

# # TODO : penser à renommer le fichier joblib
# joblib.dump(grid.best_estimator_,'RandomForest_BestModel_.joblib')

# RandomForest test best model

# rfc=joblib.load('RandomForest_BestModel_08166.joblib')
# y_pred = rfc.predict(X_test_scaled)

# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for our best RandomForest")
# plt.show()

# AdaBoost

# params = {'n_estimators' : range(10,110,20),
#           'learning_rate': [0.2, 0.6, 1.0, 1.4]}

# grid = GridSearchCV(AdaBoostClassifier(), params, cv=5)
# time_before = time.time()
# grid.fit(X_train_scaled, y_train)
# time_after = time.time()


# # Temps d'exécution pour AdaBoost
# time_exec = time_after - time_before
# print(f"Temps d'exécution pour AdaBoost : {time_exec}")
# print(f"Le meilleur score est : {grid.best_score_}")
# print(f"Les meilleurs paramètres sont : {grid.best_params_}")
# print(f"Le meilleur estimateur est : {grid.best_estimator_}")

# #TODO : penser à renommer le fichier joblib
# joblib.dump(grid.best_estimator_,'AdaBoost_BestModel_ .joblib')

# AdaBoost test best model

# rfc=joblib.load('AdaBoost_BestModel_08151.joblib')
# y_pred = rfc.predict(X_test_scaled)

# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for our best AdaBoost")
# plt.show()

# GradientBoosting

# params = {'n_estimators' : range(10,90,40),
#           'learning_rate': [0.6, 1.0, 1.4],
#           'max_depth' : [10, 15]
#           }

# grid = GridSearchCV(GradientBoostingClassifier(), params, cv=5)
# time_before = time.time()
# grid.fit(X_train_scaled, y_train)
# time_after = time.time()


# # Temps d'exécution pour GradientBoosting
# time_exec = time_after - time_before
# print(f"Temps d'exécution pour GradientBoosting : {time_exec}")
# print(f"Le meilleur score est : {grid.best_score_}")
# print(f"Les meilleurs paramètres sont : {grid.best_params_}")
# print(f"Le meilleur estimateur est : {grid.best_estimator_}")

# #TODO : penser à renommer le fichier joblib
# joblib.dump(grid.best_estimator_,'GradientBoosting_BestModel_ .joblib')

# GradientBoosting test best model
# rfc=joblib.load('GradientBoosting_BestModel_08119.joblib')
# y_pred = rfc.predict(X_test_scaled)

# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for our best GradientBoosting")
# plt.show()



# TODO : Une méthode de Stacking

# rf=joblib.load('RandomForest_BestModel_08166.joblib')
# ab=joblib.load('AdaBoost_BestModel_08151.joblib')
# gb=joblib.load('GradientBoosting_BestModel_08119.joblib')

# estimators = [('rf', rf),('ab', ab), ('gb', gb)]
# sc = StackingClassifier(estimators)
# sc.fit(X_train_scaled,y_train)
# cv_score = cross_val_score(sc, X_train_scaled, y_train, cv=5)
# y_pred=sc.predict(X_test_scaled)

# print(f"\nCross-Validation score optimal Stacking : {np.mean(cv_score)}")
# accuracy = accuracy_score(y_test,y_pred)
# print(f"Accuracy optimal Stacking : {accuracy}")
# creport = classification_report(y_test,y_pred)
# print(f"Classification optimal Stacking :\n {creport}")
# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for our optimal Stacking")
# plt.show()

# TODO : penser à renommer le fichier joblib
# joblib.dump(sc,'Stacking_BestModel_.joblib')

# TODO : Temps d'exécution pour Stacking




# EQUITE

# Taux d'individu >= 50000$
taux_1_y_train = (y_train == 1).mean()
print(f"Taux de valeurs '1' dans y_train : {taux_1_y_train}")

taux_0_y_train = (y_train == 0).mean()
print(f"Taux de valeurs '0' dans y_train : {taux_0_y_train}")


# Taux d'homme >= 50000$
X_train['PINCP'] = y_train['PINCP']
hommes_X_train = X_train[X_train['SEX'] == 1]
taux_hommes_label_1 = (hommes_X_train['PINCP'] == 1).mean()
print(f"Taux d'hommes ayant le label '1' : {taux_hommes_label_1}")


# Taux de femmes >= 50000$

femmes_X_train = X_train[X_train['SEX'] == 2]
taux_femmes_label_1 = (femmes_X_train['PINCP'] == 1).mean()
print(f"Taux de femmes ayant le label '1' : {taux_femmes_label_1}")


# Matrices par genre

ab=joblib.load('AdaBoost_BestModel_08151.joblib')

X_test['PINCP'] = y_test['PINCP'].astype(int)
print(f"Xtest :{X_test} ")
X1 = X_test.copy()
print(f"X1 :{X1} ")
X2 = X_test.copy()
hommes_X_test = X1[X1['SEX'] == 1]
femmes_X_test = X2[X2['SEX'] == 2]

y_test_h = hommes_X_test['PINCP']
y_test_f = femmes_X_test['PINCP']
hommes_X_test = hommes_X_test.drop(columns=['PINCP'])
femmes_X_test = femmes_X_test.drop(columns=['PINCP'])

print(f"hommes :{hommes_X_test} ")

# Preparation des donnees
sc = joblib.load('scaler.joblib')
hommes_X_test_scaled = sc.transform(hommes_X_test)
femmes_X_test_scaled = sc.transform(femmes_X_test)


predictions_h = ab.predict(hommes_X_test_scaled)
predictionsh = predictions_h.astype(int)

matrix=confusion_matrix(predictionsh,y_test_h)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix AdaBoost on Men")
plt.show()


predictions_f = ab.predict(femmes_X_test_scaled)
predictionsf = predictions_f.astype(int)

matrix=confusion_matrix(predictionsf,y_test_f)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix AdaBoost on Women")
plt.show()