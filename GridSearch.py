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
sc = StandardScaler()
sc = joblib.load('scaler.joblib')
X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)

labels = df_lab['PINCP'].map({'True':1.0, 'False':0.0})


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

rf=joblib.load('RandomForest_BestModel_08166.joblib')
ab=joblib.load('AdaBoost_BestModel_08151.joblib')
gb=joblib.load('GradientBoosting_BestModel_08119.joblib')

estimators = [('rf', rf),('ab', ab), ('gb', gb)]
sc = StackingClassifier(estimators)
sc.fit(X_train_scaled,y_train)
cv_score = cross_val_score(sc, X_train_scaled, y_train, cv=5)
y_pred=sc.predict(X_test_scaled)

print(f"\nCross-Validation score optimal Stacking : {np.mean(cv_score)}")
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy optimal Stacking : {accuracy}")
creport = classification_report(y_test,y_pred)
print(f"Classification optimal Stacking :\n {creport}")
matrix=confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix for our optimal Stacking")
plt.show()

# TODO : penser à renommer le fichier joblib
joblib.dump(sc,'Stacking_BestModel_.joblib')

# TODO : Temps d'exécution pour Stacking