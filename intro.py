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


# Repartition des ages
# plt.hist(df_feat['AGEP'], bins=range(0,100,5), edgecolor='black', color='skyblue')
# plt.title("Age Distribution")
# plt.xlabel("Age Ranges")
# plt.ylabel("Frequency")
# plt.show()

# # Pourcentage d'hommes et pourcentage de femmes
# values = df_feat['SEX']
# print("REPARTITION DES SEXES\n")
# print(f"Pourcentage de femmes : {(values.value_counts()[2.0])*100/size}")
# print(f"Pourcentages d'hommes : {(values.value_counts()[1.0])*100/size}")


# Repartition des catégories professionnelles
# values, counts = np.unique(df_feat['COW'],return_counts=True)
# plt.bar(values, counts, align='center', alpha=0.7, color='skyblue', edgecolor='black')
# plt.title("Class of worker Distribution")
# plt.xlabel("Class of worker Ranges")
# plt.ylabel("Frequency")
# plt.show()

# Pour compter le nombre de personnes prise en compte dans COW (pour trouver nb N/A)
# sum = 0
# for v in counts :
#     sum += v
# print(f"nbr total de personne COW : {sum}")
# print(f"type 1 : {counts[0]}")


# Repartition des races

# values, counts = np.unique(df_feat['RAC1P'],return_counts=True)
# plt.bar(values, counts, align='center', alpha=0.7, color='skyblue', edgecolor='black')
# plt.title("Race Distribution")
# plt.xlabel("Race Ranges")
# plt.ylabel("Frequency")
# plt.show()
# print(f"type 1 : {counts[0]}")


# Repartition des statuts maritaux
# values, counts = np.unique(df_feat['MAR'],return_counts=True)
# plt.bar(values, counts, align='center', alpha=0.7, color='skyblue', edgecolor='black')
# plt.title("Class of marital status Distribution")
# plt.xlabel("Class of marital status Ranges")
# plt.ylabel("Frequency")
# plt.show()

# Répartition des niveaux d'éducation
values, counts = np.unique(df_feat['SCHL'],return_counts=True)
plt.bar(values, counts, align='center', alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Class of educational attainment Distribution")
plt.xlabel("Class of educational attainment Ranges")
plt.ylabel("Frequency")
plt.show()


# SPLIT TRAIN TEST part
X_train, X_test, y_train, y_test = train_test_split(df_feat, df_lab, test_size=0.4, random_state=13)


# Preparation des donnees
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
joblib.dump(sc,'scaler.joblib')
X_test_scaled = sc.transform(X_test)

labels = df_lab['PINCP'].map({'True':1.0, 'False':0.0})


#  Cross-validation

# Paramètres par défaut

# RandomForest 

# rfc=RandomForestClassifier()
# rfc.fit(X_train_scaled,y_train)
# cv_score = cross_val_score(rfc, X_train_scaled, y_train, cv=5)
# y_pred = rfc.predict(X_test_scaled)

# print(f"\nCross-Validation score default RandomForest : {np.mean(cv_score)}")
# accuracy = accuracy_score(y_test,y_pred)
# print(f"Accuracy default RandomForest : {accuracy}")
# creport = classification_report(y_test,y_pred)
# print(f"Classification default RandomForest :\n {creport}")

# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for default RandomForest")
# plt.show()


# TODO : Temps d'exécution pour RandomForest 


# AdaBoost

# abc=AdaBoostClassifier()
# abc.fit(X_train_scaled,y_train)
# cv_score = cross_val_score(abc, X_train_scaled, y_train, cv=5)
# y_pred=abc.predict(X_test_scaled)

# print(f"\nCross-Validation score default AdaBoost : {np.mean(cv_score)}")
# accuracy = accuracy_score(y_test,y_pred)
# print(f"Accuracy default AdaBoost : {accuracy}")
# creport = classification_report(y_test,y_pred)
# print(f"Classification default AdaBoost :\n {creport}")
# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for default AdaBoost")
# plt.show()


# TODO : Temps d'exécution pour AdaBoost 



# TODO : GradientBoosting

# gbc = GradientBoostingClassifier()
# gbc.fit(X_train_scaled,y_train)
# cv_score = cross_val_score(gbc, X_train_scaled, y_train, cv=5)
# y_pred=gbc.predict(X_test_scaled)

# print(f"\nCross-Validation score default GradientBoosting : {np.mean(cv_score)}")
# accuracy = accuracy_score(y_test,y_pred)
# print(f"Accuracy default GradientBoosting : {accuracy}")
# creport = classification_report(y_test,y_pred)
# print(f"Classification default GradientBoosting :\n {creport}")
# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for default GradientBoosting")
# plt.show()

# TODO : Temps d'exécution pour GradientBoosting



# TODO : Une méthode de Stacking

# estimators = [('rf', RandomForestClassifier()),('ab', AdaBoostClassifier())]
# sc = StackingClassifier(estimators)
# sc.fit(X_train_scaled,y_train)
# cv_score = cross_val_score(sc, X_train_scaled, y_train, cv=5)
# y_pred=sc.predict(X_test_scaled)

# print(f"\nCross-Validation score default Stacking : {np.mean(cv_score)}")
# accuracy = accuracy_score(y_test,y_pred)
# print(f"Accuracy default Stacking : {accuracy}")
# creport = classification_report(y_test,y_pred)
# print(f"Classification default Stacking :\n {creport}")
# matrix=confusion_matrix(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.title("Confusion matrix for default Stacking")
# plt.show()


# TODO : Temps d'exécution pour Stacking


