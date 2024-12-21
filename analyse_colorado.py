import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

# Chemin vers le fichier CSV
file_features = './colorado_nevada/acsincome_co_features.csv'
file_labels = './colorado_nevada/acsincome_co_label.csv'

df_feat_init = pd.read_csv(file_features)
size = len(df_feat_init['SEX'])
df_lab = pd.read_csv(file_labels)

sc = joblib.load('scaler.joblib')
df_feat_scaled = sc.transform(df_feat_init)

df_feat = pd.DataFrame(df_feat_scaled, columns=df_feat_init.columns)
df_lab['PINCP'] = df_lab['PINCP'].astype(int) 

# print(f"size :{size} ")

# Repartition des ages
# plt.hist(df_feat_init['AGEP'], bins=range(0,100,5), edgecolor='black', color='lightgreen')
# plt.title("Age Distribution")
# plt.xlabel("Age Ranges")
# plt.ylabel("Frequency")
# plt.show()

# Pourcentage d'hommes et pourcentage de femmes
# values = df_feat_init['SEX']
# print("REPARTITION DES SEXES\n")
# print(f"Pourcentage de femmes : {(values.value_counts()[2.0])*100/size}")
# print(f"Pourcentages d'hommes : {(values.value_counts()[1.0])*100/size}")

# Répartition des niveaux d'éducation
# values, counts = np.unique(df_featinit['SCHL'],return_counts=True)
# plt.bar(values, counts, align='center', alpha=0.7, color='lightgreen', edgecolor='black')
# plt.title("Class of educational attainment Distribution")
# plt.xlabel("Class of educational attainment Ranges")
# plt.ylabel("Frequency")
# plt.show()

rf=joblib.load('RandomForest_BestModel_08166.joblib')
ab=joblib.load('AdaBoost_BestModel_08151.joblib')
gb=joblib.load('GradientBoosting_BestModel_08119.joblib')

predictions_rf = rf.predict(df_feat)
predictions_ab = ab.predict(df_feat)
predictions_gb = gb.predict(df_feat)

predictions_rf_int = predictions_rf.astype(int)
predictions_ab_int = predictions_ab.astype(int)
predictions_gb_int = predictions_gb.astype(int)

# RandomForest

accuracy = accuracy_score(df_lab['PINCP'],predictions_rf_int)
print(f"Accuracy default RandomForest : {accuracy}")
creport = classification_report(df_lab['PINCP'],predictions_rf_int)
print(f"Classification default RandomForest :\n {creport}")

matrix=confusion_matrix(df_lab['PINCP'],predictions_rf_int)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix RandomForest on Colorado")
plt.show()

# AdaBoost

accuracy = accuracy_score(df_lab['PINCP'],predictions_ab_int)
print(f"Accuracy default AdaBoost : {accuracy}")
creport = classification_report(df_lab['PINCP'],predictions_ab_int)
print(f"Classification default AdaBoost :\n {creport}")

matrix=confusion_matrix(df_lab['PINCP'],predictions_ab_int)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix AdaBoost on Colorado")
plt.show()

# GradientBoosting

accuracy = accuracy_score(df_lab['PINCP'],predictions_gb_int)
print(f"Accuracy default GradientBoosting : {accuracy}")
creport = classification_report(df_lab['PINCP'],predictions_gb_int)
print(f"Classification default GradientBoosting :\n {creport}")

matrix=confusion_matrix(df_lab['PINCP'],predictions_gb_int)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.title("Confusion matrix for GradientBoosting")
plt.show()





