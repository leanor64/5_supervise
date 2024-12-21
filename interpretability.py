import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

# Chemin vers le fichier CSV
file_features = './californie/alt_acsincome_ca_features_85(1).csv'
file_labels = './californie/alt_acsincome_ca_labels_85.csv'

df_feat = pd.read_csv(file_features)
df_lab = pd.read_csv(file_labels)

# Matrice de correlation initiale

#df_lab['PINCP'] = df_lab['PINCP'].astype(int) 
#df_feat['PINCP'] = df_lab['PINCP']

# corr = df_feat.corr()

# cax = plt.matshow(corr, cmap='coolwarm')
# plt.colorbar(cax)
# plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
# plt.yticks(np.arange(len(corr.columns)), corr.columns)
# plt.title('Correlation matrix')
# plt.show()

# Matrice de correlation comparaison modèles

rf=joblib.load('RandomForest_BestModel_08166.joblib')
ab=joblib.load('AdaBoost_BestModel_08151.joblib')
gb=joblib.load('GradientBoosting_BestModel_08119.joblib')


predictions_rf = rf.predict(df_feat)
predictions_ab = ab.predict(df_feat)
predictions_gb = gb.predict(df_feat)

print(f"predab : {predictions_ab} ")

df_lab['PINCP'] = df_lab['PINCP'].astype(int)  
df_feat['PINCP'] = df_lab['PINCP']

df_feat['PINCP_RF'] = predictions_rf
df_feat['PINCP_AB'] = predictions_ab
df_feat['PINCP_GB'] = predictions_gb

corr = df_feat.corr()

cax = plt.matshow(corr, cmap='coolwarm')
plt.colorbar(cax)
plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(np.arange(len(corr.columns)), corr.columns)
plt.title('Correlation matrix')
plt.show()

