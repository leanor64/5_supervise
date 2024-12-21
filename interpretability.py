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

df_lab['PINCP'] = df_lab['PINCP'].astype(int)  # Convertir les booléens en 1 (True) et 0 (False)
df_feat['PINCP'] = df_lab['PINCP']



#f, ax = plt.subplots(figsize=(10, 8))
corr = df_feat.corr()

cax = plt.matshow(corr, cmap='coolwarm')
plt.colorbar(cax)
plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(np.arange(len(corr.columns)), corr.columns)
plt.title('Correlation matrix')
plt.show()

