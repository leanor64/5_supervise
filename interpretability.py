import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

# Chemin vers le fichier CSV
file_features = './californie/alt_acsincome_ca_features_85(1).csv'
file_labels = './californie/alt_acsincome_ca_labels_85.csv'

df_feat_init = pd.read_csv(file_features)
df_lab = pd.read_csv(file_labels)

sc = joblib.load('scaler.joblib')
df_feat_scaled = sc.transform(df_feat_init)

df_feat = pd.DataFrame(df_feat_scaled, columns=df_feat_init.columns)


# Matrice de correlation initiale

df_lab['PINCP'] = df_lab['PINCP'].astype(int) 
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

predictions_rf_int = predictions_rf.astype(int)
predictions_ab_int = predictions_ab.astype(int)
predictions_gb_int = predictions_gb.astype(int)



# df_lab['PINCP'] = df_lab['PINCP'].astype(int)  
# df_feat['PINCP'] = df_lab['PINCP']

# df_feat['PINCP_RF'] = predictions_rf_int
# df_feat['PINCP_AB'] = predictions_ab_int
# df_feat['PINCP_GB'] = predictions_gb_int

# corr = df_feat.corr()

# cax = plt.matshow(corr, cmap='coolwarm')
# plt.colorbar(cax)
# plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
# plt.yticks(np.arange(len(corr.columns)), corr.columns)
# plt.title('Correlation matrix')
# plt.show()


# feature importance


# feature_names = df_feat_init.columns
# mdi_importances = pd.Series(
#     rf.feature_importances_, index=feature_names
# ).sort_values(ascending=True)

# ax = mdi_importances.plot.barh()
# ax.set_title("Random Forest Feature Importances (MDI)")
# ax.figure.tight_layout()
# plt.show()


result = permutation_importance(
    rf, df_feat, df_lab['PINCP'], n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=df_feat.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances for RandomForest")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()


result = permutation_importance(
    ab, df_feat, df_lab['PINCP'], n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=df_feat.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances for AdaBoost")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()



result = permutation_importance(
    gb, df_feat, df_lab['PINCP'], n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=df_feat.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances for GradientBoosting")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()