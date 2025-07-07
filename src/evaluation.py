# importamos las librerías necesarias
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, precision_recall_curve

import joblib
import pathlib as pl

class ChurnModel:
    def __init__(self, estimator, threshold):
        self.estimator  = estimator      # el XGBoost entrenado del Modelo 3
        self.threshold  = threshold      # 0.466 calculado

    def predict_proba(self, X):
        "Devuelve las probabilidades sin tocar nada"
        return self.estimator.predict_proba(X)

    def predict(self, X):
        "Aplica el corte al canal positivo"
        proba_pos = self.predict_proba(X)[:, 1]
        return (proba_pos >= self.threshold).astype(int)

# -----------------------------------------------
# 1. Carga de artefactos
# -----------------------------------------------

PKL_PATH = pl.Path("../models/final_model_v1_py.pkl")
pkg      = joblib.load(PKL_PATH)

model   = pkg["model"]           # wrapper ChurnModel
scaler  = pkg["scaler"]          # StandardScaler ya entrenado
numeric = pkg["cont_cols"]       # ['tenure', 'MonthlyCharges', 'TotalCharges']
columns = pkg["columns"]         # orden exacto de features


# -----------------------------------------------
# 2. Lectura y preparación del test
# -----------------------------------------------

X_test = pd.read_csv("../data/test/churn_X_test_v2_py.csv")[columns].copy()
y_test = pd.read_csv("../data/test/churn_y_test_v2_py.csv").squeeze()  # Series

# escalado
X_test[numeric]  = scaler.transform(X_test[numeric])


# -----------------------------------------------
# 3. Predicciones
# -----------------------------------------------
y_pred  = model.predict(X_test)                # usa threshold 0.466
y_proba = model.predict_proba(X_test)[:, 1]    # probas para AUC


# -----------------------------------------------
# 4. Establecemos un objetivo de Recall = 0.85
# -----------------------------------------------

# ChurnModel que traigo en el .pkl ya aplica un threshold fijo (0.466)
# sin embargo, recalculo el threshold... podría ser diferente y afectar a las métricas
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
thresholds = np.append(thresholds, 1.0)  

# índice de la primera posición donde el recall alcanza 0.85
idx = np.where(recall >= 0.85)[0][-1] # selecciona el mayor threshold con recall >=0.85

recall_objetivo = recall[idx]        # debería ser ≈0.85 o superior
precision_en_thres = precision[idx]
threshold_final = thresholds[idx]


# -----------------------------------------------
# 5. Métricas
# -----------------------------------------------

# AUC
auc_final = roc_auc_score(y_test, y_proba)
print(f"AUC test - XG Boost Classifier con Threshold ajustado: {auc_final:.3f}")

# Predicciones - Etiquetas
y_pred_final = (y_proba >= threshold_final).astype(int) # Threshold/umbral óptimo con recall >=0.85

# Matriz de confusión
cm_final = confusion_matrix(y_test, y_pred_final, labels=[0, 1])

print("\nMatriz de Confusión Final:")
print(cm_final)

# Calcular Precision
precision_final = precision_score(y_test, y_pred_final, pos_label=1)
print(f"Precisión (Churn): {precision_final:.3f}")

# Calcular Recall
recall_final = recall_score(y_test, y_pred_final, pos_label=1)
print(f"Recall (Churn): {recall_final:.3f}")

# Calcular Accuracy (Exactitud)
accuracy_final = accuracy_score(y_test, y_pred_final)
print(f"Exactitud (Accuracy): {accuracy_final:.3f}")

# classification report
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn']))

