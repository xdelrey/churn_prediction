# importamos las librerías necesarias
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import joblib

# Lectura de datos y muestra inicial
df = pd.read_csv('..\\data\\processed\\churn_clean_v2_py.csv')

# Definimos Features y Target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Dividimos los datos en Train (80%) y Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20,
                                                    random_state=10)

# Guardado de datos Train y Test
X_train.to_csv('..\\data\\train\\churn_X_train_v2_py.csv', index=False, encoding='utf-8')
X_test.to_csv('..\\data\\test\\churn_X_test_v2_py.csv', index=False, encoding='utf-8')

y_train.to_csv('..\\data\\train\\churn_y_train_v2_py.csv', index=False, encoding='utf-8')
y_test.to_csv('..\\data\\test\\churn_y_test_v2_py.csv', index=False, encoding='utf-8')

############################# ESCALADO  ############################# 
# variables numéricas
numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']

# escalado
scaler = StandardScaler()

X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric]  = scaler.transform(X_test[numeric])

########################### MODELO FINAL ############################
######################## XG BOOST CLASSIFIER ########################
#####################################################################

# configuración del modelo XG Boost Classifier
xgb = XGBClassifier(
        objective='binary:logistic', # típico para un problema binario (Churn / No Churn) y salida en forma de probabilidad (0-1)
        eval_metric='auc',
        scale_pos_weight=3, # 75 : 25 ≈ 3 (balance del target)
        random_state=10
    )

# Grid Search
param_grid_3 = {
    'n_estimators': [300, 400, 500, 1000],
    'learning_rate': [0.015, 0.02, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6]
}

cv_3 = StratifiedKFold(
    n_splits=5, 
    shuffle=True, 
    random_state=10
    )

grid_3 = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_3,
    scoring='recall',
    cv=cv_3,
    verbose=2,
    n_jobs=-1
)

grid_3.fit(X_train, y_train)

################ guardamos el modelo ################
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
    
# Wrapper con el mejor estimador y el threshold óptimo
final_model = ChurnModel(grid_3.best_estimator_, threshold=0.466)

# guardamos el pkl
joblib.dump({
    "model"     : final_model,
    "scaler"    : scaler,
    "cont_cols" : numeric,
    "columns"   : X_train.columns
}, "../models/final_model_v1_py.pkl")