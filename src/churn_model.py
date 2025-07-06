"""
Wrapper para aplicar un umbral fijo a las probabilidades
de un estimador binario (por ejemplo, XGBClassifier).
"""

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
