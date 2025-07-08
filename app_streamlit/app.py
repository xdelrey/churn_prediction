# app_streamlit/app.py

from pathlib import Path
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ────────────────────────────
# 1. Rutas del proyecto
# ────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent     # .../churn_prediction
SRC_DIR    = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"

# ────────────────────────────
# 2. Habilitar imports locales
# ────────────────────────────
sys.path.append(str(SRC_DIR))

import churn_model                       # módulo real en src/churn_model.py
sys.modules["ChurnModel"] = churn_model  # alias que pide pickle

# Parches adicionales que pickle podría necesitar
churn_model.dtype    = np.dtype
churn_model.ndarray  = np.ndarray

# ────────────────────────────
# 3. Cargar el modelo
# ────────────────────────────
MODEL_PATH = MODELS_DIR / "final_model_v0.pkl"

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

# Umbral de decisión: si el objeto lo trae, úsalo; si no, define uno manualmente
DEFAULT_THRESHOLD = 0.5
threshold = getattr(model, "threshold", DEFAULT_THRESHOLD)

# ────────────────────────────
# 4. Definir función de inferencia
# ────────────────────────────
def infer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve probabilidades de churn y predicción binaria en un DataFrame nuevo.
    Se asume que 'model' implementa .predict_proba, o que es un ndarray de probs.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:, 1]
    elif isinstance(model, np.ndarray):
        # Si el modelo es un array con las probabilidades de cada fila ya calculadas
        proba = model
    else:
        raise TypeError("El objeto cargado no implementa 'predict_proba' ni es un ndarray.")

    pred = (proba >= threshold).astype(int)

    df_out = df.copy()
    df_out["proba_churn"] = proba
    df_out["pred_churn"]  = pred
    return df_out

pkg      = joblib.load(MODEL_PATH)
model    = pkg["model"]          # wrapper ChurnModel (xgb + threshold 0.466)
scaler   = pkg["scaler"]         # StandardScaler entrenado
cont_cols = pkg["cont_cols"]     # ['tenure', 'MonthlyCharges', 'TotalCharges']
all_cols  = pkg["columns"]       # orden exacto que espera el modelo
threshold = getattr(model, "threshold", 0.5)

# ────────────────────────────
# 5. transformaciones
# ────────────────────────────
st.set_page_config(page_title="📡 Predicción de Baja de Clientes", page_icon="📡")

# One-hot y escalado //////////////////////////////////////////////////////////////////
def build_feature_row(user_dict: dict) -> pd.DataFrame:
    """Convierto las entradas en un DataFrame"""
    # arranco todo a 0
    row = {col: 0 for col in all_cols}

    # asigno numéricas y binarias
    for key, val in user_dict.items():
        if key in row:
            row[key] = val

    # one-hot: InternetService
    if user_dict["InternetService"] != "DSL":           # DSL es la baseline (drop_first=True)
        row[f"InternetService_{user_dict['InternetService']}"] = 1

    # one-hot: Contract
    if user_dict["Contract"] != "Month-to-month":
        row[f"Contract_{user_dict['Contract']}"] = 1

    # one-hot: PaymentMethod
    if user_dict["PaymentMethod"] != "Electronic check":
        row[f"PaymentMethod_{user_dict['PaymentMethod']}"] = 1

    df = pd.DataFrame([row])[all_cols].copy()

    # escalado de numéricas
    df[cont_cols] = scaler.transform(df[cont_cols])

    return df


# 3. ───────── UI ────────────────────────────────────────────
st.title("📊 Predicción de Baja de Clientes")

left_col, mid_col, right_col = st.columns([1.7, 1.7, 1.2])

# ───────── Izq: Datos demográficos y contrato ─────────────
with left_col:
    st.subheader("👤 Datos Básicos")
    is_female      = st.selectbox("Género",           ["Male", "Female"]) == "Female"
    senior         = st.selectbox("Senior", ["No", "Yes"])      == "Yes"
    partner        = st.selectbox("Pareja",        ["No", "Yes"])      == "Yes"
    dependents     = st.selectbox("Dependientes",     ["No", "Yes"])      == "Yes"

    st.subheader("📄 Detalles de la Cuenta")
    phone_service  = st.selectbox("Servicio: Teléfono",  ["No", "Yes"])      == "Yes"
    multiple_lines = st.selectbox("Varias Líneas", ["No phone service", "No", "Yes"])
    internet_plan  = st.selectbox("Servicio: Internet", ["DSL", "Fiber optic", "No"])
    contract       = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Método de Pago", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

# ───────── Medio: Servicios + Cargos ──────────────────────
with mid_col:
    st.subheader("🛠️ Servicios Adicionales")
    online_sec   = st.selectbox("Seguridad Online",   ["No internet service", "No", "Yes"])
    online_back  = st.selectbox("Backup Online",     ["No internet service", "No", "Yes"])
    device_prot  = st.selectbox("Protección Dispositivos", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Soporte Técnico",      ["No internet service", "No", "Yes"])
    stream_tv    = st.selectbox("Streaming TV",      ["No internet service", "No", "Yes"])
    stream_mov   = st.selectbox("Streaming Películas",  ["No internet service", "No", "Yes"])
    paperless    = st.selectbox("Facturación Paperless", ["No", "Yes"]) == "Yes"

    st.subheader("💵 Cargos")
    tenure        = st.slider("Meses de permanencia en el servicio",            0,  72, 12)
    monthly_chg   = st.slider("Facturación Mensual (€)",     15.0,150.0, 70.0)
    total_chg     = st.slider("Total Facturado (€)",         0.0,9000.0, 500.0)

# ───────── Build feature vector ─────────────────────────────
user_inputs = {
    # numéricas
    "tenure": tenure,
    "MonthlyCharges": monthly_chg,
    "TotalCharges": total_chg,
    # binarias directas
    "is_female": int(is_female),
    "SeniorCitizen": int(senior),
    "Partner": int(partner),
    "Dependents": int(dependents),
    "PhoneService": int(phone_service),
    "PaperlessBilling": int(paperless),
    # servicios binarios en Yes/No | No internet service
    "MultipleLines": 0 if multiple_lines == "No" else (0 if multiple_lines == "No phone service" else 1),
    "OnlineSecurity": 0 if online_sec  == "No" else (0 if online_sec  == "No internet service" else 1),
    "OnlineBackup":   0 if online_back == "No" else (0 if online_back == "No internet service" else 1),
    "DeviceProtection": 0 if device_prot == "No" else (0 if device_prot == "No internet service" else 1),
    "TechSupport":   0 if tech_support == "No" else (0 if tech_support == "No internet service" else 1),
    "StreamingTV":   0 if stream_tv    == "No" else (0 if stream_tv    == "No internet service" else 1),
    "StreamingMovies":0 if stream_mov   == "No" else (0 if stream_mov   == "No internet service" else 1),
    # categorías multi-clase
    "InternetService": internet_plan,
    "Contract": contract,
    "PaymentMethod": payment_method
}

X_pred = build_feature_row(user_inputs)

# ───────── Dcha: predicción + visualización ───────────────
with right_col:
    st.subheader("🔮 Predicción")
    if st.button("🚀 Predict Churn"):
        proba = model.predict_proba(X_pred)[:, 1]
        pred  = (proba >= threshold).astype(int)[0]

        if pred:
            st.error(f"🚨 **¡Alerta! Probabilidad de baja: {proba[0]:.2%}**")
        else:
            st.success(f"🎉 **Cliente fidelizado. Prob. de baja: {proba[0]:.2%}**")

        # Tabla para el usuario
        st.subheader("ℹ Valores introducidos")
        show_df = (
            pd.DataFrame({"Característica": X_pred.columns, "Valor": X_pred.iloc[0]})
              .query("Valor != 0")      # oculta los ceros para aligerar
              .reset_index(drop=True)
        )
        st.dataframe(show_df, use_container_width=True)

# ───────── Footer ───────────────────────────────────────────
st.markdown("---")
st.info("ℹ️ Este asistente utiliza un modelo XGBoost entrenado sobre el dataset "
        "Telco Customer Churn, con un umbral fijo de {:.3f}.".format(threshold))
