# Predicción de *Churn* en Telecom 📉📱

Proyecto de *Machine Learning* para predecir las bajas *(churn)* de clientes en una compañía de telecomunicaciones.

> "Adquirir un nuevo cliente es hasta 25 × más caro que retener uno existente." 🤑

---

## 🌐 Contexto del problema

Detectar con antelación qué clientes abandonarán el servicio permite dirigir campañas de retención y optimizar recursos. El objetivo principal fue **maximizar el *****Recall***** (≥ 0.85)** aun sacrificando *Precision*, minimizando así los *Falsos Negativos* 🛡️.

## 🗃️ Dataset

- Fuente: *WA\_Fn‑UseC\_Telco‑Customer‑Churn* (IBM) – 7 043 registros × 21 variables.
- Información demográfica, tipo de contrato, servicios contratados y facturación.
- Variable objetivo: `Churn` (Sí/No).

## 🔍 Proceso de trabajo

1. Exploración inicial & EDA
   - Distribuciones, correlaciones, desequilibrio de clases. 📊
2. Limpieza & _Feature Engineering_ 🧹
   - Conversión de binarias y "cuasi‑binarias" a 0/1.
   - Agrupación de categorías "No internet/phone service" → "No".
   - Conversión de `gender` a numérica.
   - Eliminación de `customerID`.
   - Normalización de variables numéricas.
   - Manejo de nulos (11 en `TotalCharges` → 0 donde `tenure = 0`).
3. Modelado 🤖
   - Regresión Logística (baseline).
   - Random Forest.
   - **XGBoost** (con y sin ajuste de *threshold*).
4. Evaluación
   - *Accuracy*, *Precision*, *Recall*, AUC‑ROC.
   - Ajuste de *threshold* (0.46) para mejorar *Recall*.
5. Despliegue
   - API & demo (Streamlit).

## 📊 Resultados

| Fecha      | Modelo              | `params` principales                      | Precision | Recall    | Accuracy  | AUC       |
| ---------- | ------------------- | ----------------------------------------- | --------- | --------- | --------- | --------- |
| 2025‑07‑06 | *Baseline* (LogReg) | C=10, solver=liblinear                    | **0.477** | 0.784     | 0.738     | 0.843     |
| 2025‑07‑06 | Random Forest       | max\_depth=10, n\_estimators=600          | **0.545** | 0.729     | **0.786** | **0.848** |
| 2025‑07‑06 | XGBoost             | lr=0.015, max\_depth=3, n\_estimators=400 | 0.480     | 0.822     | 0.740     | 0.846     |
| 2025‑07‑06 | XGBoost (thr 0.46)  | mismo que arriba                          | 0.471     | **0.851** | 0.731     | 0.846     |

> 📌 **Trade‑off**: alcanzar *Recall* ≈ 0.85 implica *Precision* ≈ 0.45 (≈ 55 % falsas alarmas). Debe discutirse con negocio el coste/beneficio de gestionar esas alertas.

## 📂 Estructura del repositorio

```text
churn_prediction/
├── app_streamlit/                 # Demo interactiva 🖥️
├── data/
│   ├── raw/                       # Dataset original
│   ├── processed/                 # CSVs limpios & feature‑engineered
│   └── train/, test/              # Splits
├── models/                        # Modelos .pkl y artefactos
├── notebooks/                    
│   ├── 01_Fuentes.ipynb           # Ingesta de datos
│   ├── 02_LimpiezaEDA.ipynb       # Limpieza & EDA
│   └── 03_Entrenamiento_Evaluacion.ipynb
├── src/
│   ├── churn_model.py             # Clase predictora
│   ├── data_processing.py         # Pipeline de limpieza
│   ├── training.py                # Entrenamiento
│   └── evaluation.py              # Métricas
└── README.md                      # (este documento)
```

## 🌍 App Demo

- [Streamlit App](https://churn-prediction-cprc.onrender.com/)  

## 🗒️ Pendientes / Próximos pasos

- Mejora en el balanceo del _Target_.
- Profundizar en _Feature Engineering_ para disminuir el número de _Features_.
- Desarrollar nuevos modelos.

---

### ✍️ Autor

**Xabi del Rey** — *Product Manager & Data Science Bootcamper*\
Contacto → [LinkedIn](https://www.linkedin.com/in/xabidelrey/) 🤝
