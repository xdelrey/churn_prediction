# importamos las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Lectura de datos y muestra inicial
df_raw = pd.read_csv('..\\data\\raw\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df_raw

# Convertimos los espacios " " de TotalCharges a 0's y transformamos a float64
df['TotalCharges'] = (
    df['TotalCharges']
      .replace(r'^\s*$', '0', regex=True)   # "^\\s*$" = solo espacios
      .astype(float)                        # conversión final a float64
)

# Eliminamos 'customerID'
df = df.drop(columns='customerID')

# Transformación de variables binarias y 'cuasibinarias'
# Lista de las columnas con Yes / No (y otras opciones equiparables a No)
cols_yes_no = [
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'PaperlessBilling',
    'Churn'
     ]

# Iteración sobre cada columna y aplicamos el mapeo a 1's y 0's.
for col in cols_yes_no:
    df[col] = df[col].map({
        'Yes': 1,
        'No': 0,
        'No phone service': 0, # este valor lo considero un 'No'
        'No internet service':0 # este valor lo considero un 'No'
        })

# Convertimos también 'gender' a numérica
df['gender'] = (df['gender'] == 'Female').astype(int)

# Cambiamos el nombre de la columna para reflejar lo que indican los valores
df.rename(columns={'gender':'is_female'}, inplace=True)

# Variables Categóricas ################################################
porc_InternetService = ((df['InternetService'].value_counts() / len(df )) * 100).round(2)
porc_Contract = ((df['Contract'].value_counts() / len(df )) * 100).round(2)
porc_PaymentMethod = ((df['PaymentMethod'].value_counts() / len(df )) * 100).round(2)

# Variables categóricas con múltiples valores
multi_cat = ['InternetService', 'Contract', 'PaymentMethod']

# OneHotEncoding
df = pd.get_dummies(df, columns=multi_cat, drop_first=True)

# Guardado de datos limpios
df.to_csv('..\\data\\processed\\churn_clean_v2_py.csv', index=False, encoding='utf-8')
df_clean = df

# Variables Numéricas ################################################
# Describe de las columnas numéricas
numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']

