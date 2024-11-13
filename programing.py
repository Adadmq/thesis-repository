import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Código actual para cargar y procesar datos
balance_sheet = pd.read_csv('balance_sheet.csv')
income_statement = pd.read_csv('income_statement.csv')
cash_flow = pd.read_csv('cash_flow.csv')

data = pd.concat([balance_sheet[['Category', 'Description', 'Current Period (30-Apr-2022)']],
                  income_statement[['Category', 'Description', 'Current Period (30-Apr-2022)']],
                  cash_flow[['Category', 'Description', 'Current Period (30-Apr-2022)']]
                 ], ignore_index=True)

data.columns = ['Fecha', 'Ingresos', 'Gastos', 'Inversiones']
data['Fecha'] = pd.to_datetime('2022-04-30')  # Fecha fija o ajustada según el rango de tus datos
data = data.set_index('Fecha')
data.fillna(0, inplace=True)

# Ratios financieros
data['Ratio_Rentabilidad_Bruta'] = data['Ingresos'] / (data['Ingresos'] + data['Gastos'])
data['Viable'] = data['Ingresos'] > data['Gastos']
data['Margen_de_Seguridad'] = (data['Ingresos'] - data['Gastos']) / data['Ingresos']
data['Es_Viable'] = data['Margen_de_Seguridad'] > 0.20

# Predicción de ingresos
data['Fecha_num'] = data.index.map(pd.Timestamp.toordinal)
X = data[['Fecha_num']]
y = data['Ingresos']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
data['Prediccion_Ingresos'] = model.predict(X)

# Integración del API de Gemini
API_KEY = 'TU_CLAVE_DE_API_GEMINI'  # Reemplaza con tu clave de API
url = "https://api.gemini.ai/v1/analisis"  # URL del API de Gemini (reemplaza por la URL correcta)
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Datos que deseas enviar al API de Gemini para análisis
payload = {
    "Ingresos": data['Ingresos'].tolist(),
    "Gastos": data['Gastos'].tolist(),
    "Prediccion_Ingresos": data['Prediccion_Ingresos'].tolist(),
    "Ratio_Rentabilidad_Bruta": data['Ratio_Rentabilidad_Bruta'].tolist()
}

# Realizar la solicitud al API
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    gemini_results = response.json()
    print("Resultados del análisis de Gemini:", gemini_results)
else:
    print("Error en la solicitud a Gemini:", response.status_code, response.text)

# Gráfico de predicciones
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Ingresos'], label="Ingresos Reales")
plt.plot(data.index, data['Prediccion_Ingresos'], linestyle='--', label="Predicción Ingresos")
plt.xlabel("Fecha")
plt.ylabel("Ingresos")
plt.legend()
plt.show()
