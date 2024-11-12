# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Cargar los datos (reemplaza 'your-data-file.csv' por el nombre del archivo CSV correspondiente)
# Ejemplo para balance_sheet.csv, income_statement.csv y cash_flow.csv
balance_sheet = pd.read_csv('/kaggle/input/balance_sheet.csv')
income_statement = pd.read_csv('/kaggle/input/income_statement.csv')
cash_flow = pd.read_csv('/kaggle/input/cash_flow.csv')

# Unificar y renombrar columnas según necesites para el análisis principal (si tus datos originales se unían en 'data')
# Ejemplo de unificación
data = pd.concat([balance_sheet[['Category', 'Description', 'Current Period (30-Apr-2022)']],
                  income_statement[['Category', 'Description', 'Current Period (30-Apr-2022)']],
                  cash_flow[['Category', 'Description', 'Current Period (30-Apr-2022)']]
                 ], ignore_index=True)

# Asigna nombres de columnas genéricos para que coincidan con 'Ingresos', 'Gastos' e 'Inversiones'
# Este es un ejemplo: ajusta según la estructura de tu CSV
data.columns = ['Fecha', 'Ingresos', 'Gastos', 'Inversiones']  # Si no tienes una fecha exacta, puedes añadir una manual

# Continuar con el resto del código
data['Fecha'] = pd.to_datetime('2022-04-30')  # Fecha fija o ajustada según el rango de tus datos
data = data.set_index('Fecha')
data.fillna(0, inplace=True)  # Reemplazar valores nulos con 0

# A partir de aquí, el resto del código sigue igual para cálculos de ratios, evaluación de viabilidad, pronósticos, etc.

# Supongamos que 'Ingresos' representa los ingresos totales y 'Gastos' los gastos totales

# Ratio de Rentabilidad Bruta
data['Ratio_Rentabilidad_Bruta'] = data['Ingresos'] / (data['Ingresos'] + data['Gastos'])

# Ratio de Liquidez (si tienes una columna de activos y pasivos)
data['Ratio_Liquidez'] = data['Activos'] / data['Pasivos']  # Ejemplo

# Otros ratios financieros, como ROI (Return on Investment) o ROE (Return on Equity), también pueden calcularse

# Evaluar si los ingresos son consistentemente mayores que los gastos
data['Viable'] = data['Ingresos'] > data['Gastos']

# Agregar una métrica de viabilidad en función de un margen mínimo de ingresos sobre gastos
data['Margen_de_Seguridad'] = (data['Ingresos'] - data['Gastos']) / data['Ingresos']

# Considerar viable si el margen de seguridad supera un umbral, por ejemplo, el 20%
data['Es_Viable'] = data['Margen_de_Seguridad'] > 0.20
 
 # Crear datos de entrenamiento y prueba para el modelo
# Suponiendo que queremos predecir 'Ingresos' usando el índice temporal
data['Fecha_num'] = data.index.map(pd.Timestamp.toordinal)  # Convertir la fecha a número para regresión
X = data[['Fecha_num']]
y = data['Ingresos']

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir ingresos futuros
data['Prediccion_Ingresos'] = model.predict(X)  # Predicción en todo el rango de fechasplt.figure(figsize=(10, 6))
plt.plot(data.index, data['Ingresos'], label="Ingresos Reales")
plt.plot(data.index, data['Prediccion_Ingresos'], linestyle='--', label="Predicción Ingresos")
plt.xlabel("Fecha")
plt.ylabel("Ingresos")
plt.legend()
plt.show()