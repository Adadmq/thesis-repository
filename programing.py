# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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