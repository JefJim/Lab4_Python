# Importación de bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Configuración de visualización
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)

## 1. Lectura del conjunto de datos desde GitHub
url = "https://raw.githubusercontent.com/[tu_usuario]/[tu_repositorio]/main/datos_policiales.csv"
try:
    df = pd.read_csv(url)
    print("Datos cargados exitosamente")
except Exception as e:
    print(f"Error al cargar los datos: {e}")

## 2. Visualización de características básicas del conjunto de datos
print("\n=== Información básica del dataset ===")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nResumen estadístico:")
print(df.describe(include='all'))

## 3. Cambio de nombres de columnas a español
# (Asumiendo que ya están en español, verificamos)
print("\nNombres de columnas actuales:")
print(df.columns.tolist())

## 4. Detección de valores nulos con evidencia gráfica
print("\nValores nulos por columna:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Mapa de calor de valores nulos en el dataset")
plt.savefig("valores_nulos.png")
plt.show()

## 5. Detección de valores atípicos con evidencia gráfica
# Análisis para columnas numéricas (Edad)
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Edad'])
plt.title("Diagrama de caja para la variable Edad")
plt.savefig("outliers_edad.png")
plt.show()

## 6. Función para imputación de valores
def imputar_valores(df, estrategia='mean'):
    """
    Función para imputar valores faltantes según la estrategia especificada.
    
    Parámetros:
    - df: DataFrame de pandas
    - estrategia: 'mean' para media, 'median' para mediana, 'most_frequent' para moda
    
    Retorna:
    - DataFrame con valores imputados
    """
    imputer = SimpleImputer(strategy=estrategia)
    
    # Imputación para variables numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Imputación para variables categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    return df

# Aplicamos la función de imputación
df_imputado = imputar_valores(df.copy())
print("\nValores nulos después de imputación:")
print(df_imputado.isnull().sum())

## 7. Conversión de tipos de datos
# Convertir Fecha a datetime
df_imputado['Fecha'] = pd.to_datetime(df_imputado['Fecha'], errors='coerce')

# Verificar tipos de datos
print("\nTipos de datos después de conversión:")
print(df_imputado.dtypes)

## 8. Conversión de variables categóricas a numéricas
# Usamos Label Encoding para variables categóricas ordinales
cat_cols = ['Delito', 'SubDelito', 'Victima', 'SubVictima', 'Genero', 'Nacionalidad', 'Provincia', 'Canton', 'Distrito']

le = LabelEncoder()
for col in cat_cols:
    if col in df_imputado.columns:
        df_imputado[col+'_encoded'] = le.fit_transform(df_imputado[col].astype(str))

## 9. Estandarización de datos numéricos
numeric_cols = ['Edad']  # Añadir otras columnas numéricas si existen
scaler = StandardScaler()
df_imputado[numeric_cols] = scaler.fit_transform(df_imputado[numeric_cols])

## 10. Análisis de correlación
# Calculamos matriz de correlación para variables numéricas
numeric_df = df_imputado.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de correlación de variables numéricas")
plt.savefig("correlacion.png")
plt.show()

## 11. Identificación de variable dependiente
# Basado en el análisis exploratorio, podríamos considerar:
# - 'Delito' como variable objetivo para clasificación
# - 'Edad' como variable objetivo para regresión (si hay suficiente contexto)

print("\nPosibles variables dependientes:")
print("- Para clasificación: 'Delito' (predecir tipo de delito)")
print("- Para regresión: 'Edad' (predecir edad de víctima, si hay patrones)")

## 12. Selección de modelo
# Dependiendo del problema:
# - Clasificación: Random Forest o Regresión Logística
# - Regresión: Random Forest Regressor o Regresión Lineal

print("\nModelos candidatos:")
print("1. Clasificación (predecir tipo de delito): Random Forest")
print("   - Razón: Maneja bien variables categóricas y relaciones no lineales")
print("2. Regresión (predecir edad de víctima): Random Forest Regressor")
print("   - Razón: Robustez ante outliers y relaciones complejas")

## 13. Guardado del dataset procesado
df_imputado.to_csv('data_process.csv', index=False)
print("\nDataset procesado guardado como 'data_process.csv'")

## 14. Documentación de integrantes del grupo
"""
Integrantes del grupo:
1. Jefry Jiménez Rocha - 208320789
2. Diego - Carné
3. Nombre Apellido - Carné

Fecha de entrega: DD/MM/AAAA
"""