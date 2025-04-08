# Laboratorio #4 - Minería de Datos
# Universidad Técnica Nacional
# Integrantes del grupo: [Nombre1, Nombre2, Nombre3]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 2. Cargar conjunto de datos desde GitHub
url = "https://github.com/JefJim/Lab4_Python/blob/main/demographic-and-socioeconomic-statistics-indicators-for-costa-rica-13.csv"
try:
    df = pd.read_csv(url)
    print("Datos cargados exitosamente desde GitHub")
except Exception as e:
    print(f"Error al cargar datos: {e}")
    # Cargar datos locales en caso de error
    df = pd.read_csv("datos_backup.csv")

# 4. Visualizar características básicas del conjunto de datos
print("\n=== Características básicas del dataset ===")
print(f"Dimensiones del dataset: {df.shape}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nResumen estadístico:")
print(df.describe())
print("\nInformación del dataset:")
print(df.info())

# 5. Cambiar nombres de columnas a español
# Asumiendo que el dataset original está en inglés
nombres_espanol = {
    'column1': 'columna1',
    'column2': 'columna2',
    # ... agregar todas las columnas
}
df = df.rename(columns=nombres_espanol)
print("\nNombres de columnas en español:")
print(df.columns)

# 6. Determinar valores nulos con evidencia gráfica
print("\n=== Valores nulos ===")
print("Cantidad de valores nulos por columna:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Mapa de calor de valores nulos")
plt.savefig("valores_nulos.png")
plt.show()

# 7. Identificar valores atípicos con evidencia gráfica
print("\n=== Valores atípicos ===")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot de {col}")
plt.tight_layout()
plt.savefig("valores_atipicos.png")
plt.show()

# 8. Funciones para imputación de variables
def imputar_nulos(df, estrategia='media'):
    """Imputa valores nulos según la estrategia especificada"""
    imputer = SimpleImputer(strategy=estrategia)
    df_imputado = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['int64', 'float64'])), 
                              columns=df.select_dtypes(include=['int64', 'float64']).columns)
    
    for col in df.select_dtypes(exclude=['int64', 'float64']).columns:
        df_imputado[col] = df[col].fillna(df[col].mode()[0])
    
    return df_imputado

def manejar_atipicos(df, metodo='IQR'):
    """Maneja valores atípicos usando el método IQR o Z-score"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if metodo == 'IQR':
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Opción 1: Eliminar atípicos
            # df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Opción 2: Winsorizar (reemplazar con los límites)
            df[col] = np.where(df[col] < lower_bound, lower_bound, 
                               np.where(df[col] > upper_bound, upper_bound, df[col]))
    
    elif metodo == 'Zscore':
        for col in numeric_cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            df[col] = np.where(np.abs(z_scores) > 3, 
                              np.sign(z_scores) * 3 * df[col].std() + df[col].mean(), 
                              df[col])
    
    return df

# Aplicar funciones de imputación
df = imputar_nulos(df, estrategia='media')
df = manejar_atipicos(df, metodo='IQR')

# 9. Conversión de tipos de datos
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"Columna {col} convertida a datetime")
        except:
            pass

# 10. Conversión de variables categóricas a numéricas
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_encoder.fit_transform(df[col])
    print(f"Columna categórica {col} convertida a numérica")

# 11. Estandarización o normalización
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 12. Correlación de variables con evidencia gráfica
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de correlación")
plt.savefig("correlacion.png")
plt.show()

# 13. Identificar variable dependiente y modelo candidato
# Asumimos que la última columna es la variable dependiente
variable_dependiente = df.columns[-1]
print(f"\nVariable dependiente identificada: {variable_dependiente}")

# Determinar si es problema de clasificación o regresión
if df[variable_dependiente].nunique() < 10:  # Asumimos que es clasificación si hay pocos valores únicos
    print("Problema de clasificación detectado")
    modelo_recomendado = "Random Forest"
else:
    print("Problema de regresión detectado")
    modelo_recomendado = "Regresión Lineal"

print(f"Modelo recomendado: {modelo_recomendado}")

# 14. Guardar conjunto de datos procesado
df.to_csv('data_process.csv', index=False)
print("\nDataset procesado guardado como 'data_process.csv'")