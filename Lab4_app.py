#15. Laboratorio #4 - Minería de Datos
# Universidad Técnica Nacional
# Integrantes del grupo: [Jefry Jiménez Rocha, Diego Francisco Umaña Salas, Marleny Molina Sobalvarro]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

#1. Usar  python
#2 Utilice el conjunto de datos, de su elección
# 3. Cargar conjunto de datos desde GitHub
url = "https://raw.githubusercontent.com/JefJim/Lab4_Python/main/Costa%20Rica%20Total%20deceases%202014%20-%202021.csv"
try:
    df = pd.read_csv(url, encoding='utf-8-sig')
    print("Datos cargados exitosamente desde GitHub")
except Exception as e:
    print(f"Error al cargar datos: {e}")
    # Cargar datos locales en caso de error
    df = pd.read_csv("Costa Rica Total deceases 2014 - 2021.csv", encoding='utf-8-sig')
    
# Limpieza inicial: eliminar filas totalmente vacías si las hay
df = df.dropna(how='all')

# 4. Visualizar características básicas del conjunto de datos
print("\n=== Características básicas del dataset ===")
print(f"Dimensiones del dataset: {df.shape}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nResumen estadístico:")
print(df.describe(include='all'))  # Incluye también variables categóricas
print("\nInformación del dataset:")
print(df.info())

# 5. Cambiar nombres de columnas a español (ya están en español, pero podemos estandarizar)
nombres_espanol = {
    'anotrab': 'anio',
    'mestrab': 'mes',
    'nacionalid': 'nacionalidad',
    'Sexo': 'sexo',
    'estcivil': 'estado_civil',
    'edads': 'edad',
    'edadsrec': 'grupo_edad',
    'provincia': 'provincia',
    'pc': 'distrito_residencia',
    'IU': 'indice_urbanizacion',
    'causamuer': 'codigo_causa_muerte',
    'des_causa': 'descripcion_causa_muerte',
    'autopsia': 'autopsia',
    'asistmed': 'asistencia_medica',
    'instmurio': 'lugar_muerte',
    'provocu': 'provincia_muerte',
    'pcocu': 'distrito_muerte',
    'diadef': 'dia_defuncion',
    'mesdef': 'mes_defuncion',
    'anodef': 'anio_defuncion',
    'ocuparec': 'ultima_ocupacion',
    'nacmadre': 'nacionalidad_madre',
    'provregis': 'provincia_registro',
    'pcregis': 'distrito_registro',
    'diadeclara': 'dia_declaracion',
    'mesdeclara': 'mes_declaracion',
    'anodeclara': 'anio_declaracion',
    'grgruposcb': 'grupo_to17',
    'gruposcb': 'grupo_to63',
}
df = df.rename(columns=nombres_espanol)
print("\nNombres de columnas estandarizados:")
print(df.columns)
# 6. Determinar valores nulos con evidencia gráfica
print("\n=== Verificación de valores nulos ===")
print("\n=== Valores nulos por columna ===")
print(df.isnull().sum())
if df.isnull().sum().sum() == 0:
    print("✅ No se encontraron valores nulos en el dataset")
    # Crear un gráfico indicando que no hay nulos
    plt.figure(figsize=(6, 2))
    plt.text(0.5, 0.5, 'No se encontraron valores nulos en el dataset', 
             ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.title("Estado de valores nulos")
    plt.savefig("valores_nulos.png")
    plt.show()
else:
    print("⚠️ Se encontraron valores nulos:")
    print(df.isnull().sum())
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Mapa de calor de valores nulos")
    plt.savefig("valores_nulos.png")
    plt.show()

# 7. Identificar valores atípicos solo en columnas numéricas
print("\n🔢 Identificación de columnas numéricas:")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Columnas numéricas encontradas:", numeric_cols)
print("\n=== Valores atípicos ===")
if not numeric_cols:
    print("⚠️ No se encontraron columnas numéricas en el dataset")
    print("📌 Tipos de datos encontrados:")
    print(df.dtypes)
else:
    # 7.1. Análisis de valores atípicos
    print("\n📊 Análisis de valores atípicos:")
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(1, len(numeric_cols), i)
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot de {col}")
    plt.tight_layout()
    plt.savefig("valores_atipicos.png")
    plt.show()

#8. Funciones para imputación de variables (adaptadas para este dataset)
def imputar_nulos(df):
    """Imputa valores nulos para preparar el dataset para clasificación."""
    # Columnas numéricas
    num_cols = ['edad', 'anio', 'indice_urbanizacion', 'dia_defuncion', 'anio_defuncion', 'dia_declaracion', 'anio_declaracion', 'grupo_to17']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Columnas categóricas clave para clasificación
    cat_cols = ['sexo', 'estado_civil', 'provincia', 'descripcion_causa_muerte']
    for col in cat_cols:
        df[col] = df[col].fillna('Desconocido')
    
    return df

def manejar_atipicos(df):
    """Maneja valores atípicos usando el método IQR"""
    # Solo aplicamos a 'Total_defunciones' y 'Edad'
    for col in ['edad', 'anio', 'indice_urbanizacion', 'dia_defuncion', 'anio_defuncion', 'dia_declaracion', 'anio_declaracion', 'grupo_to17']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Winsorizar (reemplazar con los límites)
        df[col] = np.where(df[col] < lower_bound, lower_bound, 
                          np.where(df[col] > upper_bound, upper_bound, df[col]))
    
    return df

#Aplicar funciones de imputación
df = imputar_nulos(df)
df = manejar_atipicos(df)

#Análisis de la variable dependiente: descripcion_causa_muerte
print("\n=== Análisis de la variable dependiente (descripcion_causa_muerte) ===")
print("Distribución de causas de muerte:")
print(df['descripcion_causa_muerte'].value_counts().head(10))  # Top 10 causas

# Visualización de las causas más frecuentes
plt.figure(figsize=(12, 6))
df['descripcion_causa_muerte'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 causas de muerte en Costa Rica (2014-2021)')
plt.xlabel('Causa de muerte')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("top10_causas_muerte.png")
plt.show()

# 13. Identificar variable dependiente y modelo candidato
# En este caso, 'Total_defunciones' podría ser la variable dependiente
le = LabelEncoder()
df['causa_muerte_codificada'] = le.fit_transform(df['descripcion_causa_muerte'])

# Identificación del problema y modelo recomendado
variable_dependiente = 'descripcion_causa_muerte'
print(f"\nVariable dependiente seleccionada: {variable_dependiente}")
print("Tipo de problema: Clasificación multiclase")
modelo_recomendado = "Random Forest Classifier"
print(f"Modelo recomendado: {modelo_recomendado} (por su manejo de variables categóricas y robustez)")

# 14. Guardar el dataset procesado
df.to_csv('data_process.csv', index=False)
print("\nDataset procesado guardado como 'data_process.csv'")

# Opcional: Guardar también como archivo .ipynb
# Este código debería copiarse a un notebook de Jupyter y guardarse como lab4_IC2025.ipynb