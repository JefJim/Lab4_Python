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
    df = pd.read_csv(url, encoding='latin-1')  # Usamos latin-1 por posibles caracteres especiales
    print("Datos cargados exitosamente desde GitHub")
except Exception as e:
    print(f"Error al cargar datos: {e}")
    # Cargar datos locales en caso de error
    df = pd.read_csv("Costa Rica Total deceases 2014 - 2021.csv", encoding='latin-1')

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
    'des_causa': 'descripción_causa_muerte',
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
df['Total_defunciones'] = 1  # Cada registro representa 1 defunción
total_defunciones = len(df)
print(f"\n📌 Total de defunciones registradas: {total_defunciones:,}")
# 6. Determinar valores nulos con evidencia gráfica
print("\n=== Verificación de valores nulos ===")
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

# 8. Funciones para imputación de variables (adaptadas para este dataset)
def imputar_nulos(df):
    """Imputa valores nulos según el tipo de columna"""
    # Para columnas numéricas
    df['Total_defunciones'] = df['Total_defunciones'].fillna(df['Total_defunciones'].median())
    df['edad'] = df['edad'].fillna(df['edad'].median())
    
    # Para columnas categóricas
    cat_cols = ['sexo', 'estado_civil', 'provincia', 'distrito_residencia', 'nacionalidad', 'descripción_causa_muerte']
    for col in cat_cols:
        df[col] = df[col].fillna('Desconocido')
    
    return df

def manejar_atipicos(df):
    """Maneja valores atípicos usando el método IQR"""
    # Solo aplicamos a 'Total_defunciones' y 'Edad'
    for col in ['Total_defunciones', 'edad']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Winsorizar (reemplazar con los límites)
        df[col] = np.where(df[col] < lower_bound, lower_bound, 
                          np.where(df[col] > upper_bound, upper_bound, df[col]))
    
    return df

# Aplicar funciones de imputación
df = imputar_nulos(df)
df = manejar_atipicos(df)

# 9. Conversión de tipos de datos
df['anio'] = df['anio'].astype('int')
df['mes'] = df['mes'].astype('category')  # Mes es categórico ordinal

# 10. Conversión de variables categóricas a numéricas (solo las necesarias)
# Para este dataset, podríamos no convertir todas ya que muchas son descriptivas
label_encoder = LabelEncoder()
cols_to_encode = ['sexo', 'estado_civil', 'provincia', 'distrito_residencia', 'nacionalidad']
for col in cols_to_encode:
    df[col+'_encoded'] = label_encoder.fit_transform(df[col])

# 11. Estandarización solo de las columnas numéricas continuas
scaler = StandardScaler()
df[['Total_defunciones', 'edad']] = scaler.fit_transform(df[['Total_defunciones', 'edad']])

# 12. Correlación de variables (solo numéricas)
numeric_cols_for_corr = ['Total_defunciones', 'edad', 'anio'] + [col for col in df.columns if '_encoded' in col]
plt.figure(figsize=(15, 10))
corr_matrix = df[numeric_cols_for_corr].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Matriz de correlación")
plt.savefig("correlacion.png")
plt.show()

# 13. Identificar variable dependiente y modelo candidato
# En este caso, 'Total_defunciones' podría ser la variable dependiente
variable_dependiente = 'Total_defunciones'
print(f"\nVariable dependiente identificada: {variable_dependiente}")

# Dado que 'Total_defunciones' es numérica continua, sería un problema de regresión
print("Problema de regresión detectado (predicción de cantidad de defunciones)")
modelo_recomendado = "Random Forest Regressor"
print(f"Modelo recomendado: {modelo_recomendado} (por su capacidad para manejar múltiples predictores)")

# 14. Guardar conjunto de datos procesado
def corregir_caracteres(texto):
    if isinstance(texto, str):
        return texto.replace('Ã', 'í')
    return texto

# Aplicar la corrección a todas las columnas de tipo objeto (strings)
for columna in df.select_dtypes(include=['object']).columns:
    df[columna] = df[columna].apply(corregir_caracteres)

# Verificación de resultados
print("\n🔍 Verificación de corrección de caracteres:")
# Mostrar algunas filas que contenían el problema (si existen)
filas_con_problema = df.apply(lambda row: row.astype(str).str.contains('Ã').any(), axis=1)
if filas_con_problema.any():
    print("Se encontraron y corrigieron caracteres 'Ã' en las siguientes filas:")
    print(df[filas_con_problema].head())
else:
    print("✅ No se encontraron más caracteres 'Ã' en el dataset")

# Guardar el dataset corregido
df.to_csv('datos_corregidos.csv', index=False, encoding='utf-8-sig')
print("\n💾 Dataset con caracteres corregidos guardado como 'datos_corregidos.csv'")
df.to_csv('data_process.csv', index=False)
print("\nDataset procesado guardado como 'data_process.csv'")

# Opcional: Guardar también como archivo .ipynb
# Este código debería copiarse a un notebook de Jupyter y guardarse como lab4_IC2025.ipynb