
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

# 1. Lectura del conjunto de datos desde GitHub
url_excel = "https://github.com/JefJim/Lab4_Python/raw/main/Estad%C3%ADsticas%20Policiales%202020.xlsx"
df = pd.read_excel(url_excel)

# 2. Visualización de características básicas del conjunto de datos
print("\n=== Información básica del dataset ===")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nResumen estadístico:")
print(df.describe(include='all'))

# 3. Verificación de nombres de columnas
print("\nNombres de columnas actuales:")
print(df.columns.tolist())

# 4. Detección de valores nulos con evidencia gráfica
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Porcentaje de nulos por columna
nulos = df.isnull().mean() * 100
nulos = nulos[nulos > 0]  # Solo muestra columnas con nulos

if df.isnull().sum().sum() == 0:
    print("✅ No hay valores nulos en el dataset.")
else:
    # Solo grafica si hay nulos
    nulos = df.isnull().mean() * 100
    nulos = nulos[nulos > 0]
    
    if not nulos.empty:
        plt.figure(figsize=(10, 4))
        nulos.plot(kind='bar', color='red', edgecolor='black')
        plt.title("Porcentaje de valores nulos por columna")
        plt.ylabel("Porcentaje de nulos (%)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.show()
    else:
        print("✅ No hay columnas con valores nulos.")

# 5. Limpieza y transformación de la columna Edad (categorías -> números)
df['Edad'] = df['Edad'].astype(str).str.strip().str.lower()

df['Edad'] = df['Edad'].astype(str).str.strip().str.lower().replace({
    'menor de edad': 'Menor de edad',
    'mayor de edad': 'Adulto',
    'adulto mayor': 'Adulto mayor',
    'desconocido': 'Desconocido'
})
plt.figure(figsize=(10, 8))
colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#A5A5A5']  # Colores modernos
explode = (0.05, 0.05, 0.05, 0.05)  # Separar ligeramente las porciones


# 6. Detección de valores atípicos con evidencia gráfica
patches, texts, autotexts = plt.pie(
    df['Edad'].value_counts(),
    labels=['Adulto', 'Menor de edad', 'Adulto mayor', 'Desconocido'],  # Etiquetas claras
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=explode,
    textprops={'fontsize': 12},
    pctdistance=0.8
)
plt.setp(autotexts, color='white', weight='bold')  # Porcentajes en blanco y negrita
plt.setp(texts, fontsize=12, weight='bold')  # Etiquetas en negrita
plt.title('Distribución por Grupo de Edad', pad=20, fontsize=14, weight='bold')
plt.legend(
    title="Categorías:",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    labels=['Adulto (18-64 años)', 'Menor de edad (<18)', 'Adulto mayor (65+)', 'Desconocido']
)

# Ajustar layout y guardar
plt.tight_layout()
plt.savefig('distribucion_edad_mejorado.png', dpi=300, bbox_inches='tight')
plt.show()
# 7. Función para imputación de valores
def imputar_valores(df, estrategia='mean'):
    imputer = SimpleImputer(strategy=estrategia)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    cat_cols = df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    return df

# 8. Imputación de valores faltantes
df_imputado = imputar_valores(df.copy())
print("\nValores nulos después de imputación:")
print(df_imputado.isnull().sum())

# 9. Conversión de tipos de datos
df_imputado['Fecha'] = pd.to_datetime(df_imputado['Fecha'], errors='coerce')

# 10. Conversión de variables categóricas a numéricas
cat_cols = ['Delito', 'SubDelito', 'Victima', 'SubVictima', 'Genero', 'Nacionalidad', 'Provincia', 'Canton', 'Distrito']
le = LabelEncoder()
for col in cat_cols:
    if col in df_imputado.columns:
        df_imputado[col+'_encoded'] = le.fit_transform(df_imputado[col].astype(str))

# 11. Estandarización de datos numéricos
numeric_cols = ['Edad']
scaler = StandardScaler()
df_imputado[numeric_cols] = scaler.fit_transform(df_imputado[numeric_cols])

# 12. Análisis de correlación
numeric_df = df_imputado.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de correlación de variables numéricas")
plt.savefig("correlacion.png")
plt.show()

# 13. Identificación de variable dependiente
print("\nPosibles variables dependientes:")
print("- Para clasificación: 'Delito' (predecir tipo de delito)")
print("- Para regresión: 'Edad' (predecir edad de víctima, si hay patrones)")

# 14. Selección de modelo
print("\nModelos candidatos:")
print("1. Clasificación (predecir tipo de delito): Random Forest")
print("2. Regresión (predecir edad de víctima): Random Forest Regressor")

# 15. Guardado del dataset procesado
df_imputado.to_csv('data_process.csv', index=False)
print("\nDataset procesado guardado como 'data_process.csv'")

# 16. Documentación de integrantes del grupo
print("""
Integrantes del grupo:
1. Jefry Jiménez Rocha - 208320789
2. Diego - Carné
3. Nombre Apellido - Carné

Fecha de entrega: DD/MM/AAAA
""")
