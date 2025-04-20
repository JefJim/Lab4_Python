import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Crear una instancia de LabelEncoder
le = LabelEncoder()
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

# Codificar las variables categóricas
df['sexo_encoded'] = le.fit_transform(df['sexo'])
df['estado_civil_encoded'] = le.fit_transform(df['estado_civil'])
df['provincia_encoded'] = le.fit_transform(df['provincia'])
df['provincia_registro_encoded'] = le.fit_transform(df['provincia_registro'])
df['grupo_edad_encoded'] = le.fit_transform(df['grupo_edad'])
df['nacionalidad_encoded'] = le.fit_transform(df['nacionalidad'])
df['distrito_residencia_encoded'] = le.fit_transform(df['distrito_residencia'])
df['causa_muerte_encoded'] = le.fit_transform(df['descripcion_causa_muerte'])
df['provincia_muerte_encoded'] = LabelEncoder().fit_transform(df['provincia_muerte'].astype(str))

# 8. Funciones para imputación de variables (adaptadas para este dataset)
# def imputar_nulos(df):
#     """Imputa valores nulos según el tipo de columna"""
#     # Para columnas numéricas
#     df['Total_defunciones'] = df['Total_defunciones'].fillna(df['Total_defunciones'].median())
#     df['edad'] = df['edad'].fillna(df['edad'].median())
    
#     # Para columnas categóricas
#     cat_cols = ['sexo', 'estado_civil', 'provincia', 'distrito_residencia', 'nacionalidad', 'descripción_causa_muerte']
#     for col in cat_cols:
#         df[col] = df[col].fillna('Desconocido')
    
#     return df

# def manejar_atipicos(df):
#     """Maneja valores atípicos usando el método IQR"""
#     # Solo aplicamos a 'Total_defunciones' y 'Edad'
#     for col in ['Total_defunciones', 'edad']:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
        
#         # Winsorizar (reemplazar con los límites)
#         df[col] = np.where(df[col] < lower_bound, lower_bound, 
#                           np.where(df[col] > upper_bound, upper_bound, df[col]))
    
#     return df

# Aplicar funciones de imputación
#df = imputar_nulos(df)
#df = manejar_atipicos(df)


# 9. Conversión de tipos de datos
# 10. Conversión de variables categóricas a numéricas (solo las necesarias)
# 11. Estandarización solo de las columnas numéricas continuas

# Crear un diccionario de diccionarios
frecuencia_por_columna = {}

# Recorremos cada columna del DataFrame
for columna in df.columns:
    conteo = df[columna].value_counts().to_dict()
    frecuencia_por_columna[columna] = conteo

for col in list(frecuencia_por_columna.keys())[:1]:  # Solo muestra las primeras 5 columnas para visualizar
    print(f"\n📊 Frecuencias en la columna: {col}")
    print(frecuencia_por_columna[col])


#Predecir la causa de muerte
# Filtrar causas más comunes
top_causas = df['descripcion_causa_muerte'].value_counts().nlargest(10).index
df = df[df['descripcion_causa_muerte'].isin(top_causas)]

# Asegurar que edad sea numérica
df['edad'] = df['edad'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)

# Variables independientes y dependiente
X = df[['edad', 'sexo_encoded', 'estado_civil_encoded', 'provincia_encoded',
        'provincia_registro_encoded', 'grupo_edad_encoded', 'nacionalidad_encoded',
        'provincia_muerte_encoded']]

y = df['descripcion_causa_muerte']

# Rellenar valores nulos si existen
X = X.fillna(-1)

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Random Forest optimizado
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)
# Evaluación
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
# Mostrar las 5 causas con mejor precisión
print("📊 Top 5 causas de muerte con mayor precisión:")
print(df_report.sort_values('precision', ascending=False).head(5)[['precision', 'recall', 'f1-score']])
#matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues")
plt.title("Matriz de Confusión - Causa de Muerte")
plt.tight_layout()
plt.show()
#Predecir la causa de muerte

#Predecir la causa de muerte2
X_test_copy = X_test.copy()
X_test_copy['real'] = y_test
X_test_copy['pred'] = model.predict(X_test)
# Decodificar los nombres de provincia
label_decoder = LabelEncoder()
label_decoder.fit(df['provincia_muerte'].astype(str))
X_test_copy['provincia_nombre'] = label_decoder.inverse_transform(X_test_copy['provincia_muerte_encoded'])
# Crear el resumen por provincia
resumen = []
for provincia in X_test_copy['provincia_nombre'].unique():
    subset = X_test_copy[X_test_copy['provincia_nombre'] == provincia]
    if subset.empty:
        continue
    report = classification_report(subset['real'], subset['pred'], output_dict=True)
    
    # Encontrar la clase con mejor F1-score (excluyendo promedios)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    top_causa = report_df.sort_values('f1-score', ascending=False).iloc[0]
    causa_nombre = report_df.sort_values('f1-score', ascending=False).index[0]

    resumen.append({
        'Provincia': provincia,
        'Causa de Muerte Más Probable a Ocurrir en 2022': causa_nombre,
        'Precision': round(top_causa['precision'], 3),
        'Recall': round(top_causa['recall'], 3),
        'F1-Score': round(top_causa['f1-score'], 3)
    })

# Convertir a DataFrame
resumen_df = pd.DataFrame(resumen)
# Mostrar el resumen ordenado por provincia
print("📊 Causas de muerte con mayor probabilidad de ocurrir en 2022 por provincia:\n")
print(resumen_df.sort_values('Provincia').to_string(index=False))


#predecir la causa de muerte más probable en 2022 por grupo etario
# Filtrar años del 2014 al 2021
df_filtrado = df[df['anio'].between(2014, 2021)].copy()

# Inicializar LabelEncoders
le_edad = LabelEncoder()
le_causa = LabelEncoder()

# Codificar variables
df_filtrado['grupo_edad_encoded'] = le_edad.fit_transform(df_filtrado['grupo_edad'])
df_filtrado['causa_encoded'] = le_causa.fit_transform(df_filtrado['descripcion_causa_muerte'])

X = df_filtrado[['grupo_edad_encoded']]
y = df_filtrado['causa_encoded']

# División de datos para evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Entrenar modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Calcular métricas de clasificación
y_pred = modelo.predict(X_test)
reporte = classification_report(y_test, y_pred, target_names=le_causa.classes_, output_dict=True)
df_metricas = pd.DataFrame(reporte).transpose()

# Preparar predicción por grupo etario para 2022
grupos_edad = le_edad.classes_
grupos_edad_encoded = le_edad.transform(grupos_edad)
predicciones = modelo.predict(grupos_edad_encoded.reshape(-1, 1))
causas_probables = le_causa.inverse_transform(predicciones)

# Añadir métricas por causa predecida
metricas_asociadas = df_metricas.loc[causas_probables][['precision', 'recall', 'f1-score']].reset_index(drop=True)

# Resultado final
df_resultado = pd.DataFrame({
    'Grupo_Edad': grupos_edad,
    'Causa de Muerte Más Probable': causas_probables
})
# Unir con métricas
df_resultado = pd.concat([df_resultado, metricas_asociadas], axis=1)
# Ordenar por grupo de edad
df_resultado = df_resultado.sort_values(by='Grupo_Edad').reset_index(drop=True)
# Mostrar resultados
print("📊 Causas de muerte más probables por grupo etario en 2022:\n")
print(df_resultado)
#predecir la causa de muerte más probable en 2022 por grupo etario

#predecir la causa de muerte basándonos en grupo etario, sexo, estado civil, provincia y atención médica
# Copia del dataframe entre 2014 y 2021
df_modelo = df[df['anio'].between(2014, 2021)].copy()

# Variables a usar para predicción
# 12. Correlación de variables 
variables = ['grupo_edad', 'sexo', 'provincia', 'asistencia_medica', 'lugar_muerte', 'provincia_muerte', 'distrito_residencia', 'nacionalidad', 'ultima_ocupacion', 'distrito_muerte']

X = df_modelo[variables].copy()
y = df_modelo['descripcion_causa_muerte']

# Codificar variables categóricas
for col in variables:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Codificar variable objetivo
y_encoded = LabelEncoder().fit_transform(y.astype(str))

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Mostrar matriz de correlación de predictores
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("🔍 Matriz de correlación entre variables predictoras")
plt.savefig("matriz_correlacion.png")
plt.tight_layout()
plt.show()

# 1. Asistencia médica vs lugar de muerte
plt.figure(figsize=(10, 6))
pd.crosstab(df_modelo['asistencia_medica'], df_modelo['lugar_muerte'], normalize='index') \
    .plot(kind='bar', stacked=True, cmap='viridis', title='Relación entre asistencia médica y lugar de muerte')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafico_asismedica_lugarmuerte.png")
plt.show()

# 2. Provincia de muerte distinta a la de residencia
df_modelo['vive_muere_distinto'] = df_modelo['provincia'] != df_modelo['provincia_muerte']
plt.figure(figsize=(6, 4))
df_modelo['vive_muere_distinto'].value_counts(normalize=True).plot(kind='bar', color='orange')
plt.title('¿Murió en provincia distinta a la de residencia?')
plt.xticks(ticks=[0, 1], labels=["Misma provincia", "Distinta provincia"], rotation=0)
plt.savefig("grafico_vive_muere_distinto.png")
plt.ylabel("Proporción")
plt.tight_layout()
plt.show()

# 3. Autopsia vs grupo de edad
plt.figure(figsize=(12, 6))
pd.crosstab(df_modelo['grupo_edad'], df_modelo['autopsia'], normalize='index') \
    .plot(kind='bar', stacked=True, colormap='coolwarm', title='Proporción de autopsias por grupo de edad')
plt.xticks(rotation=90)
plt.savefig("grafico_autopsias_grupopedad.png")
plt.tight_layout()
plt.show()

# 4. Última ocupación vs grupo_to63
plt.figure(figsize=(10, 6))
pd.crosstab(df_modelo['grupo_edad'], df_modelo['ultima_ocupacion'], normalize='index') \
    .iloc[:, :10].plot(kind='bar', stacked=True, cmap='tab20', title='Distribución ocupacional por grupo_edad (Top 10)')
plt.xticks(rotation=45)
plt.savefig("grafico_ultima_ocupacion.png")
plt.tight_layout()
plt.show()

# 5. Mapa de calor: causas vs grupo etario (solo las más frecuentes para legibilidad)
top_causas = df_modelo['descripcion_causa_muerte'].value_counts().nlargest(10).index
heatmap_data = pd.crosstab(df_modelo[df_modelo['descripcion_causa_muerte'].isin(top_causas)]['grupo_edad'],
                           df_modelo['descripcion_causa_muerte'])
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='magma', annot=True, fmt='d')
plt.title("Mapa de calor: causas de muerte más comunes por grupo etario")
plt.xticks(rotation=45)
plt.savefig("grafico_causa_muertes_comunes_grupoedad.png")
plt.tight_layout()
plt.show()
#predecir la causa de muerte basándonos en grupo etario, sexo, estado civil, provincia y atención médica