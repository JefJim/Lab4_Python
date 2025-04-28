import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import folium
from folium.plugins import MarkerCluster
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from folium.plugins import HeatMap

#1. Usar  python
#2 Utilice el conjunto de datos, de su elección
# 3. Cargar conjunto de datos desde GitHub
url = "https://raw.githubusercontent.com/JefJim/Lab4_Python/main/Costa%20Rica%20Total%20deceases%202014%20-%202021.csv"

dtype_dict = {28: str}  # Ajusta el índice según tu CSV
try:
    df = pd.read_csv(
        url,
        encoding='utf-8-sig',
        dtype=dtype_dict,
        low_memory=False
    )
    print("Datos cargados exitosamente desde GitHub")
except Exception as e:
    print(f"Error al cargar datos: {e}")
    df = pd.read_csv(
        "Costa Rica Total deceases 2014 - 2021.csv",
        encoding='utf-8-sig',
        dtype=dtype_dict,
        low_memory=False
    )
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

# =============================================================================
# PREPARACIÓN DE DATOS PARA MODELO DE RIESGO TEMPORAL-ESPACIAL
# =============================================================================
#modelo XGBoost Classifier
# Convertir mes a numérico de manera robusta
if df['mes'].dtype == 'object':
    # Si los meses están en texto (Enero, Febrero...)
    meses_espanol = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
        'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    df['mes_num'] = df['mes'].map(meses_espanol)
else:
    # Si ya son numéricos pero como floats
    df['mes_num'] = df['mes'].astype(float).astype(int)

# Asegurar que el año sea numérico
df['anio_num'] = pd.to_numeric(df['anio'], errors='coerce')

# Eliminar filas con fechas inválidas
df = df.dropna(subset=['anio_num', 'mes_num'])

# Crear fecha con formato correcto (asegurando día 1)
df['fecha_str'] = (
    df['anio_num'].astype(int).astype(str) + '-' + 
    df['mes_num'].astype(int).astype(str) + '-01'
)

df['fecha'] = pd.to_datetime(
    df['fecha_str'],
    format='%Y-%m-%d',
    errors='coerce'
)

# Eliminar filas con fechas no válidas
df = df.dropna(subset=['fecha'])

with open("Distritos_de_Costa_Rica.geojson", encoding="utf-8") as f:
    distritos_geojson = json.load(f)

#modelo XGBoost Classifier
def entrenar_modelo_avanzado(df):
    # Crear características temporales avanzadas
    df['dia_año'] = df['fecha'].dt.dayofyear
    df['semana_epidemiologica'] = df['fecha'].dt.isocalendar().week
    
    # Binning de edad
    bins = [0, 18, 30, 45, 60, 75, 90, 120]
    labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-90', '90+']
    df['edad_bin'] = pd.cut(df['edad'], bins=bins, labels=labels)
    
    # Preparar variables
    X = pd.get_dummies(df[['provincia', 'sexo', 'edad_bin', 'dia_año', 'semana_epidemiologica']])
    y = (df['edad'] < df['edad'].quantile(0.25)).astype(int)  # Riesgo: menor que percentil 25 de edad
    
    # Pipeline del modelo
    modelo = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    # Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(modelo, X, y, cv=tscv, scoring='roc_auc')
    print(f"\nValidación del modelo - AUC promedio: {scores.mean():.2f} ± {scores.std():.2f}")
    
    # Entrenar modelo final
    modelo.fit(X, y)
    return modelo

modelo_riesgo = entrenar_modelo_avanzado(df)

def preparar_fechas(df):
    # Convertir mes a numérico
    if df['mes'].dtype == 'object':
        meses_espanol = {
            'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
            'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
        }
        df['mes_num'] = df['mes'].map(meses_espanol)
    else:
        df['mes_num'] = pd.to_numeric(df['mes'], errors='coerce')
    
    # Asegurar año numérico
    df['anio_num'] = pd.to_numeric(df['anio'], errors='coerce')
    
    # Crear fecha
    df['fecha'] = pd.to_datetime(
        df['anio_num'].astype(str) + '-' + 
        df['mes_num'].astype(str) + '-01',
        errors='coerce'
    )
    
    return df.dropna(subset=['fecha'])

df = preparar_fechas(df)
# --- Función de análisis ---
def analisis_completo(group):
    # Causa principal de muerte
    causa_principal = group['descripcion_causa_muerte'].mode()[0]
    count_causa = (group['descripcion_causa_muerte'] == causa_principal).sum()
    
    # Estadísticas por grupo de edad
    edad_stats = group['edad'].describe()
    
    # Porcentaje por sexo
    sexo_counts = group['sexo'].value_counts(normalize=True).to_dict()
    
    # Tendencia temporal (últimos 12 meses)
    ultimo_año = group[group['fecha'] >= (group['fecha'].max() - pd.DateOffset(months=12))]
    cambio = (ultimo_año.shape[0] - group.shape[0]/7) / (group.shape[0]/7) * 100  # Cambio porcentual
    
    return pd.Series({
        'Causa_Principal': causa_principal,
        'Porcentaje_Causa_Principal': f"{count_causa/len(group)*100:.1f}%",
        'Total_Defunciones': len(group),
        'Edad_Promedio': f"{edad_stats['mean']:.1f} ± {edad_stats['std']:.1f}",
        'Distribucion_Sexo': sexo_counts,
        'Tendencia_12Meses': f"{cambio:.1f}%",
        'Top_3_Causas': group['descripcion_causa_muerte'].value_counts().nlargest(3).to_dict()
    })

# Aplicar análisis por provincia
analisis_provincias = df.groupby('provincia').apply(analisis_completo).reset_index()

# 3. Generar mapa interactivo con información enriquecida
m = folium.Map(location=[9.7489, -83.7534], zoom_start=7, tiles='cartodbpositron')

# Capa de clusters para mejor visualización
marker_cluster = MarkerCluster().add_to(m)

# Coordenadas de provincias
coordenadas = {
    'San José': (9.93, -84.08),
    'Alajuela': (10.02, -84.22),
    'Cartago': (9.86, -83.92),
    'Heredia': (10.00, -84.12),
    'Guanacaste': (10.43, -85.40),
    'Puntarenas': (9.97, -84.83),
    'Limón': (10.00, -83.03)
}

# Añadir marcadores con información detallada
for provincia, data in analisis_provincias.iterrows():
    # Obtener coordenadas
    lat, lon = coordenadas.get(data['provincia'], (0, 0))
    
    # Crear contenido del popup
    html = f"""
    <div style="width: 300px;">
        <h4 style="color: #2b8cbe; border-bottom: 1px solid #eee; padding-bottom: 5px;">{data['provincia']}</h4>
        <p><b>Total defunciones (2014-2021):</b> {data['Total_Defunciones']:,}</p>
        <p><b>Causa principal:</b> {data['Causa_Principal']} ({data['Porcentaje_Causa_Principal']})</p>
        <p><b>Edad promedio:</b> {data['Edad_Promedio']} años</p>
        <p><b>Tendencia últimos 12 meses:</b> {data['Tendencia_12Meses']}</p>
        
        <h5 style="margin-top: 10px; color: #2b8cbe;">Distribución por sexo:</h5>
        <ul>
            <li>Masculino: {data['Distribucion_Sexo'].get('Hombres', 0)*100:.1f}%</li>
            <li>Femenino: {data['Distribucion_Sexo'].get('Mujeres', 0)*100:.1f}%</li>
        </ul>
        
        <h5 style="margin-top: 10px; color: #2b8cbe;">Top 3 causas de muerte:</h5>
        <ol>
    """
    
    for causa, count in data['Top_3_Causas'].items():
        porcentaje = count/data['Total_Defunciones']*100
        html += f"<li>{causa} ({porcentaje:.1f}%)</li>"
    
    html += """
        </ol>
    </div>
    """
    
    # Añadir marcador al cluster
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(html, max_width=350),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# 4. Añadir capa de calor de densidad
from folium.plugins import HeatMap

# Preparar datos para heatmap (lat, lon, peso)
heat_data = []
for _, row in df.iterrows():
    provincia = row['provincia']
    if provincia in coordenadas:
        lat, lon = coordenadas[provincia]
        heat_data.append([lat, lon, 1])  # Peso 1 por cada registro

HeatMap(heat_data, radius=25, blur=15).add_to(m)

# 5. Añadir controles de capas
folium.LayerControl().add_to(m)

# 6. Guardar mapa mejorado
m.save('mapa_mortalidad_provincias.html')
print("\n✅ Mapa interactivo mejorado generado: 'mapa_mortalidad_provincias.html'")

with open('Distritos_de_Costa_Rica.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

# 2. Crear un diccionario de coordenadas por distrito
# (usaremos el centroide de cada polígono)
distrito_coords = {}
distrito_provincia = {}
def obtener_coordenadas(feature):
    """Función robusta para extraer coordenadas de un feature GeoJSON"""
    if feature is None:
        raise ValueError("El feature es None")
    
    geometry = feature.get('geometry')
    if geometry is None:
        raise ValueError("El feature no tiene geometría")
    
    coords = geometry.get('coordinates')
    if coords is None:
        raise ValueError("La geometría no tiene coordenadas")
    
    # Debug: Mostrar tipo de geometría
    geom_type = geometry.get('type')    
    try:
        if geom_type == 'Point':
            return float(coords[1]), float(coords[0])  # (lat, lon)
        
        elif geom_type == 'Polygon':
            exterior_ring = coords[0]
            lons = [float(p[0]) for p in exterior_ring]
            lats = [float(p[1]) for p in exterior_ring]
            return sum(lats)/len(lats), sum(lons)/len(lons)
        
        elif geom_type == 'MultiPolygon':
            all_lats = []
            all_lons = []
            for polygon in coords:
                exterior_ring = polygon[0]
                all_lons.extend([float(p[0]) for p in exterior_ring])
                all_lats.extend([float(p[1]) for p in exterior_ring])
            return sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons)
        
        else:
            raise ValueError(f"Tipo de geometría no soportado: {geom_type}")
    
    except (IndexError, TypeError) as e:
        raise ValueError(f"Error procesando coordenadas: {str(e)}") from e

# Uso seguro en tu flujo principal:
for feature in geojson_data['features']:
    propiedades = feature['properties']
    nombre_provincia = propiedades['NOM_PROV']
    nombre_distrito = propiedades['NOM_DIST']
    try:
        lat, lon = obtener_coordenadas(feature)
        # Aquí puedes usar lat y lon para tus marcadores
        
    except ValueError as e:
        print(f"⚠️ No se pudo procesar feature: {str(e)}")
        print("Feature problemático:", feature.get('properties', {}).get('NOM_PROV', 'Sin nombre'))
        continue  # Continuar con el siguiente feature

# 3. Función para encontrar el distrito más cercano cuando no hay coincidencia exacta
from geopy.distance import geodesic

def encontrar_distrito_mas_cercano(provincia, distrito_buscar):
    posibles = []
    for (p, d), coord in distrito_coords.items():
        if p == provincia:
            posibles.append((d, coord))
    
    if not posibles:
        return None
    
    # Buscar coincidencia exacta primero
    for d, coord in posibles:
        if d.lower() == distrito_buscar.lower():
            return (d, coord)
    
    # Si no hay coincidencia exacta, encontrar el más cercano por nombre
    from difflib import get_close_matches
    nombres_distritos = [d for d, coord in posibles]
    matches = get_close_matches(distrito_buscar.lower(), [d.lower() for d in nombres_distritos], n=1)
    
    if matches:
        match = matches[0]
        for d, coord in posibles:
            if d.lower() == match:
                return (d, coord)
    
    return None
# Diccionario completo de mapeo de distritos
mapeo_distritos = {
    # Distritos que son cantones (usar distrito cabecera)
    'ABANGARES': 'LAS JUNTAS',
    'ACOSTA': 'SAN IGNACIO',
    'AGUIRRE': 'QUEPOS',
    'ALAJUELA': 'ALAJUELA (CENTRO)',
    'ALFARO RUÍZ': 'ZARCERO',
    'ALVARADO': 'PACAYAS',
    'ASERRÍ': 'ASERRI',
    'BARVA': 'SAN PEDRO',
    'BELÉN': 'SAN ANTONIO',
    'CARRILLO': 'FILADELFIA',
    'CARTAGO': 'ORIENTAL',
    'CORREDORES': 'CORREDOR',
    'COTO BRUS': 'SAN VITO',
    'CURRIDABAT': 'SAN PABLO',
    'DOTA': 'SANTA MARIA',
    'EL GUARCO': 'EL TEJAR',
    'ESCAZÚ': 'ESCAZU',
    'ESPARZA': 'ESPARZA',
    'FLORES': 'SAN JOAQUIN',
    'GARABITO': 'JACO',
    'GOICOECHEA': 'SAN FRANCISCO',
    'GOLFITO': 'GOLFITO',
    'GRECIA': 'GRECIA',
    'GUÁCIMO': 'GUACIMO',
    'GUATUSO': 'SAN RAFAEL',
    'HEREDIA': 'HEREDIA (CENTRO)',
    'JIMÉNEZ': 'JUAN VIÑAS',
    'LA UNIÓN': 'TRES RIOS',
    'LEÓN CORTÉS': 'SAN PABLO',
    'LEÓN CORTÉS CASTRO': 'SAN PABLO',
    'LIBERIA': 'LIBERIA (CABECERA)',
    'LIMÓN': 'LIMON (CENTRO)',
    'LOS CHILES': 'LOS CHILES',
    'MONTES DE OCA': 'SAN PEDRO',
    'MONTES DE ORO': 'MIRAMAR',
    'MORA': 'CIUDAD COLON',
    'MORAVIA': 'SAN VICENTE',
    'NANDAYURE': 'CARMONA',
    'NARANJO': 'NARANJO',
    'NICOYA': 'NICOYA',
    'OROTINA': 'OROTINA',
    'OSA': 'PUERTO CORTES',
    'PALMARES': 'PALMARES',
    'PARAÍSO': 'PARAISO',
    'PARRITA': 'PARRITA',
    'PÉREZ ZELEDÓN': 'SAN ISIDRO DEL GENERAL',
    'POÁS': 'SAN PEDRO',
    'POCOCÍ': 'GUAPILES',
    'PUNTARENAS': 'PUNTARENAS (CENTRO)',
    'PURISCAL': 'SANTIAGO',
    'QUEPOS HASTA 2015 SE LLAMÓ AGUIRRE': 'QUEPOS',
    'RÍO CUARTO': 'RIO CUARTO',
    'RÍO CUARTO CREADO COMO CANTÓN EN JULIO 2019': 'RIO CUARTO',
    'SAN CARLOS': 'CIUDAD QUESADA',
    'SAN ISIDRO': 'SAN ISIDRO',
    'SAN JOSÉ': 'CATEDRAL',
    'SAN MATEO': 'SAN MATEO',
    'SAN PABLO': 'SAN PABLO',
    'SAN RAFAEL': 'SAN RAFAEL',
    'SAN RAMÓN': 'SAN RAMON',
    'SANTA ANA': 'SANTA ANA',
    'SANTA BÁRBARA': 'SANTA BARBARA',
    'SANTA CRUZ': 'SANTA CRUZ',
    'SANTO DOMINGO': 'SANTO DOMINGO',
    'SARAPIQUÍ': 'PUERTO VIEJO',
    'SARCHÍ HASTA 2018 SE LLAMÓ VALVERDE VEGA': 'SARCHI NORTE',
    'SIQUIRRES': 'SIQUIRRES',
    'TALAMANCA': 'BILLY KING',
    'TARRAZÚ': 'SAN MARCOS',
    'TIBÁS': 'SAN JUAN',
    'TILARÁN': 'TILARAN',
    'TURRIALBA': 'TURRIALBA',
    'TURRUBARES': 'SAN PABLO',
    'UPALA': 'UPALA',
    'VALVERDE VEGA': 'SARCHI NORTE',
    'VÁZQUEZ DE CORONADO': 'SAN ISIDRO',
    'ZARCERO HASTA 2010 SE LLAMÓ ALFARO RUÍZ': 'ZARCERO',
    
    # Correcciones adicionales para nombres especiales
    'DESAMPARADOS': 'DESAMPARADOS',
    'ALAJUELITA': 'ALAJUELITA',
    'MERCEDES': 'MERCEDES',
    'SAN VICENTE': 'SAN VICENTE',
    'CONCEPCIÓN': 'CONCEPCION',
    'GUADALUPE': 'GUADALUPE',
    'CALLE BLANCOS': 'CALLE BLANCOS',
    'PATARRÁ': 'PATARRA',
    'SAN SEBASTIÁN': 'SAN SEBASTIAN',
    'ULATINA': 'ULATINA',
    'SAGRADA FAMILIA': 'SAGRADA FAMILIA',
    'SAN JUAN DE DIOS': 'SAN JUAN DE DIOS',
    'HOSPITAL': 'HOSPITAL',
    'CATEDRAL': 'CATEDRAL',
    'ZAPOTE': 'ZAPOTE',
    'SAN FRANCISCO': 'SAN FRANCISCO',
    'URUCA': 'URUCA',
    'MATA REDONDA': 'MATA REDONDA',
    'PAVAS': 'PAVAS',
    'HATILLO': 'HATILLO',
    'SAN ISIDRO DE CORONADO': 'SAN ISIDRO',
    'SANTA ROSA': 'SANTA ROSA',
    'SAN RAFAEL DE CORONADO': 'SAN RAFAEL',
    'DULCE NOMBRE': 'DULCE NOMBRE',
    'SAN ANTONIO': 'SAN ANTONIO',
    'LA RIVERA': 'LA RIVERA',
    'SANTA MARÍA': 'SANTA MARIA',
    'SAN JERÓNIMO': 'SAN JERONIMO',
    'SAN JUAN DE MATA': 'SAN JUAN DE MATA',
    'SAN LUIS': 'SAN LUIS',
    'CARRILLO': 'CARRILOS',
    'BELLAVISTA': 'BELLAVISTA',
    'LIMONCILLO': 'LIMONCITO',
    'MATINA': 'MATINA',
    'BATÁN': 'BATAN',
    'CARRILLOS': 'CARRILLOS',
    'SAN RAFAEL DE HEREDIA': 'SAN RAFAEL',
    'SAN ISIDRO DE HEREDIA': 'SAN ISIDRO',
    'SAN FRANCISCO DE HEREDIA': 'SAN FRANCISCO',
    'SAN PABLO DE HEREDIA': 'SAN PABLO',
    'SANTO DOMINGO DE HEREDIA': 'SANTO DOMINGO',
    'SANTA BÁRBARA DE HEREDIA': 'SANTA BARBARA',
    'SAN RAFAEL DE ALAJUELA': 'SAN RAFAEL',
    'SAN ISIDRO DE ALAJUELA': 'SAN ISIDRO',
    'SAN ANTONIO DE ALAJUELA': 'SAN ANTONIO',
    'SAN JOSECITO': 'SAN JOSECITO',
    'SAN JOSÉ DE LA MONTAÑA': 'SAN JOSE DE LA MONTAÑA',
    'SAN RAFAEL ABAJO': 'SAN RAFAEL ABAJO',
    'SAN RAFAEL ARRIBA': 'SAN RAFAEL ARRIBA',
    'SAN JUAN GRANDE': 'SAN JUAN GRANDE',
    'SAN JUAN DE DIOS': 'SAN JUAN DE DIOS',
    'SAN VICENTE DE MORAVIA': 'SAN VICENTE',
    'SAN JERÓNIMO DE MORAVIA': 'SAN JERONIMO',
    'SAN VICENTE DE SAN JOSÉ': 'SAN VICENTE',
    'SAN RAFAEL DE SAN JOSÉ': 'SAN RAFAEL',
    'SAN ANTONIO DE SAN JOSÉ': 'SAN ANTONIO',
    'SAN ISIDRO DE SAN JOSÉ': 'SAN ISIDRO',
    'SAN FRANCISCO DE SAN JOSÉ': 'SAN FRANCISCO',
    'SAN PABLO DE SAN JOSÉ': 'SAN PABLO',
    'SANTO DOMINGO DE SAN JOSÉ': 'SANTO DOMINGO',
    'SANTA BÁRBARA DE SAN JOSÉ': 'SANTA BARBARA',
    'SANTA MARÍA DE SAN JOSÉ': 'SANTA MARIA',
    'SAN JOSÉ DE SAN JOSÉ': 'CATEDRAL',
    'SAN SEBASTIÁN DE SAN JOSÉ': 'SAN SEBASTIAN',
    'SAN MIGUEL DE SAN JOSÉ': 'SAN MIGUEL',
    'SAN JUAN DE SAN JOSÉ': 'SAN JUAN',
    'SAN PEDRO DE SAN JOSÉ': 'SAN PEDRO',
    'SAN RAFAEL DE SAN JOSÉ': 'SAN RAFAEL',
    'SAN ANTONIO DE SAN JOSÉ': 'SAN ANTONIO',
    'SAN ISIDRO DE SAN JOSÉ': 'SAN ISIDRO',
    'SAN FRANCISCO DE SAN JOSÉ': 'SAN FRANCISCO',
    'SAN PABLO DE SAN JOSÉ': 'SAN PABLO',
    'SANTO DOMINGO DE SAN JOSÉ': 'SANTO DOMINGO',
    'SANTA BÁRBARA DE SAN JOSÉ': 'SANTA BARBARA',
    'SANTA MARÍA DE SAN JOSÉ': 'SANTA MARIA',
    'SAN JOSÉ DE SAN JOSÉ': 'CATEDRAL',
    'SAN SEBASTIÁN DE SAN JOSÉ': 'SAN SEBASTIAN',
    'SAN MIGUEL DE SAN JOSÉ': 'SAN MIGUEL',
    'SAN JUAN DE SAN JOSÉ': 'SAN JUAN',
    'SAN PEDRO DE SAN JOSÉ': 'SAN PEDRO'
}
def encontrar_distrito_geojson(nombre_distrito, provincia=None):
    # Normalizar el nombre del distrito
    nombre_distrito = nombre_distrito.upper().strip()
    
    # Aplicar mapeo si existe
    nombre_mapeado = mapeo_distritos.get(nombre_distrito, nombre_distrito)
    
    # Buscar coincidencia exacta primero
    for feature in geojson_data['features']:
        props = feature['properties']
        if props['NOM_DIST'].upper() == nombre_mapeado:
            return feature
    
    # Si no se encuentra, buscar coincidencia parcial
    for feature in geojson_data['features']:
        props = feature['properties']
        if nombre_mapeado in props['NOM_DIST'].upper() or props['NOM_DIST'].upper() in nombre_mapeado:
            print(f"⚠️ Coincidencia parcial: {nombre_distrito} -> {props['NOM_DIST']}")
            return feature
    
    # Si aún no se encuentra, usar búsqueda difusa
    from difflib import get_close_matches
    nombres_disponibles = [f['properties']['NOM_DIST'].upper() for f in geojson_data['features']]
    matches = get_close_matches(nombre_mapeado, nombres_disponibles, n=1, cutoff=0.6)
    
    if matches:
        for feature in geojson_data['features']:
            if feature['properties']['NOM_DIST'].upper() == matches[0]:
                print(f"⚠️ Usando coincidencia difusa: {nombre_distrito} -> {matches[0]}")
                return feature
    
    print(f"❌ Distrito no encontrado: {nombre_distrito}")
    return None
# 4. Análisis por distrito
def analisis_por_distrito(group):
    causa_principal = group['descripcion_causa_muerte'].mode()[0]
    count_causa = (group['descripcion_causa_muerte'] == causa_principal).sum()
    
    # Estadísticas por grupo de edad
    edad_stats = group['edad'].describe()
    
    # Porcentaje por sexo
    sexo_counts = group['sexo'].value_counts(normalize=True).to_dict()
    
    # Tendencia temporal (últimos 12 meses)
    ultimo_año = group[group['fecha'] >= (group['fecha'].max() - pd.DateOffset(months=12))]
    cambio = (ultimo_año.shape[0] - group.shape[0]/7) / (group.shape[0]/7) * 100  # Cambio porcentual
    
    return pd.Series({
        'Causa_Principal': causa_principal,
        'Porcentaje_Causa_Principal': f"{count_causa/len(group)*100:.1f}%",
        'Total_Defunciones': len(group),
        'Edad_Promedio': f"{edad_stats['mean']:.1f} ± {edad_stats['std']:.1f}",
        'Distribucion_Sexo': sexo_counts,
        'Tendencia_12Meses': f"{cambio:.1f}%",
        'Top_3_Causas': group['descripcion_causa_muerte'].value_counts().nlargest(3).to_dict()
    })

# Aplicar análisis por distrito
df['distrito_normalizado'] = df['distrito_muerte'].str.upper().str.strip().map(mapeo_distritos).fillna(df['distrito_muerte'].str.upper().str.strip())
# Luego generamos el análisis por distrito normalizado
analisis_distritos = df.groupby(['distrito_normalizado']).apply(analisis_por_distrito).reset_index()

# Creamos el mapa
m_distritos = folium.Map(location=[9.7489, -83.7534], zoom_start=8, tiles='cartodbpositron')
# 5. Crear mapa interactivo por distrito
# Capa de clusters
marker_cluster = MarkerCluster().add_to(m_distritos)

# Añadir marcadores para cada distrito con datos
# Primero normalizamos los nombres en el DataFrame
df['distrito_normalizado'] = df['distrito_muerte'].str.upper().str.strip().map(mapeo_distritos).fillna(df['distrito_muerte'].str.upper().str.strip())

# Luego generamos el análisis por distrito normalizado
analisis_distritos = df.groupby(['distrito_normalizado']).apply(analisis_por_distrito).reset_index()

# Creamos el mapa
m_distritos = folium.Map(location=[9.7489, -83.7534], zoom_start=8, tiles='cartodbpositron')
marker_cluster = MarkerCluster().add_to(m_distritos)

for _, row in analisis_distritos.iterrows():
    distrito_nombre = row['distrito_normalizado']
    
    feature = encontrar_distrito_geojson(distrito_nombre)
    if not feature:
        continue
        
    propiedades = feature['properties']
    distrito_real = propiedades['NOM_DIST']
    provincia_real = propiedades['NOM_PROV']
    # Primero obtenemos el diccionario de distribución por sexo
    distribucion_sexo = row.get('Distribucion_Sexo', {})

    # Calculamos los porcentajes por separado
    porc_masculino = distribucion_sexo.get('Masculino', distribucion_sexo.get('Hombres', 0)) * 100
    porc_femenino = distribucion_sexo.get('Femenino', distribucion_sexo.get('Mujeres', 0)) * 100

    # Obtener coordenadas
    try:
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            coords = geometry['coordinates'][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            lat, lon = sum(lats)/len(lats), sum(lons)/len(lons)
        elif geometry['type'] == 'MultiPolygon':
            all_lats = []
            all_lons = []
            for polygon in geometry['coordinates']:
                coords = polygon[0]
                all_lons.extend([c[0] for c in coords])
                all_lats.extend([c[1] for c in coords])
            lat, lon = sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons)
        else:
            continue
    except Exception as e:
        print(f"Error calculando centroide para {distrito_real}: {str(e)}")
        continue
    
    # Crear popup con los datos
    html = f"""
    <div style="width: 300px; font-family: Arial, sans-serif;">
        <h3 style="color: #2b8cbe; border-bottom: 2px solid #2b8cbe; padding-bottom: 5px; margin-bottom: 10px;">
            {distrito_real}, {provincia_real}
        </h3>
        <p><b>Total defunciones:</b> {row.get('Total_Defunciones', 'N/A')}</p>
        <p><b>Edad promedio:</b> {row.get('Edad_Promedio', 'N/A')}</p>
        
        <h4 style="color: #2b8cbe; margin-bottom: 5px; font-size: 14px;">Distribución por sexo:</h4>
        <ul style="margin-top: 5px; padding-left: 20px;">
            <li>Masculino: {porc_masculino:.1f}%</li>
            <li>Femenino: {porc_femenino:.1f}%</li>
        </ul>
        
        <h4 style="color: #2b8cbe; margin-bottom: 5px; font-size: 14px;">Top 3 causas de muerte:</h4>
        <ol style="margin-top: 5px; padding-left: 20px;">
    """

    # Añadir las causas de muerte
    top3_causas = row.get('Top_3_Causas', {})
    if isinstance(top3_causas, dict):
        for causa, count in top3_causas.items():
            total = row.get('Total_Defunciones', 1)
            porcentaje = (count / total) * 100 if total else 0
            html += f"<li style='margin-bottom: 3px;'>{causa} <b>({porcentaje:.1f}%)</b></li>"
    else:
        html += "<li>Datos no disponibles</li>"

    # Cerrar el HTML
    html += """
        </ol>
        <div style="margin-top: 10px; font-size: 11px; color: #666; text-align: right;">
            Datos: 2014-2021 | Distrito original: """ + str(row.get('distrito_muerte', 'N/A')) + """
        </div>
    </div>
    """
    
    # Añadir marcador
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(html, max_width=350),
        icon=folium.Icon(color='blue', icon='info-circle', prefix='fa'),
        tooltip=f"{distrito_real}, {provincia_real}"
    ).add_to(marker_cluster)

# Añadir capa de GeoJSON
folium.GeoJson(
    geojson_data,
    style_function=lambda x: {'fillColor': '#2b8cbe', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.2},
    tooltip=folium.GeoJsonTooltip(
        fields=['NOM_PROV', 'NOM_DIST'],
        aliases=['Provincia:', 'Distrito:'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    )
).add_to(m_distritos)

# Guardar el mapa
m_distritos.save('mapa_mortalidad_distrito.html')
print("✅ Mapa generado exitosamente: 'mapa_mortalidad_por_distrito_final.html'")