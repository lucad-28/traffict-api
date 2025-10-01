import pandas as pd
import re
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import joblib
import tensorflow
from math import radians, sin, cos, sqrt, atan2
load_model = tensorflow.keras.models.load_model
LSTM = tensorflow.keras.layers.LSTM
CustomObjectScope = tensorflow.keras.utils.CustomObjectScope
import uvicorn

# Inicializar FastAPI
app = FastAPI(
    title="Traffic SPI Prediction API",
    description="API para predicción de Speed Performance Index usando LSTM",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y scalers al iniciar
print("Cargando modelo y scalers...")
try:
    # Usar CustomObjectScope para evitar problemas con argumentos deprecados
    custom_objects = {}
    modelo = load_model(
        'modelo_lstm_trafico_final.h5',
        custom_objects=custom_objects,
        compile=False
    )

    # Recompilar
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    print("✅ Modelo y scalers cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando archivos: {e}")
    modelo = None
    scaler_X = None
    scaler_y = None

print("Cargando metadata de estaciones...")
try:
    df_stations = pd.read_pickle('df_stations_necesario.pkl')
    print(f"✅ Metadata cargado: {len(df_stations)} estaciones")
except Exception as e:
    print(f"❌ Error cargando metadata: {e}")
    df_stations = None
    
# Modelo de datos para la entrada
class TrafficSequence(BaseModel):
    """
    Secuencia de 12 intervalos de tiempo (1 hora de historial)
    Cada intervalo contiene 7 features
    """
    sequence: List[List[float]] = Field(
        ...,
        description="Lista de 12 listas, cada una con 7 features: [Total_Flow, Avg_Occupancy, Avg_Speed, Hour, Day_of_Week, Lanes, Lane_Type_encoded]",
        min_items=12,
        max_items=12
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [
                    [150.0, 0.15, 65.0, 8, 0, 4, 1],  # t-60 min
                    [180.0, 0.18, 63.0, 8, 0, 4, 1],  # t-55 min
                    [200.0, 0.20, 60.0, 8, 0, 4, 1],  # t-50 min
                    [220.0, 0.22, 58.0, 8, 0, 4, 1],  # t-45 min
                    [240.0, 0.24, 55.0, 8, 0, 4, 1],  # t-40 min
                    [250.0, 0.25, 53.0, 8, 0, 4, 1],  # t-35 min
                    [260.0, 0.26, 50.0, 8, 0, 4, 1],  # t-30 min
                    [270.0, 0.28, 48.0, 8, 0, 4, 1],  # t-25 min
                    [280.0, 0.30, 45.0, 8, 0, 4, 1],  # t-20 min
                    [290.0, 0.32, 43.0, 9, 0, 4, 1],  # t-15 min
                    [300.0, 0.35, 40.0, 9, 0, 4, 1],  # t-10 min
                    [310.0, 0.38, 38.0, 9, 0, 4, 1]   # t-5 min
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta de la predicción"""
    spi_predicted: float = Field(..., description="Speed Performance Index predicho (0-100)")
    congestion_level: int = Field(..., description="Nivel de congestión: 0=Muy fluido, 1=Fluido, 2=Congestión leve, 3=Congestión severa")
    congestion_label: str = Field(..., description="Etiqueta descriptiva del nivel de congestión")
    status: str = Field(..., description="Estado de la predicción")


def classify_spi(spi: float) -> tuple:
    """Clasifica el SPI en nivel de congestión"""
    if spi > 75:
        return 0, "Very smooth"
    elif spi > 50:
        return 1, "Smooth"
    elif spi > 25:
        return 2, "Mild congestion"
    else:
        return 3, "Heavy congestion"


@app.get("/")
def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Traffic SPI Prediction API",
        "status": "running",
        "model_loaded": modelo is not None,
        "endpoints": {
            "prediction": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Verifica el estado de la API y modelo"""
    return {
        "status": "healthy" if modelo is not None else "unhealthy",
        "model_loaded": modelo is not None,
        "scalers_loaded": scaler_X is not None and scaler_y is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_spi(data: TrafficSequence):
    """
    Predice el SPI (Speed Performance Index) para el siguiente intervalo de 5 minutos
    
    **Input**: Secuencia de 12 intervalos (1 hora de historial) con 7 features cada uno:
    - Total_Flow: Flujo vehicular total
    - Avg_Occupancy: Ocupación promedio (0-1)
    - Avg_Speed: Velocidad promedio (mph)
    - Hour: Hora del día (0-23)
    - Day_of_Week: Día de la semana (0=Lunes, 6=Domingo)
    - Lanes: Número de carriles
    - Lane_Type_encoded: Tipo de carril codificado (0, 1, 2...)
    
    **Output**: SPI predicho y nivel de congestión
    """
    
    # Validar que el modelo esté cargado
    if modelo is None or scaler_X is None or scaler_y is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifica que los archivos existan: modelo_lstm_trafico_final.keras, scaler_X.pkl, scaler_y.pkl"
        )
    
    try:
        # Validar dimensiones de cada timestep
        for i, timestep in enumerate(data.sequence):
            if len(timestep) != 7:
                raise HTTPException(
                    status_code=400,
                    detail=f"Timestep {i} tiene {len(timestep)} features, se esperan 7"
                )
        
        # Convertir a numpy array
        X_input = np.array(data.sequence)  # Shape: (12, 7)
        
        # Normalizar
        X_scaled = scaler_X.transform(X_input.reshape(-1, 7)).reshape(1, 12, 7)  # Shape: (12, 7)
        
        # Predicción
        y_pred_scaled = modelo.predict(X_scaled, verbose=0)
        
        # Desnormalizar
        spi_predicted = scaler_y.inverse_transform(y_pred_scaled)[0][0]
        
        # Asegurar que esté en rango válido
        spi_predicted = float(np.clip(spi_predicted, 0, 100))
        
        # Clasificar nivel de congestión
        congestion_level, congestion_label = classify_spi(spi_predicted)
        
        return PredictionResponse(
            spi_predicted=round(spi_predicted, 2),
            congestion_level=congestion_level,
            congestion_label=congestion_label,
            status="success"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en el formato de datos: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )
    
class StationInfo(BaseModel):
    ID: int
    Fwy: int
    Dir: str
    District: int
    County: float
    City: float
    State_PM: float
    Abs_PM: float
    Latitude: float
    Longitude: float
    Type: str
    Lanes: int
    Name: str

class StationsResponse(BaseModel):
    total: int
    stations: List[StationInfo]

def clean_postmile(value):
    if isinstance(value, str):
        # quita letras iniciales, conserva solo el número
        match = re.search(r"[\d\.]+", value)
        if match:
            return float(match.group())
        return 0.0
    return float(value)

# Endpoint para obtener todas las estaciones
@app.get("/stations", response_model=StationsResponse)
def get_stations():
    """Devuelve información de todas las estaciones disponibles"""
    
    if df_stations is None:
        raise HTTPException(
            status_code=503,
            detail="Metadata de estaciones no disponible"
        )
    df_clean = df_stations.copy()
    df_clean["State_PM"] = df_clean["State_PM"].apply(clean_postmile)
    df_clean["Abs_PM"] = df_clean["Abs_PM"].apply(clean_postmile)
    df_clean = df_clean.replace([float('inf'), float('-inf')], pd.NA)
    df_clean = df_clean.dropna(subset=["Latitude", "Longitude"])
    df_clean = df_clean.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
    df_clean = df_clean[df_clean["Type"].isin(["ML", "HV"])]
    stations_list = df_clean.to_dict('records')
    
    return StationsResponse(
        total=len(stations_list),
        stations=stations_list
    )

# Endpoint para obtener una estación específica
@app.get("/stations/{station_id}", response_model=StationInfo)
def get_station_by_id(station_id: int):
    """Devuelve información de una estación específica por ID"""
    
    if df_stations is None:
        raise HTTPException(
            status_code=503,
            detail="Metadata de estaciones no disponible"
        )
    
    station = df_stations[df_stations['ID'] == station_id]
    
    if station.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Estación {station_id} no encontrada"
        )
    
    return station.to_dict('records')[0]

class RouteRequest(BaseModel):
    """Request para buscar rutas alternativas"""
    origin_station: int = Field(..., description="ID de estación origen")
    destination_station: int = Field(..., description="ID de estación destino")
    current_predictions: Dict[int, float] = Field(
        ..., 
        description="Diccionario con predicciones SPI actuales {station_id: spi_value}"
    )
    num_routes: int = Field(default=3, ge=1, le=5, description="Número de rutas alternativas (1-5)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "origin_station": 400001,
                "destination_station": 400050,
                "current_predictions": {
                    400001: 85.5,
                    400002: 45.2,
                    400003: 60.8,
                    400050: 75.3
                },
                "num_routes": 3
            }
        }

class RouteInfo(BaseModel):
    """Información detallada de una ruta"""
    route_id: int
    route_name: str
    stations: List[int]
    num_stations: int
    total_time_minutes: float
    total_distance_km: float
    avg_spi: float
    min_spi: float
    congested_segments: int
    total_segments: int
    congestion_percentage: float
    status: str

class RouteResponse(BaseModel):
    """Respuesta con rutas sugeridas"""
    origin: Dict[str, float]
    destination: Dict[str, float]
    routes: List[RouteInfo]
    recommendation: str
    graph_stats: Optional[Dict[str, int]] = None

def haversine_distance(coord1, coord2):
    """Calcula distancia en km entre dos coordenadas (lat, lon)"""
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    R = 6371
    return R * c


def find_connected_stations(station, stations_df, max_distance_km=5):
    """Encuentra estaciones conectadas (mismo freeway y dirección, cercanas)"""
    candidates = stations_df[
        (stations_df['Fwy'] == station['Fwy']) &
        (stations_df['Dir'] == station['Dir']) &
        (stations_df['ID'] != station['ID'])
    ]
    
    connected = []
    for _, candidate in candidates.iterrows():
        dist = haversine_distance(
            (station['Latitude'], station['Longitude']),
            (candidate['Latitude'], candidate['Longitude'])
        )
        
        if dist <= max_distance_km:
            connected.append({
                'id': candidate['ID'],
                'distance': dist,
                'lat': candidate['Latitude'],
                'lon': candidate['Longitude']
            })
    
    return connected


def get_congestion_level_label(spi):
    """Clasifica nivel de congestión según SPI"""
    if spi >= 75:
        return "fluido"
    elif spi >= 50:
        return "moderado"
    elif spi >= 25:
        return "congestionado"
    else:
        return "severo"


def build_traffic_graph(stations_df, spi_predictions):
    """
    Construye grafo dirigido con estaciones como nodos y segmentos como aristas
    Peso = tiempo estimado basado en distancia y SPI predicho
    """
    G = nx.DiGraph()

    # OPTIMIZACIÓN: Solo usar estaciones que tienen predicciones SPI
    relevant_station_ids = set(spi_predictions.keys())
    relevant_stations_df = stations_df[stations_df['ID'].isin(relevant_station_ids)]

    print(f"🔧 Construyendo grafo con {len(relevant_stations_df)} estaciones (de {len(stations_df)} totales)")

    # Agregar nodos solo para estaciones relevantes
    for _, station in relevant_stations_df.iterrows():
        G.add_node(
            station['ID'],
            lat=station['Latitude'],
            lon=station['Longitude'],
            fwy=station['Fwy'],
            direction=station['Dir']
        )

    # Agregar aristas solo entre estaciones relevantes
    for _, station in relevant_stations_df.iterrows():
        connected = find_connected_stations(station, relevant_stations_df)

        for next_station in connected:
            next_id = next_station['id']
            distance = next_station['distance']

            spi = spi_predictions.get(next_id, 50)

            # Calcular peso (tiempo estimado)
            speed_factor = max(0.1, spi / 100)
            travel_time = distance / speed_factor

            G.add_edge(
                station['ID'],
                next_id,
                weight=travel_time,
                distance=distance,
                spi=spi,
                congestion_level=get_congestion_level_label(spi)
            )

    print(f"✅ Grafo construido: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    return G


def calculate_route_metrics(path, graph):
    """Calcula métricas detalladas de una ruta"""
    total_time = 0
    total_distance = 0
    spi_values = []
    congested_count = 0
    
    for i in range(len(path) - 1):
        edge = graph[path[i]][path[i+1]]
        
        total_time += edge['weight']
        total_distance += edge['distance']
        spi = edge['spi']
        spi_values.append(spi)
        
        if spi < 50:
            congested_count += 1
    
    return {
        'time': total_time,
        'distance': total_distance,
        'avg_spi': np.mean(spi_values) if spi_values else 50,
        'min_spi': min(spi_values) if spi_values else 50,
        'congested': congested_count
    }


def get_route_status(avg_spi):
    """Clasifica status general de la ruta"""
    if avg_spi >= 75:
        return "excellent"
    elif avg_spi >= 60:
        return "good"
    elif avg_spi >= 40:
        return "moderate"
    else:
        return "congested"


def generate_recommendation(routes):
    """Genera recomendación inteligente basada en las rutas"""
    if not routes:
        return "No hay rutas disponibles"
    
    best_route = routes[0]
    
    if best_route['avg_spi'] < 40 and len(routes) > 1:
        alternative = routes[1]
        time_diff = alternative['total_time_minutes'] - best_route['total_time_minutes']
        
        if time_diff < 5 and alternative['avg_spi'] > best_route['avg_spi'] + 10:
            return (f"⚠️ Ruta principal congestionada (SPI {best_route['avg_spi']:.1f}). "
                   f"Se recomienda Alternativa 1: solo {time_diff:.1f} min más "
                   f"pero con mejor flujo (SPI {alternative['avg_spi']:.1f})")
    
    if best_route['avg_spi'] >= 60:
        return f"✅ Ruta óptima con buen flujo de tráfico (SPI {best_route['avg_spi']:.1f})"
    
    return f"⚡ Ruta con tráfico moderado (SPI {best_route['avg_spi']:.1f}). Considere horarios alternativos."


def find_optimal_routes_dijkstra(origin_id, dest_id, stations_df, spi_predictions, k=3):
    """
    Encuentra las k mejores rutas entre origen y destino usando Dijkstra
    """
    import time
    start_time = time.time()

    G = build_traffic_graph(stations_df, spi_predictions)
    print(f"⏱️ Grafo construido en {time.time() - start_time:.2f}s")

    if origin_id not in G or dest_id not in G:
        return {
            "error": "Estación origen o destino no encontrada en el grafo",
            "origin": origin_id,
            "destination": dest_id
        }

    try:
        path_start = time.time()
        best_path = nx.shortest_path(G, origin_id, dest_id, weight='weight')
        best_time = nx.shortest_path_length(G, origin_id, dest_id, weight='weight')
        print(f"⏱️ Ruta óptima encontrada en {time.time() - path_start:.2f}s")
    except nx.NetworkXNoPath:
        return {
            "error": "No existe ruta entre origen y destino",
            "origin": origin_id,
            "destination": dest_id
        }

    try:
        alt_start = time.time()
        # OPTIMIZACIÓN: Usar iterador y limitar a k rutas con timeout
        all_paths = []
        path_generator = nx.shortest_simple_paths(G, origin_id, dest_id, weight='weight')

        for i, path in enumerate(path_generator):
            if i >= k:  # Solo obtener k rutas
                break
            all_paths.append(path)

        print(f"⏱️ {len(all_paths)} rutas alternativas encontradas en {time.time() - alt_start:.2f}s")
    except nx.NetworkXNoPath:
        all_paths = [best_path]

    paths_to_analyze = all_paths
    
    routes = []
    for i, path in enumerate(paths_to_analyze):
        metrics = calculate_route_metrics(path, G)
        
        routes.append({
            "route_id": i + 1,
            "route_name": "Ruta Óptima" if i == 0 else f"Alternativa {i}",
            "stations": path,
            "num_stations": len(path),
            "total_time_minutes": round(metrics['time'], 2),
            "total_distance_km": round(metrics['distance'], 2),
            "avg_spi": round(metrics['avg_spi'], 1),
            "min_spi": round(metrics['min_spi'], 1),
            "congested_segments": metrics['congested'],
            "total_segments": len(path) - 1,
            "congestion_percentage": round((metrics['congested'] / max(1, len(path) - 1)) * 100, 1),
            "status": get_route_status(metrics['avg_spi'])
        })
    
    recommendation = generate_recommendation(routes)
    
    return {
        "origin": {
            "station_id": origin_id,
            "current_spi": spi_predictions.get(origin_id, 50)
        },
        "destination": {
            "station_id": dest_id,
            "current_spi": spi_predictions.get(dest_id, 50)
        },
        "routes": routes,
        "recommendation": recommendation,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()
        }
    }

@app.post("/routes/suggest", response_model=RouteResponse)
def suggest_alternative_routes(request: RouteRequest):
    """
    Encuentra rutas alternativas óptimas usando algoritmo de Dijkstra
    
    **Input**:
    - origin_station: ID de estación origen
    - destination_station: ID de estación destino  
    - current_predictions: Diccionario con SPI predicho para cada estación
    - num_routes: Número de rutas alternativas a devolver (1-5)
    
    **Output**:
    - Rutas ordenadas por tiempo estimado
    - Métricas detalladas (tiempo, distancia, SPI promedio)
    - Recomendación inteligente
    """
    
    if df_stations is None:
        raise HTTPException(
            status_code=503,
            detail="Metadata de estaciones no disponible"
        )
    
    try:
        result = find_optimal_routes_dijkstra(
            origin_id=request.origin_station,
            dest_id=request.destination_station,
            stations_df=df_stations,
            spi_predictions=request.current_predictions,
            k=request.num_routes
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=404,
                detail=result["error"]
            )
        
        return RouteResponse(**result)
        
    except nx.NetworkXError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en el grafo de rutas: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al calcular rutas: {str(e)}"
        )

if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )