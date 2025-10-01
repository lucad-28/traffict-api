import pandas as pd
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import joblib
import tensorflow
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
    allow_origins=["http://localhost:3000"],  # URL de tu Next.js
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


if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )