from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import uvicorn

# Inicializar FastAPI
app = FastAPI(
    title="Traffic SPI Prediction API",
    description="API para predicción de Speed Performance Index usando LSTM",
    version="1.0.0"
)

# Cargar modelo y scalers al iniciar
print("Cargando modelo y scalers...")
try:
    modelo = load_model('modelo_lstm_trafico_final.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    print("✅ Modelo y scalers cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando archivos: {e}")
    modelo = None
    scaler_X = None
    scaler_y = None


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
        X_scaled = scaler_X.transform(X_input)  # Shape: (12, 7)
        
        # Reshape para el modelo: (1, 12, 7)
        X_model = X_scaled.reshape(1, 12, 7)
        
        # Predicción
        y_pred_scaled = modelo.predict(X_model, verbose=0)
        
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


if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )