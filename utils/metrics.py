"""
Módulo para calcular métricas de evaluación del modelo.
Incluye R², MAE, RMSE, MAPE y otras métricas de regresión.
"""

import numpy as np
from typing import Tuple, Dict


def calcular_r_squared(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula R² (coeficiente de determinación).
    
    Args:
        y_real: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        Valor de R² entre 0 y 1.
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    return float(r_squared)


def calcular_mae(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula MAE (Mean Absolute Error).
    
    Args:
        y_real: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        MAE.
    """
    return float(np.mean(np.abs(y_real - y_pred)))


def calcular_rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula RMSE (Root Mean Squared Error).
    
    Args:
        y_real: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        RMSE.
    """
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def calcular_mape(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula MAPE (Mean Absolute Percentage Error).
    
    Args:
        y_real: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        MAPE (como porcentaje).
    """
    mask = y_real != 0
    if not np.any(mask):
        return 0.0
    
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    return float(mape)


def calcular_metricas_completas(y_real: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_real: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        Diccionario con todas las métricas.
    """
    return {
        'R²': calcular_r_squared(y_real, y_pred),
        'MAE': calcular_mae(y_real, y_pred),
        'RMSE': calcular_rmse(y_real, y_pred),
        'MAPE': calcular_mape(y_real, y_pred)
    }


def obtener_explicacion_metricas() -> str:
    """
    Retorna una explicación de las métricas del modelo.
    
    Returns:
        Texto con explicación.
    """
    return """
    ### Explicación de Métricas del Modelo XGBoost
    
    #### **R² (Coeficiente de Determinación)**
    - Rango: 0 a 1 (idealmente cercano a 1)
    - Interpre: Qué porcentaje de la varianza en los datos reales es explicada por el modelo
    - Ejemplo: R² = 0.85 significa que el modelo explica el 85% de la variabilidad
    
    #### **MAE (Error Absoluto Medio)**
    - Unidades: Mismas que la variable objetivo (en este caso, unidades de demanda)
    - Interpretar: Desviación promedio entre predicciones y valores reales
    - Ventaja: Fácil de interpretar y menos sensible a outliers
    - Ejemplo: MAE = 50 significa que en promedio, la predicción se desvía 50 unidades
    
    #### **RMSE (Error Cuadrático Medio)**
    - Unidades: Mismas que la variable objetivo
    - Interpretar: Desviación promedio ponderada (penaliza errores grandes más)
    - Ventaja: Penaliza errores grandes, útil cuando estos son críticos
    - Ejemplo: RMSE = 75 significa que hay errores significativos en algunas predicciones
    
    #### **MAPE (Error Porcentual Absoluto Medio)**
    - Rango: Porcentaje (%)
    - Interpretar: Error promedio relativo al valor real
    - Ventaja: Independiente de la escala, comparable entre variables
    - Ejemplo: MAPE = 15% significa que las predicciones se desvían 15% en promedio
    
    ---
    
    ### Importancia de Variables en XGBoost
    
    **¿Qué mide?**
    - La importancia de cada variable refleja cuánto contribuye a reducir el error en las predicciones
    - XGBoost calcula esto basándose en cuántas veces se usa cada variable y qué mejora produce
    
    **¿Por qué importa?**
    - Identifica qué variables son más relevantes para predecir la demanda
    - Ayuda a optimizar la recolección de datos (enfocarse en variables importantes)
    - Permite entender mejor los factores que impulsan la demanda del producto
    
    **Interpretación:**
    - Variables con importancia alta: Son decisivas en el modelo
    - Variables con importancia baja: Podrían removerse sin afectar mucho el rendimiento
    """


def obtener_interpretacion_residuales() -> str:
    """
    Retorna interpretación de residuales.
    
    Returns:
        Texto con explicación.
    """
    return """
    ### Análisis de Residuales
    
    **¿Qué son los residuales?**
    - Son las diferencias entre los valores reales y los predichos
    - Residual = Valor Real - Valor Predicho
    
    **¿Qué observar?**
    - **Distribución aleatoria alrededor de cero**: Indica buen ajuste del modelo
    - **Patrones o tendencias**: Sugiere que el modelo no captura alguna relación
    - **Outliers**: Casos donde el modelo falla significativamente
    - **Varianza constante**: Idealmente, el error no debe aumentar con los valores predichos
    
    **Interpretación en el contexto de demanda:**
    - Si hay residuales grandes negativos: El modelo sobrestima la demanda en algunos casos
    - Si hay residuales grandes positivos: El modelo subestima la demanda en algunos casos
    - Residuales consistentemente distribuidos: El modelo es confiable
    """
