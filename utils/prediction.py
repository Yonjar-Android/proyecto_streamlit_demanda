"""
Módulo para realizar predicciones con el modelo XGBoost.
Incluye validación de entrada y manejo de errores.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


def realizar_prediccion(modelo: Any, caracteristicas: Dict[str, float]) -> Tuple[float, bool]:
    """
    Realiza una predicción individual con el modelo.
    
    Args:
        modelo: Modelo XGBoost Regressor cargado.
        caracteristicas: Diccionario con las características para la predicción.
        
    Returns:
        Tupla (predicción, éxito). La predicción es 0 si falló.
    """
    if modelo is None:
        return 0.0, False
    
    try:
        # Convertir características a DataFrame si es necesario
        if isinstance(caracteristicas, dict):
            df = pd.DataFrame([caracteristicas])

            # Reordenar columnas exactamente como el modelo espera
            if hasattr(modelo, "feature_names_in_"):
                df = df.reindex(
                columns=modelo.feature_names_in_,
                fill_value=0
        )
        else:
            df = caracteristicas
        
        # Realizar predicción
        prediccion = modelo.predict(df)[0]
        
        # Asegurar que la predicción sea positiva (no puede haber demanda negativa)
        prediccion = max(0, float(prediccion))
        
        return prediccion, True
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return 0.0, False


def realizar_predicciones_batch(modelo: Any, datos: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza predicciones para múltiples registros.
    
    Args:
        modelo: Modelo XGBoost Regressor cargado.
        datos: DataFrame con los datos para predicción.
        
    Returns:
        DataFrame con predicciones añadidas.
    """
    if modelo is None:
        return datos.copy()
    
    try:
        predicciones = modelo.predict(datos)
        # Asegurar predicciones positivas
        predicciones = np.maximum(predicciones, 0)
        datos_resultado = datos.copy()
        datos_resultado['prediccion_demanda'] = predicciones
        return datos_resultado
    except Exception as e:
        print(f"Error en predicciones batch: {str(e)}")
        return datos.copy()


def calcular_stock_recomendado(demanda_predicha: float, factor_seguridad: float = 1.2) -> float:
    """
    Calcula el stock recomendado basado en la demanda predicha.
    
    Args:
        demanda_predicha: Demanda predicha por el modelo.
        factor_seguridad: Factor de seguridad (default 1.2 = 20% extra).
        
    Returns:
        Stock recomendado.
    """
    return demanda_predicha * factor_seguridad


def calcular_diferencia_stock(stock_actual: float, stock_recomendado: float) -> Tuple[float, str]:
    """
    Calcula la diferencia entre stock actual y recomendado.
    
    Args:
        stock_actual: Stock actual.
        stock_recomendado: Stock recomendado.
        
    Returns:
        Tupla (diferencia, estado).
    """
    diferencia = stock_recomendado - stock_actual
    
    if diferencia > 0:
        estado = "🔴 FALTA STOCK"
    elif diferencia < -0.1 * stock_recomendado:
        estado = "🟠 EXCESO DE STOCK"
    else:
        estado = "🟢 ÓPTIMO"
    
    return diferencia, estado


def obtener_explicacion_prediccion(demanda: float, stock_actual: float, diferencia: float) -> str:
    """
    Genera una explicación textual de la predicción.
    
    Args:
        demanda: Demanda predicha.
        stock_actual: Stock actual.
        diferencia: Diferencia entre stock recomendado y actual.
        
    Returns:
        Texto explicativo.
    """
    explicacion = f"""
    **Análisis de la Predicción:**
    
    - **Demanda Estimada:** {demanda:.2f} unidades
    - **Stock Actual:** {stock_actual:.2f} unidades
    - **Diferencia:** {diferencia:.2f} unidades
    
    """
    
    if diferencia > 0:
        explicacion += f"⚠️ Se requieren **{diferencia:.2f} unidades más** para satisfacer la demanda proyectada."
    elif diferencia < 0:
        explicacion += f"✓ Hay **{abs(diferencia):.2f} unidades de exceso** de stock que podrían optimizarse."
    else:
        explicacion += "✓ El stock actual es **óptimo** para la demanda proyectada."
    
    return explicacion
