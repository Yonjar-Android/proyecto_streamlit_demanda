"""
Módulo para cargar el modelo XGBoost Regressor.
Implementa caching de Streamlit para optimizar el rendimiento.
"""

import joblib
import streamlit as st
from typing import Any


@st.cache_resource
def cargar_modelo() -> Any:
    """
    Carga el modelo XGBoost Regressor desde el archivo pkl.
    
    Returns:
        Modelo XGBoost Regressor cargado.
        
    Raises:
        FileNotFoundError: Si el archivo del modelo no existe.
    """
    try:
        modelo = joblib.load("modelo_predictor_ventas2.pkl")
        return modelo
    except FileNotFoundError:
        st.error("❌ Error: El archivo 'modelo_predictor_ventas.pkl' no se encontró.")
        st.info("Por favor, asegúrese de que el archivo está en el directorio raíz del proyecto.")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None


def obtener_info_modelo(modelo: Any) -> dict:
    """
    Extrae información del modelo XGBoost.
    
    Args:
        modelo: Modelo XGBoost Regressor.
        
    Returns:
        Diccionario con información del modelo.
    """
    if modelo is None:
        return {}

    
    info = {
        "tipo": "XGBoost Regressor",
        "num_features": modelo.n_features_in_ if hasattr(modelo, 'n_features_in_') else "N/A",
        "num_estimadores": modelo.n_estimators if hasattr(modelo, 'n_estimators') else "N/A",
    }
    
    return info
