"""
Aplicación Principal: Dashboard de Predicción de Demanda
Streamlit app con XGBoost Regressor para visualizar y explotar modelo ML.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils.load_model import cargar_modelo, obtener_info_modelo
from utils.charts import COLOR_PRINCIPAL
from utils.charts import COLOR_SECUNDARIO

# Configurar página
st.set_page_config(
    page_title="Predicción de Demanda",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar modelo
modelo = cargar_modelo()

# Estilos personalizados
st.markdown(f"""
<style>
    :root {{
        --color-principal: {COLOR_PRINCIPAL};
        --color-secundario: {COLOR_SECUNDARIO};
    }}
    
    .main-title {{
        color: {COLOR_PRINCIPAL};
        font-weight: bold;
        font-size: 2.5em;
        margin-bottom: 10px;
    }}
    
    .subtitle {{
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }}
    
    .metric-card {{
    background: linear-gradient(
        135deg,
        {COLOR_PRINCIPAL}20 0%,
        {COLOR_PRINCIPAL}05 100%
    );
    padding: 20px;
    border-radius: 10px;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}}

.primary-card {{
    border-left: 5px solid {COLOR_PRINCIPAL};
}}

.secondary-card {{
    border-left: 5px solid {COLOR_SECUNDARIO};
}}

.metric-card h4 {{
    margin-top: 0;
    margin-bottom: 10px;
}}

.metric-card p {{
    margin-bottom: 0;
}}
</style>
""", unsafe_allow_html=True)

# SIDEBAR GLOBAL
with st.sidebar:
    st.title("⚙️ Configuración Global")
    
    st.divider()
    
    # Selectores globales
    st.subheader("🎯 Parámetros")
    
    producto_global = st.selectbox(
        "ID Producto",
        list(range(1, 100)),
        help="Selecciona el ID del producto",
        key="sidebar_producto"
    )
    
    año_global = st.slider(
        "Año",
        min_value=2027,
        max_value=2028,
        value=2027,
        key="sidebar_año"
    )
    
    mes_global = st.selectbox(
        "Mes",
        ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        key="sidebar_mes"
    )
    
    st.divider()
    
    # Información del modelo
    st.subheader("🤖 Información del Modelo")
    
    if modelo is not None:
        info_modelo = obtener_info_modelo(modelo)
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.metric(
                label="Tipo",
                value="XGBoost",
                help="Algoritmo de Machine Learning"
            )
        
        with col_info2:
            st.metric(
                label="Tarea",
                value="Regresión",
                help="Predicción de valores continuos"
            )
        
        if info_modelo:
            st.write(f"**Variables de entrada:** {info_modelo.get('num_features', 'N/A')}")
            st.write(f"**Árboles:** {info_modelo.get('num_estimadores', 'N/A')}")
        
        st.success("✓ Modelo cargado correctamente")
    else:
        st.error("❌ Error al cargar el modelo")
    
    st.divider()

    
    # Footer del sidebar
    st.markdown(f"""
    <div style='text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;'>
        <p style='color: #666; font-size: 0.8em;'>
            <strong>XGBoost Regressor</strong><br>
            Dashboard Analítico<br>
            <strong style='color: {COLOR_PRINCIPAL}'>Predicción de Demanda</strong>
        </p>
        <p style='color: #999; font-size: 0.75em; margin-top: 10px;'>
            v1.0 | {datetime.now().strftime('%Y')}
        </p>
    </div>
    """, unsafe_allow_html=True)

# CONTENIDO PRINCIPAL
st.markdown("""
<div class='main-title'>📊 Dashboard de Predicción de Demanda</div>
<div class='subtitle'>XGBoost Regressor | Análisis y Exploración Interactiva</div>
""", unsafe_allow_html=True)

# Descripción general
col_desc1, col_desc2 = st.columns([2, 1])

with col_desc1:
    st.markdown("""
    
    Esta aplicación integra un modelo de Machine Learning entrenado con **XGBoost Regressor**
    para predecir la demanda futura de productos.
    
    **Características principales:**
    - 🔮 Predicciones individuales puntuales
    - 📅 Planeación anual de inventario
    - 🎲 Simulación What-If de escenarios
    - 🔬 Análisis detallado del modelo
    - 📦 Predicción masiva por lote
    """)

with col_desc2:
    st.markdown(f"""
    
    **Estado del Modelo:**
    - ✓ Activo
    - ✓ Cargado
    - ✓ Operacional
    """)

st.divider()

# Sección de módulos
st.markdown("### 📑 Módulos Disponibles")

col_mod1, col_mod2 = st.columns(2)

with col_mod1:
    st.markdown("""
    <div class='metric-card primary-card'>
        <h4>🔮 Predicción Individual</h4>
        <p>Realiza predicciones puntuales personalizadas con análisis detallado y recomendaciones de stock.</p>
    </div>
    """, unsafe_allow_html=True)

with col_mod2:
    st.markdown("""
    <div class='metric-card secondary-card'>
        <h4>📅 Planeación Anual</h4>
        <p>Visualiza la demanda proyectada para cada mes con análisis automático de estacionalidad.</p>
    </div>
    """, unsafe_allow_html=True)



st.space(size="small")

col_mod3, col_mod4 = st.columns(2)

with col_mod3:
    st.markdown("""
    <div class='metric-card primary-card'>
        <h4>🎲 What-If Analysis</h4>
        <p>Simula escenarios modificando variables para optimizar decisiones comerciales.</p>
    </div>
    """, unsafe_allow_html=True)

with col_mod4:
    st.markdown("""
    <div class='metric-card secondary-card'>
        <h4>🔬 Análisis del Modelo</h4>
        <p>Evalúa el rendimiento con métricas completas, importancia de variables y residuales.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Indicadores técnicos
col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)

with col_tech1:
    st.metric("Modelo", "XGBoost", help="Algoritmo de predicción")

with col_tech2:
    st.metric("Estado", "✓ Activo", help="Sistema operacional")

with col_tech3:
    st.metric("Módulos", "5", help="Funcionalidades disponibles")

with col_tech4:
    st.metric("Versión", "1.0", help="Versión de la aplicación")

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 1.2rem; padding: 20px 0;'>
    <p style='color: {COLOR_PRINCIPAL}; font-size: 0.8em; margin-top: 10px;'>
        <strong>Selecciona un módulo en el menú izquierdo para comenzar</strong>
    </p>
</div>
""", unsafe_allow_html=True)
