"""
Página 1: Predicción Individual de Demanda
Predice la demanda de un producto específico para un mes y año seleccionados.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.load_model import cargar_modelo
from utils.prediction import realizar_prediccion

# Configuración de página
st.set_page_config(
    page_title="Predicción Individual de Demanda",
    page_icon="📊",
    layout="wide"
)

# Color principal
COLOR_PRINCIPAL = "#E04761"

# Cargar modelo
modelo = cargar_modelo()

if modelo is None:
    st.error("No se pudo cargar el modelo. Por favor, revise los logs.")
    st.stop()

# Título principal
st.title("📊 Predicción Individual de Demanda")

st.markdown("""
Realiza una predicción de demanda para un producto específico en un mes y año determinados.
Ingresa los parámetros clave y obtén la estimación de demanda en tiempo real.
""")

# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL ESCENARIO
# ============================================================================

st.subheader("⚙️ Configuración del Escenario")

col1, col2, col3, col4 = st.columns(4)

with col1:
    id_producto = st.selectbox(
        "ID Producto",
        list(range(1, 100)),
        help="Selecciona el ID del producto",
        key="id_producto"
    )

with col2:
    precio_unitario = st.number_input(
        "Precio Unitario (C$)",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="Precio unitario del producto"
    )

with col3:
    porcentaje_descuento = st.number_input(
        "Porcentaje Descuento (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        help="Porcentaje de descuento aplicado al producto"
    )

with col4:
    mes = st.selectbox(
        "Mes",
        ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        help="Mes para la predicción",
        key="mes"
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    año = st.number_input(
        "Año",
        min_value=2027,
        max_value=2028,
        value=2027,
        step=1,
        help="Año para la predicción",
        key="año"
    )

with col6:
    st.empty()

with col7:
    st.empty()

with col8:
    st.empty()

# Calcular trimestre y monto de descuento
mes_map = {
    "Enero": 1, "Febrero": 2, "Marzo": 3,
    "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9,
    "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
mes_num = mes_map[mes]
trimestre = (mes_num - 1) // 3 + 1
monto_descuento = (precio_unitario * porcentaje_descuento / 100)

# Variables automáticas
hora = 12  # Hora promedio del día
dia_semana_num = 0  # Lunes por defecto

# Botón principal
col_button = st.columns([2, 1, 2])
with col_button[1]:
    predecir_boton = st.button(
        "🔮 Predecir Demanda",
        use_container_width=True,
        type="primary",
        key="predecir"
    )

# ============================================================================
# PROCESAMIENTO Y GENERACIÓN DE PREDICCIÓN
# ============================================================================

if predecir_boton:
    with st.spinner("Calculando predicción..."):
        # Preparar datos para predicción
        datos_prediccion = {
            'precio_unitario': precio_unitario,
            'descuento_porcentaje': porcentaje_descuento,
            'monto_descuento': monto_descuento,
            'mes': mes_num,
            'trimestre': trimestre,
            'año': año,
            'dia_semana': dia_semana_num,
            'hora': hora,
            'sucursal_Secundaria': 0,
            'sucursal_Terciaria': 0,
            'categoria_producto_Electronica': 0,
            'categoria_producto_Hogar': 0,
            'categoria_producto_Moda': 0,
        }
        
        # Crear todas las columnas One-Hot de producto (id_producto_2 a id_producto_99)
        for i in range(2, 100):
            datos_prediccion[f'id_producto_{i}'] = 0
        
        # Activar únicamente el producto seleccionado (si no es 1)
        if id_producto != 1:
            datos_prediccion[f'id_producto_{id_producto}'] = 1
        
        # Realizar predicción
        demanda_predicha, exito = realizar_prediccion(modelo, datos_prediccion)
        
        if exito:
            # ====================================================================
            # SECCIÓN 2: RESULTADO DE LA PREDICCIÓN
            # ====================================================================
            
            st.markdown("---")
            st.subheader("📈 Resultado de la Predicción")
            
            # Mostrar métrica principal
            col_metrica = st.columns([2, 1, 2])
            with col_metrica[1]:
                st.metric(
                    label="Demanda Estimada",
                    value=f"{demanda_predicha:,.0f}",
                    delta="unidades",
                    help="Demanda predicha para el período seleccionado"
                )
            
            # ====================================================================
            # SECCIÓN 3: DETALLES DE LA PREDICCIÓN
            # ====================================================================
            
            st.markdown("---")
            st.subheader("📋 Detalles de la Predicción")
            
            col_param1, col_param2, col_param3 = st.columns(3)
            
            with col_param1:
                st.write("**Producto y Período:**")
                st.write(f"- ID Producto: {id_producto}")
                st.write(f"- Mes: {mes}")
                st.write(f"- Año: {año}")
            
            with col_param2:
                st.write("**Información Temporal:**")
                st.write(f"- Trimestre: {trimestre}")
            
            with col_param3:
                st.write("**Datos Económicos:**")
                st.write(f"- Precio Unitario: C${precio_unitario:.2f}")
                st.write(f"- Descuento: {porcentaje_descuento:.1f}%")
                st.write(f"- Monto Descuento: C${monto_descuento:.2f}")
            
            # Footer
            st.divider()
            st.markdown(f"""
            <div style='text-align: center; color: #999; font-size: 0.85em;'>
                <p>
                    <strong style='color: {COLOR_PRINCIPAL}'>Predicción Individual de Demanda</strong> | 
                    Modelo de Predicción: XGBoost Regressor | 
                    {mes} {año}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.error("❌ Error al realizar la predicción. Verifica los datos ingresados.")

else:
    # Mensaje inicial si no se ha generado predicción
    st.info("💡 **Cómo usar:** Configura los parámetros del escenario (Producto, Precio, Descuento, Mes y Año) y haz clic en 'Predecir Demanda' para obtener la estimación de demanda para el período seleccionado.")

