"""
Página 3: Simulación What-If
Permite comparar dos escenarios hipotéticos y observar el impacto en la demanda.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.load_model import cargar_modelo
from utils.prediction import realizar_prediccion

# Configuración de página
st.set_page_config(
    page_title="Simulación What-If",
    page_icon="🎲",
    layout="wide"
)

# Color principal
COLOR_PRINCIPAL = "#E04761"
COLOR_SECUNDARIO = "#2E86AB"

# Cargar modelo
modelo = cargar_modelo()

if modelo is None:
    st.error("No se pudo cargar el modelo. Por favor, revise los logs.")
    st.stop()

st.title("🎲 Simulación What-If")

st.markdown("""
Compara dos escenarios hipotéticos para un producto específico.
Observa cómo cambios en precio y descuento impactan la demanda estimada.
""")

# ============================================================================
# SECCIÓN 1: PARÁMETROS GLOBALES
# ============================================================================

st.subheader("⚙️ Parámetros Globales")

col1, col2, col3 = st.columns(3)

with col1:
    id_producto = st.selectbox(
        "ID Producto",
        list(range(1, 100)),
        help="Selecciona el ID del producto a simular",
        key="producto_whatif"
    )

with col2:
    mes = st.selectbox(
        "Mes",
        ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        help="Mes para la simulación",
        key="mes_whatif"
    )

with col3:
    año = st.number_input(
        "Año",
        min_value=2027,
        max_value=2035,
        value=2027,
        step=1,
        help="Año para la simulación",
        key="año_whatif"
    )

# Parámetros automáticos
hora = 12  # Hora promedio del día
dia_semana_num = 0  # Lunes por defecto

# Mapear día de semana (solo para referencia)
dias_semana_map = {
    "Lunes": 0,
    "Martes": 1,
    "Miércoles": 2,
    "Jueves": 3,
    "Viernes": 4,
    "Sábado": 5,
    "Domingo": 6
}

# Calcular trimestre y mes_num
mes_map = {
    "Enero": 1, "Febrero": 2, "Marzo": 3,
    "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9,
    "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
mes_num = mes_map[mes]
trimestre = (mes_num - 1) // 3 + 1

# ============================================================================
# SECCIÓN 2: COMPARACIÓN DE ESCENARIOS
# ============================================================================

st.markdown("---")
st.subheader("⚖️ Comparación de Escenarios")

col_esc_a, col_esc_b = st.columns(2)

# ============================================================================
# ESCENARIO A
# ============================================================================

with col_esc_a:
    st.markdown("### 📌 Escenario A")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        precio_a = st.slider(
            "Precio Unitario (C$)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            key="precio_a_whatif"
        )
    
    with col_a2:
        descuento_a = st.slider(
            "Porcentaje Descuento (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="descuento_a_whatif"
        )
    
    # Calcular monto descuento automáticamente
    monto_descuento_a = (precio_a * descuento_a / 100)

# ============================================================================
# ESCENARIO B
# ============================================================================

with col_esc_b:
    st.markdown("### 📌 Escenario B")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        precio_b = st.slider(
            "Precio Unitario (C$)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            key="precio_b_whatif"
        )
    
    with col_b2:
        descuento_b = st.slider(
            "Porcentaje Descuento (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            key="descuento_b_whatif"
        )
    
    # Calcular monto descuento automáticamente
    monto_descuento_b = (precio_b * descuento_b / 100)

# ============================================================================
# FUNCIÓN AUXILIAR: CONSTRUIR DATOS DE PREDICCIÓN
# ============================================================================

def construir_datos_prediccion(id_prod, precio, descuento_pct, monto_descuento, mes_n, trim, año_n, hora_n, dia_sem_n):
    """
    Construye el diccionario de datos para la predicción con One-Hot Encoding correcto.
    """
    datos = {
        'precio_unitario': precio,
        'descuento_porcentaje': descuento_pct,
        'monto_descuento': monto_descuento,
        'mes': mes_n,
        'trimestre': trim,
        'año': año_n,
        'dia_semana': dia_sem_n,
        'hora': hora_n,
        'sucursal_Secundaria': 0,
        'sucursal_Terciaria': 0,
        'categoria_producto_Electronica': 0,
        'categoria_producto_Hogar': 0,
        'categoria_producto_Moda': 0,
    }
    
    # Crear todas las columnas One-Hot de producto (id_producto_2 a id_producto_99)
    for i in range(2, 100):
        datos[f'id_producto_{i}'] = 0
    
    # Activar únicamente el producto seleccionado (si no es 1)
    if id_prod != 1:
        datos[f'id_producto_{id_prod}'] = 1
    
    return datos

# ============================================================================
# CALCULAR PREDICCIONES PARA AMBOS ESCENARIOS
# ============================================================================

with st.spinner("Calculando predicciones para ambos escenarios..."):
    datos_a = construir_datos_prediccion(
        id_producto, precio_a, descuento_a, monto_descuento_a,
        mes_num, trimestre, año, hora, dia_semana_num
    )
    
    datos_b = construir_datos_prediccion(
        id_producto, precio_b, descuento_b, monto_descuento_b,
        mes_num, trimestre, año, hora, dia_semana_num
    )
    
    demanda_a, exito_a = realizar_prediccion(modelo, datos_a)
    demanda_b, exito_b = realizar_prediccion(modelo, datos_b)

if not (exito_a and exito_b):
    st.error("Error al calcular predicciones. Verifica los datos ingresados.")
    st.stop()

# ============================================================================
# SECCIÓN 3: RESULTADOS DE LA SIMULACIÓN
# ============================================================================

st.markdown("---")
st.subheader("📊 Resultados de la Simulación")

res_col_a, res_col_b, res_col_comp = st.columns(3)

with res_col_a:
    st.metric(
        label="Demanda Escenario A",
        value=f"{demanda_a:,.0f}",
        delta="unidades",
        help="Demanda predicha para el Escenario A"
    )

with res_col_b:
    st.metric(
        label="Demanda Escenario B",
        value=f"{demanda_b:,.0f}",
        delta="unidades",
        help="Demanda predicha para el Escenario B"
    )

with res_col_comp:
    variacion_absoluta = demanda_b - demanda_a
    variacion_porcentual = (variacion_absoluta / demanda_a * 100) if demanda_a > 0 else 0
    
    st.metric(
        label="Variación (B vs A)",
        value=f"{variacion_porcentual:+.1f}%",
        delta=f"{variacion_absoluta:+.0f} unidades",
        help="Cambio en demanda entre escenarios"
    )

# ============================================================================
# SECCIÓN 4: GRÁFICO COMPARATIVO
# ============================================================================

st.markdown("---")
st.subheader("📈 Comparación de Demanda entre Escenarios")

fig_comparativa = go.Figure(data=[
    go.Bar(
        name='Escenario A',
        x=['Demanda'],
        y=[demanda_a],
        marker_color=COLOR_SECUNDARIO,
        text=f'{demanda_a:,.0f}',
        textposition='outside',
        hovertemplate='<b>Escenario A</b><br>Demanda: %{y:,.0f} unidades<extra></extra>'
    ),
    go.Bar(
        name='Escenario B',
        x=['Demanda'],
        y=[demanda_b],
        marker_color=COLOR_PRINCIPAL,
        text=f'{demanda_b:,.0f}',
        textposition='outside',
        hovertemplate='<b>Escenario B</b><br>Demanda: %{y:,.0f} unidades<extra></extra>'
    )
])

fig_comparativa.update_layout(
    barmode='group',
    title=f'Comparación de Demanda - {mes} {año}',
    yaxis_title='Unidades Demandadas',
    xaxis_title='',
    height=400,
    showlegend=True,
    template='plotly_white'
)

st.plotly_chart(fig_comparativa, use_container_width=True)

# ============================================================================
# SECCIÓN 5: ANÁLISIS DE SENSIBILIDAD
# ============================================================================

st.markdown("---")
st.subheader("🔍 Análisis de Sensibilidad")

col_analisis_precio, col_analisis_descuento = st.columns(2)

# ============================================================================
# ANÁLISIS: IMPACTO DEL PRECIO EN DEMANDA
# ============================================================================

with col_analisis_precio:
    st.markdown("#### Impacto del Precio en Demanda")
    
    precios_test = np.linspace(0, 200, 15)
    demandas_precio = []
    
    for p in precios_test:
        datos_test = construir_datos_prediccion(
            id_producto, p, 0.0, 0.0,
            mes_num, trimestre, año, hora, dia_semana_num
        )
        dem, _ = realizar_prediccion(modelo, datos_test)
        demandas_precio.append(dem)
    
    fig_precio = go.Figure()
    fig_precio.add_trace(go.Scatter(
        x=precios_test,
        y=demandas_precio,
        mode='lines+markers',
        name='Demanda vs Precio',
        line=dict(color=COLOR_PRINCIPAL, width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(224, 71, 97, 0.1)',
        hovertemplate='<b>Precio: C$%{x:.0f}</b><br>Demanda: %{y:,.0f} unidades<extra></extra>'
    ))
    
    fig_precio.update_layout(
        title='Relación: Precio vs Demanda',
        xaxis_title='Precio Unitario (C$)',
        yaxis_title='Demanda Predicha (unidades)',
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_precio, use_container_width=True)

# ============================================================================
# ANÁLISIS: IMPACTO DEL DESCUENTO EN DEMANDA
# ============================================================================

with col_analisis_descuento:
    st.markdown("#### Impacto del Descuento en Demanda")
    
    descuentos_test = np.linspace(0, 100, 15)
    demandas_descuento = []
    precio_base_sensibilidad = 50.0
    
    for d in descuentos_test:
        monto_desc_test = (precio_base_sensibilidad * d / 100)
        datos_test = construir_datos_prediccion(
            id_producto, precio_base_sensibilidad, d, monto_desc_test,
            mes_num, trimestre, año, hora, dia_semana_num
        )
        dem, _ = realizar_prediccion(modelo, datos_test)
        demandas_descuento.append(dem)
    
    fig_descuento = go.Figure()
    fig_descuento.add_trace(go.Scatter(
        x=descuentos_test,
        y=demandas_descuento,
        mode='lines+markers',
        name='Demanda vs Descuento',
        line=dict(color=COLOR_SECUNDARIO, width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.1)',
        hovertemplate='<b>Descuento: %{x:.0f}%</b><br>Demanda: %{y:,.0f} unidades<extra></extra>'
    ))
    
    fig_descuento.update_layout(
        title='Relación: Descuento vs Demanda',
        xaxis_title='Porcentaje de Descuento (%)',
        yaxis_title='Demanda Predicha (unidades)',
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_descuento, use_container_width=True)

# ============================================================================
# SECCIÓN 6: TABLA COMPARATIVA DETALLADA
# ============================================================================

st.markdown("---")
st.subheader("📋 Análisis Detallado de Escenarios")

col_tabla_a, col_tabla_b = st.columns(2)

with col_tabla_a:
    st.markdown("### Escenario A")
    df_a = pd.DataFrame({
        'Parámetro': [
            'Precio Unitario',
            'Porcentaje Descuento',
            'Monto Descuento',
            'Demanda Predicha'
        ],
        'Valor': [
            f'C${precio_a:.2f}',
            f'{descuento_a:.1f}%',
            f'C${monto_descuento_a:.2f}',
            f'{demanda_a:,.0f} unidades'
        ]
    })
    st.dataframe(df_a, use_container_width=True, hide_index=True)

with col_tabla_b:
    st.markdown("### Escenario B")
    df_b = pd.DataFrame({
        'Parámetro': [
            'Precio Unitario',
            'Porcentaje Descuento',
            'Monto Descuento',
            'Demanda Predicha'
        ],
        'Valor': [
            f'C${precio_b:.2f}',
            f'{descuento_b:.1f}%',
            f'C${monto_descuento_b:.2f}',
            f'{demanda_b:,.0f} unidades'
        ]
    })
    st.dataframe(df_b, use_container_width=True, hide_index=True)

# ============================================================================
# SECCIÓN 7: RECOMENDACIONES
# ============================================================================

st.markdown("---")
st.subheader("💡 Recomendaciones")

if demanda_b > demanda_a * 1.1:
    st.success(f"""
    **✓ El Escenario B es más atractivo**
    
    Aumenta la demanda en **{variacion_porcentual:.1f}%** respecto a Escenario A 
    ({variacion_absoluta:+.0f} unidades adicionales).
    
    Considera aplicar esta estrategia de precio y descuento para incrementar ventas.
    """)
elif demanda_b < demanda_a * 0.9:
    st.warning(f"""
    **⚠️ El Escenario A es mejor**
    
    Genera **{abs(variacion_porcentual):.1f}%** más demanda que Escenario B 
    ({abs(variacion_absoluta):.0f} unidades adicionales).
    
    Recomendación: Mantener la estrategia del Escenario A para maximizar demanda.
    """)
else:
    st.info(f"""
    **📊 Escenarios similares**
    
    La diferencia es minimal (**{abs(variacion_porcentual):.1f}%**).
    Ambas estrategias tienen impacto similar en la demanda.
    """)

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #999; font-size: 0.85em;'>
    <p>
        <strong style='color: {COLOR_PRINCIPAL}'>Simulación What-If</strong> | 
        Modelo de Predicción: XGBoost Regressor | 
        Análisis de Sensibilidad
    </p>
</div>
""", unsafe_allow_html=True)
