"""
Página 2: Planeación de Inventario
Visualiza la demanda proyectada durante un año completo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.load_model import cargar_modelo
from utils.prediction import realizar_prediccion

# Configuración de página
st.set_page_config(
    page_title="Proyección de Demanda Anual",
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
st.title("📊 Proyección de Demanda Anual")

st.markdown("""
Genera una proyección de demanda para todo un año. 
Selecciona el producto y parámetros clave para anticipar la demanda futura y optimizar tu planeación.
""")

# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL ESCENARIO
# ============================================================================

st.subheader("⚙️ Configuración del Escenario")

col1, col2, col3, col4 = st.columns(4)

with col1:
    id_producto = st.selectbox(
    "ID Producto",
    list(range(1, 100))
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
    año_proyeccion = st.selectbox(
        "Año de Proyección",
        [2027, 2028],
        help="Año para la proyección de demanda",
        key="año"
    )

# Botón principal
col_button = st.columns([2, 1, 2])
with col_button[1]:
    generar_boton = st.button(
        "🔮 Generar Proyección Anual",
        use_container_width=True,
        type="primary",
        key="generar"
    )

# ============================================================================
# PROCESAMIENTO Y GENERACIÓN DE PREDICCIONES
# ============================================================================

if generar_boton:
    with st.spinner("Generando proyecciones para los 12 meses..."):
        # Meses del año
        meses = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ]
        
        predicciones = []
        
        # Generar predicción para cada mes
        for mes_num in range(1, 13):
            # Calcular variables automáticamente
            trimestre = (mes_num - 1) // 3 + 1
            dia_semana = ((mes_num * 3) % 7) + 1  # Varía según el mes de manera representativa
            hora = 12  # Hora promedio del día
            
            # Calcular monto de descuento a partir del porcentaje
            monto_descuento = (precio_unitario * porcentaje_descuento / 100)
            
            # Preparar datos para predicción
            datos_prediccion = {
            'precio_unitario': precio_unitario,
            'descuento_porcentaje': porcentaje_descuento,
            'monto_descuento': monto_descuento,
        'mes': mes_num,
        'trimestre': trimestre,
        'año': año_proyeccion,
        'sucursal_Secundaria': 0,
        'sucursal_Terciaria': 0,
            }
            
            # Crear todas las columnas One-Hot de producto
            for i in range(2, 100):
                datos_prediccion[f'id_producto_{i}'] = 0

            # Activar únicamente el producto seleccionado
            if id_producto != 1:
                datos_prediccion[f'id_producto_{id_producto}'] = 1
            
            # Realizar predicción
            demanda_predicha, exito = realizar_prediccion(modelo, datos_prediccion)
            
            if exito:
                predicciones.append({
                    'mes_num': mes_num,
                    'mes': meses[mes_num - 1],
                    'demanda': max(0, demanda_predicha)  # Asegurar valores positivos
                })
            else:
                st.error(f"Error al predecir para {meses[mes_num - 1]}")
                st.stop()
        
        # Convertir a DataFrame
        df_predicciones = pd.DataFrame(predicciones)
        
        # Calcular KPIs
        demanda_total = df_predicciones['demanda'].sum()
        promedio_mensual = df_predicciones['demanda'].mean()
        idx_max = df_predicciones['demanda'].idxmax()
        idx_min = df_predicciones['demanda'].idxmin()
        mes_mayor_demanda = df_predicciones.loc[idx_max]
        mes_menor_demanda = df_predicciones.loc[idx_min]
        
        # ====================================================================
        # SECCIÓN 2: KPIs
        # ====================================================================
        
        st.markdown("---")
        st.subheader("📈 Indicadores Clave")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric(
                label="Demanda Total Anual",
                value=f"{demanda_total:,.0f}",
                help="Suma de demanda proyectada para los 12 meses"
            )
        
        with kpi2:
            st.metric(
                label="Promedio Mensual",
                value=f"{promedio_mensual:,.0f}",
                help="Promedio de demanda por mes"
            )
        
        with kpi3:
            st.metric(
                label=f"Mayor Demanda",
                value=f"{mes_mayor_demanda['mes']}",
                delta=f"{mes_mayor_demanda['demanda']:,.0f} unidades",
                help="Mes con mayor demanda proyectada"
            )
        
        with kpi4:
            st.metric(
                label=f"Menor Demanda",
                value=f"{mes_menor_demanda['mes']}",
                delta=f"{mes_menor_demanda['demanda']:,.0f} unidades",
                help="Mes con menor demanda proyectada"
            )
        
        # ====================================================================
        # SECCIÓN 3: GRÁFICO DE EVOLUCIÓN MENSUAL
        # ====================================================================
        
        st.markdown("---")
        st.subheader("📊 Proyección de Demanda Anual")
        st.markdown("*Demanda estimada para el año seleccionado*")
        
        # Crear gráfico interactivo
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_predicciones['mes'],
            y=df_predicciones['demanda'],
            mode='lines+markers',
            name='Demanda',
            line=dict(color=COLOR_PRINCIPAL, width=3),
            marker=dict(size=10, color=COLOR_PRINCIPAL),
            hovertemplate='<b>%{x}</b><br>Demanda: %{y:,.0f} unidades<extra></extra>',
            fill='tozeroy',
            fillcolor=f"rgba(224, 71, 97, 0.1)"
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Proyección de Demanda - Año {año_proyeccion}",
                font=dict(size=18, color="#333")
            ),
            xaxis_title="Mes",
            yaxis_title="Unidades Demandadas",
            hovermode='x unified',
            template='plotly_white',
            height=450,
            margin=dict(l=50, r=30, t=80, b=50),
            font=dict(size=11),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ====================================================================
        # SECCIÓN 4: TABLA DE PLANEACIÓN
        # ====================================================================
        
        st.markdown("---")
        st.subheader("📋 Tabla de Planeación")
        
        # Mostrar tabla con opción de ordenamiento
        df_tabla = df_predicciones[['mes', 'demanda']].copy()
        df_tabla.columns = ['Mes', 'Demanda Predicha']
        df_tabla['Demanda Predicha'] = df_tabla['Demanda Predicha'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            df_tabla,
            use_container_width=True,
            hide_index=True
        )
        
        # Botón para descargar
        csv = pd.DataFrame({
            'Mes': df_predicciones['mes'],
            'Demanda Predicha': df_predicciones['demanda'].apply(lambda x: f"{x:.0f}")
        }).to_csv(index=False)
        
        st.download_button(
            label="📥 Descargar Planeación",
            data=csv,
            file_name=f"proyeccion_demanda_{año_proyeccion}.csv",
            mime="text/csv",
            use_container_width=False
        )
        
        # ====================================================================
        # INFORMACIÓN ADICIONAL
        # ====================================================================
        
        st.markdown("---")
        with st.expander("ℹ️ Información de la Proyección"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write("**Parámetros de Entrada:**")
                st.write(f"- ID Producto: {id_producto}")
                st.write(f"- Precio Unitario: C${precio_unitario:.2f}")
                st.write(f"- Porcentaje Descuento: {porcentaje_descuento:.1f}%")
                monto_desc_display = (precio_unitario * porcentaje_descuento / 100)
                st.write(f"- Monto de Descuento: C${monto_desc_display:.2f}")
                st.write(f"- Año de Proyección: {año_proyeccion}")
            
            with col_info2:
                st.write("**Estadísticas de la Proyección:**")
                st.write(f"- Demanda Máxima: {df_predicciones['demanda'].max():,.0f} unidades")
                st.write(f"- Demanda Mínima: {df_predicciones['demanda'].min():,.0f} unidades")
                st.write(f"- Desviación Estándar: {df_predicciones['demanda'].std():,.0f} unidades")
                st.write(f"- Rango: {df_predicciones['demanda'].max() - df_predicciones['demanda'].min():,.0f} unidades")
        
        # Footer
        st.divider()
        st.markdown(f"""
        <div style='text-align: center; color: #999; font-size: 0.85em;'>
            <p>
                <strong style='color: {COLOR_PRINCIPAL}'>Proyección de Demanda Anual</strong> | 
                Modelo de Predicción: XGBoost Regressor | 
                Año: {año_proyeccion}
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Mensaje inicial si no se ha generado proyección
    st.info("💡 **Cómo usar:** Configura los parámetros del escenario (Producto, Precio y Descuento) y selecciona el año para el cual deseas proyectar la demanda. Luego haz clic en 'Generar Proyección Anual' para obtener un análisis completo de demanda para los 12 meses.")

