"""
Módulo para crear visualizaciones con Plotly.
Incluye gráficos para análisis de demanda, errores y comparativas.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Tuple


# Color principal de la aplicación
COLOR_PRINCIPAL = "#E04761"
COLOR_SECUNDARIO = "#2E86AB"
COLOR_EXITO = "#06A77D"
COLOR_ADVERTENCIA = "#F77F00"

def crear_gauge_demanda(demanda: float, demanda_min: float = 0, 
                        demanda_max: float = 1000) -> go.Figure:
    """
    Crea un gráfico gauge para mostrar la demanda predicha.
    
    Args:
        demanda: Valor de demanda predicha.
        demanda_min: Mínimo del rango.
        demanda_max: Máximo del rango.
        
    Returns:
        Figura Plotly.
    """
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number",
        value=demanda,
        title={'text': "Demanda Estimada (unidades)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [demanda_min, demanda_max]},
            'bar': {'color': COLOR_PRINCIPAL},
            'steps': [
                {'range': [demanda_min, demanda_max * 0.33], 'color': "#E8F4F8"},
                {'range': [demanda_max * 0.33, demanda_max * 0.66], 'color': "#D0E8F0"},
                {'range': [demanda_max * 0.66, demanda_max], 'color': "#B8DCE8"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': demanda_max * 0.8
            }
        }
    )])
    
    fig.update_layout(
        font={'size': 12},
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def crear_comparativa_stock(stock_actual: float, stock_recomendado: float) -> go.Figure:
    """
    Crea una gráfico de barras comparativo entre stock actual y recomendado.
    
    Args:
        stock_actual: Stock actual.
        stock_recomendado: Stock recomendado.
        
    Returns:
        Figura Plotly.
    """
    fig = go.Figure(data=[
        go.Bar(
            x=['Stock Actual', 'Stock Recomendado'],
            y=[stock_actual, stock_recomendado],
            marker_color=[COLOR_SECUNDARIO, COLOR_PRINCIPAL],
            text=[f'{stock_actual:.0f}', f'{stock_recomendado:.0f}'],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Comparativa de Stock',
        xaxis_title='Tipo de Stock',
        yaxis_title='Unidades',
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def crear_serie_temporal(df: pd.DataFrame, columna_tiempo: str = 'mes',
                        columna_valor: str = 'demanda') -> go.Figure:
    """
    Crea una gráfico de línea para serie temporal.
    
    Args:
        df: DataFrame con datos.
        columna_tiempo: Nombre de columna con tiempo.
        columna_valor: Nombre de columna con valores.
        
    Returns:
        Figura Plotly.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[columna_tiempo],
        y=df[columna_valor],
        mode='lines+markers',
        name='Demanda Predicha',
        line=dict(color=COLOR_PRINCIPAL, width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(224, 71, 97, 0.2)'
    ))
    
    fig.update_layout(
        title='Demanda Proyectada - Serie Temporal',
        xaxis_title='Período',
        yaxis_title='Demanda (unidades)',
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def crear_scatter_prediccion_vs_real(y_real: np.ndarray, 
                                     y_pred: np.ndarray) -> go.Figure:
    """
    Crea scatter plot de predicción vs valores reales.
    
    Args:
        y_real: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        Figura Plotly.
    """
    # Línea ideal (y = x)
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_real,
        y=y_pred,
        mode='markers',
        name='Predicciones',
        marker=dict(
            size=8,
            color=COLOR_PRINCIPAL,
            opacity=0.6,
            line=dict(width=1, color='white')
        )
    ))
    
    # Línea ideal
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Línea Ideal (y=x)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Predicción vs Valores Reales',
        xaxis_title='Valor Real',
        yaxis_title='Valor Predicho',
        height=450,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def crear_histograma_errores(errores: np.ndarray) -> go.Figure:
    """
    Crea histograma de distribución de errores.
    
    Args:
        errores: Array de errores.
        
    Returns:
        Figura Plotly.
    """
    fig = go.Figure(data=[go.Histogram(
        x=errores,
        nbinsx=30,
        marker_color=COLOR_PRINCIPAL,
        opacity=0.7,
        name='Errores'
    )])
    
    fig.update_layout(
        title='Distribución de Errores de Predicción',
        xaxis_title='Error (Predicho - Real)',
        yaxis_title='Frecuencia',
        height=400,
        template='plotly_white'
    )
    
    return fig


def crear_feature_importance(importancias: np.ndarray, 
                             nombres_features: List[str],
                             top_n: int = 10) -> go.Figure:
    """
    Crea gráfico de importancia de variables.
    
    Args:
        importancias: Array con importancia de cada feature.
        nombres_features: Lista de nombres de features.
        top_n: Top N features a mostrar.
        
    Returns:
        Figura Plotly.
    """
    # Crear DataFrame y ordenar
    df_imp = pd.DataFrame({
        'Feature': nombres_features,
        'Importancia': importancias
    }).sort_values('Importancia', ascending=True).tail(top_n)
    
    fig = go.Figure(data=[go.Bar(
        y=df_imp['Feature'],
        x=df_imp['Importancia'],
        orientation='h',
        marker_color=COLOR_PRINCIPAL,
        text=df_imp['Importancia'].round(4),
        textposition='outside'
    )])
    
    fig.update_layout(
        title=f'Top {top_n} Variables Más Importantes',
        xaxis_title='Importancia (SHAP value)',
        yaxis_title='Variable',
        height=400 + (top_n * 20),
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def crear_residuales(y_real: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Crea gráfico de residuales.
    
    Args:
        y_real: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        Figura Plotly.
    """
    residuales = y_real - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuales,
        mode='markers',
        name='Residuales',
        marker=dict(
            size=8,
            color=COLOR_PRINCIPAL,
            opacity=0.6,
            line=dict(width=1, color='white')
        )
    ))
    
    # Línea en y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Error = 0")
    
    fig.update_layout(
        title='Análisis de Residuales',
        xaxis_title='Valor Predicho',
        yaxis_title='Residual (Real - Predicho)',
        height=450,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def crear_boxplot_predicciones(predicciones: np.ndarray, 
                               grupos: List[str] = None) -> go.Figure:
    """
    Crea boxplot de predicciones.
    
    Args:
        predicciones: Array de predicciones.
        grupos: Lista con grupo de cada predicción (opcional).
        
    Returns:
        Figura Plotly.
    """
    if grupos is None:
        fig = go.Figure(data=[go.Box(
            y=predicciones,
            name='Predicciones',
            marker_color=COLOR_PRINCIPAL
        )])
    else:
        df = pd.DataFrame({
            'Predicción': predicciones,
            'Grupo': grupos
        })
        fig = px.box(df, y='Predicción', x='Grupo', 
                     color_discrete_sequence=[COLOR_PRINCIPAL])
    
    fig.update_layout(
        title='Distribución de Predicciones',
        yaxis_title='Demanda Predicha (unidades)',
        height=400,
        template='plotly_white'
    )
    
    return fig


def crear_comparativa_escenarios(escenario_a: dict, escenario_b: dict) -> go.Figure:
    """
    Crea comparativa entre dos escenarios.
    
    Args:
        escenario_a: Diccionario con datos del escenario A.
        escenario_b: Diccionario con datos del escenario B.
        
    Returns:
        Figura Plotly.
    """
    fig = go.Figure(data=[
        go.Bar(
            name='Escenario A',
            x=list(escenario_a.keys()),
            y=list(escenario_a.values()),
            marker_color=COLOR_SECUNDARIO
        ),
        go.Bar(
            name='Escenario B',
            x=list(escenario_b.keys()),
            y=list(escenario_b.values()),
            marker_color=COLOR_PRINCIPAL
        )
    ])
    
    fig.update_layout(
        title='Comparativa de Escenarios',
        barmode='group',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig
