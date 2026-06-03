# 📊 Dashboard de Predicción de Demanda - XGBoost Regressor

Aplicación **Streamlit** profesional para visualizar, explotar y analizar un modelo de Machine Learning entrenado con **XGBoost Regressor** para predicción de demanda de productos.

![Streamlit](https://img.shields.io/badge/Streamlit-v1.36-red?style=flat-square&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Descripción del Proyecto

Este dashboard integra un modelo XGBoost para:

- **Predecir demanda futura** de productos
- **Recomendar niveles de stock** óptimos
- **Analizar estacionalidad** y patrones
- **Simular escenarios** (What-If Analysis)
- **Evaluar calidad del modelo** con métricas completas| 

---

## 🎯 Uso

### Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Pasos Iniciales

1. **Asegúrate que `modelo_demanda.pkl` esté en el directorio raíz**
2. Abre la aplicación con el comando anterior
3. Navega por el **sidebar izquierdo** para acceder a cada módulo
4. Configura los parámetros globales (Producto, Sucursal, Año, Mes)

---

## 📑 Módulos Disponibles

### 1️⃣ Predicción Individual (`1_Prediccion_Individual.py`)

**Objetivo:** Realizar predicciones puntuales personalizadas

**Características:**
- Seleccionar producto, sucursal, mes, año
- Configurar precio, descuento, temporada
- Visualizar demanda estimada con gauge
- Calcular stock recomendado
- Detectar estado del inventario (falta, exceso, óptimo)
- Obtener explicaciones textuales automáticas

**Entradas:**
- Producto, Sucursal, Mes, Año
- Precio Unitario, Descuento, Temporada
- Stock Actual, Factor de Seguridad

**Salidas:**
- Demanda predicha
- Stock recomendado
- Diferencia y estado
- Visualizaciones interactivas

---

### 2️⃣ Planeación de Inventario (`2_Planeacion_Inventario.py`)

**Objetivo:** Visualizar demanda proyectada anual

**Características:**
- Predicción automática para cada mes del año
- Identificar meses de máxima/mínima demanda
- Calcular demanda anual total
- Detectar patrones estacionales
- Análisis automático de variabilidad
- Descargar tabla resumen en CSV

**KPIs Mostrados:**
- Demanda anual proyectada
- Mes de máxima demanda
- Mes de mínima demanda
- Promedio mensual

---

### 3️⃣ Simulación What-If (`3_Simulacion_WhatIf.py`)

**Objetivo:** Comparar escenarios modificando variables

**Características:**
- Crear dos escenarios (A y B) simultáneamente
- Variar precio, descuento, temporada
- Observar impacto en demanda en tiempo real
- Gráficos de sensibilidad (Precio vs Demanda, Descuento vs Demanda)
- Calcular variación absoluta y porcentual
- Análisis de elasticidad

**Análisis de Sensibilidad:**
- Simulación de precio (10 - 150)
- Simulación de descuento (0 - 50%)
- Comparativa entre escenarios

---

### 4️⃣ Análisis del Modelo (`4_Analisis_Modelo.py`)

**Objetivo:** Evaluar la calidad y confiabilidad del modelo

**Métricas Mostradas:**
- **R²:** Proporción de varianza explicada
- **MAE:** Error absoluto medio
- **RMSE:** Error cuadrático medio

**Visualizaciones:**
- Scatter plot: Predicción vs Real (con línea ideal y=x)
- Histograma de distribución de errores
- Ranking de importancia de variables (top 10)
- Gráfico de residuales

**Explicaciones Automáticas:**
- Interpretación de cada métrica
- Cómo funciona XGBoost
- Análisis de residuales
- Guía de variables importantes

---
---

## 🔧 Configuración del Modelo

### Cargar tu Propio Modelo

Si has entrenado tu propio modelo XGBoost:

1. Serializa el modelo con `joblib`:
```python
import joblib
joblib.dump(tu_modelo, 'modelo_demanda.pkl')
```

2. Coloca el archivo en el directorio raíz del proyecto

3. La aplicación lo cargará automáticamente

## 🎓 Ejemplos de Uso

### Ejemplo 1: Predicción Individual

1. Abre la página "Predicción Individual"
2. Selecciona:
   - Producto: A
   - Sucursal: Centro
   - Mes: Marzo
   - Año: 2025
   - Precio: $50
   - Descuento: 10%
3. Haz clic en "Realizar Predicción"
4. Analiza los resultados y recomendaciones

### Ejemplo 2: Planeación Anual

1. Ve a "Planeación de Inventario"
2. Elige un producto y año
3. Observa la proyección de 12 meses
4. Descarga la tabla resumen

### Ejemplo 3: What-If Analysis

1. Abre "Simulación What-If"
2. Configura dos escenarios diferentes
3. Compara el impacto en demanda
4. Toma decisiones basadas en datos

### Ejemplo 4: Análisis de Modelo

1. Ve a "Análisis del Modelo"
2. Revisa las métricas de desempeño
3. Analiza importancia de variables
4. Lee las explicaciones automáticas

---

## 📚 Referencias y Recursos

- [Documentación Streamlit](https://docs.streamlit.io/)
- [Documentación XGBoost](https://xgboost.readthedocs.io/)
- [Documentación Plotly](https://plotly.com/python/)
- [Scikit-Learn](https://scikit-learn.org/)

---

**Última actualización:** 2026  
**Versión:** 1.0  
**Estado:** ✓ Listo para producción
