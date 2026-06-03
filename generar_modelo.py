"""
Script para generar un modelo XGBoost de ejemplo.
Ejecutar: python generar_modelo.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os

def generar_datos_sinteticos(n_samples: int = 500) -> tuple:
    """
    Genera datos sintéticos para entrenar el modelo.
    
    Args:
        n_samples: Número de muestras a generar.
        
    Returns:
        Tupla (X, y) con características y target.
    """
    np.random.seed(42)
    
    # Generar características
    producto = np.random.randint(1, 5, n_samples)
    sucursal = np.random.randint(1, 6, n_samples)
    mes = np.random.randint(1, 13, n_samples)
    año = np.random.randint(2023, 2026, n_samples)
    precio_unitario = np.random.uniform(30, 100, n_samples)
    descuento = np.random.uniform(0, 0.5, n_samples)
    temporada = np.random.randint(0, 3, n_samples)
    
    X = np.column_stack([
        producto, 
        sucursal, 
        mes, 
        año, 
        precio_unitario, 
        descuento, 
        temporada
    ])
    
    # Generar variable objetivo con relaciones sintéticas
    demanda_base = 500
    
    # Efecto de variables
    demanda = (
        demanda_base 
        + (producto * 30)  # Variar por producto
        + (sucursal * 20)  # Variar por sucursal
        + (mes * 10)  # Tendencia temporal
        + (np.sin(mes / 2) * 50)  # Estacionalidad
        + (temporada * 100)  # Efecto de temporada
        - (precio_unitario * 2)  # Precio inverso
        + (descuento * 200)  # Descuento positivo
        + np.random.normal(0, 100, n_samples)  # Ruido
    )
    
    # Asegurar demanda positiva
    y = np.maximum(demanda, 50)
    
    return X, y

def entrenar_modelo(X: np.ndarray, y: np.ndarray) -> XGBRegressor:
    """
    Entrena un modelo XGBoost.
    
    Args:
        X: Características de entrada.
        y: Variable objetivo.
        
    Returns:
        Modelo XGBRegressor entrenado.
    """
    print("📊 Entrenando modelo XGBoost...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crear y entrenar modelo
    modelo = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    modelo.fit(X_train, y_train, verbose=False)
    
    # Evaluar
    score_train = modelo.score(X_train, y_train)
    score_test = modelo.score(X_test, y_test)
    
    print(f"✓ Modelo entrenado")
    print(f"  - Score de entrenamiento (R²): {score_train:.4f}")
    print(f"  - Score de validación (R²): {score_test:.4f}")
    
    return modelo

def guardar_modelo(modelo: XGBRegressor, nombre_archivo: str = "modelo_demanda.pkl") -> None:
    """
    Guarda el modelo en un archivo pickle.
    
    Args:
        modelo: Modelo XGBRegressor a guardar.
        nombre_archivo: Nombre del archivo de salida.
    """
    print(f"💾 Guardando modelo como '{nombre_archivo}'...")
    joblib.dump(modelo, nombre_archivo)
    
    # Verificar
    if os.path.exists(nombre_archivo):
        tamaño = os.path.getsize(nombre_archivo)
        print(f"✓ Modelo guardado exitosamente")
        print(f"  - Tamaño: {tamaño / 1024:.2f} KB")
    else:
        print("❌ Error al guardar el modelo")

def main():
    """Función principal."""
    print("=" * 60)
    print("🤖 Generador de Modelo XGBoost - Predicción de Demanda")
    print("=" * 60)
    print()
    
    # Generar datos
    print("📈 Generando datos sintéticos...")
    X, y = generar_datos_sinteticos(n_samples=500)
    print(f"✓ {X.shape[0]} muestras generadas con {X.shape[1]} características")
    print()
    
    # Entrenar modelo
    modelo = entrenar_modelo(X, y)
    print()
    
    # Guardar modelo
    guardar_modelo(modelo)
    print()
    
    # Información del modelo
    print("ℹ️  Información del Modelo:")
    print(f"  - Tipo: XGBoost Regressor")
    print(f"  - Número de árboles: {modelo.n_estimators}")
    print(f"  - Profundidad máxima: {modelo.max_depth}")
    print(f"  - Variables de entrada: {modelo.n_features_in_}")
    print()
    
    # Importancia de características
    print("🎯 Importancia de Características:")
    nombres_features = ['Producto', 'Sucursal', 'Mes', 'Año', 'Precio', 'Descuento', 'Temporada']
    importancias = modelo.feature_importances_
    
    for nombre, importancia in sorted(zip(nombres_features, importancias), key=lambda x: x[1], reverse=True):
        barra = '█' * int(importancia * 50)
        print(f"  {nombre:15} │ {barra} {importancia:.4f}")
    print()
    
    # Predicción de ejemplo
    print("🔮 Ejemplo de Predicción:")
    ejemplo = np.array([[1, 1, 3, 2025, 50.0, 0.1, 1]])
    prediccion = modelo.predict(ejemplo)[0]
    print(f"  Entrada: Producto=1, Sucursal=1, Mes=3, Año=2025")
    print(f"           Precio=$50, Descuento=10%, Temporada=Media")
    print(f"  Predicción: {prediccion:.0f} unidades")
    print()
    
    print("=" * 60)
    print("✓ ¡Proceso completado exitosamente!")
    print("=" * 60)
    print()
    print("📝 Próximos pasos:")
    print("1. Asegúrate que 'modelo_demanda.pkl' está en el directorio")
    print("2. Ejecuta: streamlit run app.py")
    print("3. ¡Disfruta del dashboard!")
    print()

if __name__ == "__main__":
    main()
