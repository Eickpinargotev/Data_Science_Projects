# Predicción del Valor de Mercado para Autos Usados en Rusty Bargain

## Descripción del Proyecto

Este proyecto se enfoca en desarrollar un modelo de machine learning para predecir el valor de mercado de autos usados en la plataforma Rusty Bargain. La aplicación está diseñada para ayudar a los usuarios a determinar rápidamente el precio de sus vehículos utilizando datos históricos de especificaciones técnicas, versiones de equipamiento y precios.

### Objetivos Principales:
- **Calidad de la Predicción:** Garantizar predicciones precisas y alineadas con los valores de mercado.
- **Velocidad de Predicción:** Optimizar el modelo para ofrecer predicciones rápidas y eficientes.
- **Tiempo de Entrenamiento:** Minimizar el tiempo necesario para entrenar el modelo sin comprometer la precisión.

## Estructura del Proyecto

1. **Preparación de Datos:**
   - Limpieza de datos y generación de características para manejar valores nulos, eliminar duplicados y codificar variables categóricas.
   - Escalado de características numéricas para mejorar el rendimiento del modelo.

2. **Entrenamiento del Modelo:**
   - Implementación de varios modelos de regresión, incluyendo `RandomForestRegressor`, `CatBoostRegressor`, `LightGBMRegressor` y `XGBoostRegressor`.
   - Ajuste de hiperparámetros para optimizar el rendimiento del modelo.

3. **Evaluación del Modelo:**
   - Comparación de los modelos basados en RMSE (Root Mean Square Error) y eficiencia computacional.
   - Análisis de las predicciones del modelo para asegurar su alineación con los valores de mercado reales.

## Conclusión

El proyecto logró desarrollar un modelo predictivo efectivo para estimar el valor de mercado de autos usados. El `LightGBMRegressor` ofreció el mejor equilibrio entre velocidad y precisión, lo que lo convierte en la opción óptima para su implementación en la aplicación de Rusty Bargain.

## Instalación y Uso

1. Clona el repositorio en tu máquina local.
2. Instala las librerías necesarias usando `pip install -r requirements.txt`.
3. Ejecuta el notebook de Jupyter para entrenar el modelo y evaluar los resultados.

## Trabajo Futuro

Las mejoras futuras podrían incluir la experimentación con características adicionales, como variaciones de precios regionales y tendencias estacionales, para mejorar aún más la precisión del modelo.

## Dataset link
https://practicum-content.s3.us-west-1.amazonaws.com/datasets/car_data.csv
