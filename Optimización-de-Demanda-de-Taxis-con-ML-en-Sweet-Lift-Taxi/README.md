# Sweet Lift Taxi - Predicción de Demanda de Taxis

## Descripción del Proyecto

El proyecto **Sweet Lift Taxi** se centra en el desarrollo de un modelo predictivo para estimar la demanda de taxis en los aeropuertos durante las horas pico. La compañía ha recopilado un extenso conjunto de datos históricos sobre pedidos de taxis y busca utilizar este conocimiento para optimizar la disponibilidad de vehículos. El objetivo principal es predecir el número de pedidos de taxis en la siguiente hora, con el fin de atraer a más conductores en los momentos de mayor demanda.

El éxito del proyecto se mide mediante la métrica de error cuadrático medio de la raíz (RMSE), con la meta de mantener el RMSE por debajo de 48 en el conjunto de prueba.

## Estructura del Proyecto

1. **Descripción de los Datos:**
   - El conjunto de datos contiene registros históricos de pedidos de taxis, almacenados en el archivo `taxi.csv`. La columna `num_orders` representa el número de pedidos por hora.

2. **Preparación de los Datos:**
   - Importación y preprocesamiento de los datos, incluyendo la conversión de las fechas y la creación de nuevas características que capturan tendencias y patrones estacionales en los datos.

3. **Análisis Exploratorio de Datos (EDA):**
   - Visualización y análisis de la serie temporal para identificar componentes tendenciales, estacionales y residuales, lo que ayuda a comprender mejor el comportamiento de los pedidos a lo largo del tiempo.

4. **Formación de Modelos:**
   - Implementación de varios algoritmos de regresión, incluidos `LinearRegression`, `RandomForestRegressor`, y `LightGBMRegressor`, para predecir la demanda de taxis. Se ajustan los hiperparámetros y se comparan los modelos según su desempeño en el conjunto de prueba.

5. **Evaluación de Resultados:**
   - Evaluación del rendimiento de los modelos utilizando la métrica RMSE, con un enfoque en minimizar este valor para mejorar la precisión de las predicciones.

## Conclusión

A lo largo del proyecto, se realizó un exhaustivo preprocesamiento de datos, enriqueciendo el conjunto con nuevas características que capturan patrones cruciales de la serie temporal. Se evaluaron varios modelos predictivos, y se encontró que el **LightGBMRegressor** ofreció el mejor rendimiento, alcanzando un RMSE de 43.68 en el conjunto de prueba. Este resultado no solo cumple con el objetivo establecido, sino que también destaca la eficacia de los algoritmos de potenciación del gradiente para el análisis de series temporales en problemas de predicción de demanda.

Este proyecto proporciona una base sólida para futuras mejoras en la predicción de la demanda de taxis, permitiendo a Sweet Lift Taxi optimizar su operación y satisfacer mejor la demanda de sus clientes.