# Descripción del Proyecto

**Film Junky Union** es una comunidad vanguardista diseñada para los aficionados de las películas clásicas. Este proyecto tiene como objetivo desarrollar un sistema capaz de filtrar y categorizar reseñas de películas, entrenando un modelo que detecte automáticamente críticas negativas. Utilizando un conjunto de datos de reseñas de películas de IMDB con etiquetas de polaridad, el modelo debe ser capaz de clasificar correctamente las reseñas positivas y negativas, con un objetivo de rendimiento de un valor F1 de al menos 0.85.

## Estructura del Proyecto

1. **Inicialización:** Configuración del entorno de trabajo con las librerías necesarias.
2. **Carga de Datos:** Importación y limpieza del conjunto de datos.
3. **Análisis Exploratorio de Datos (EDA):** Visualización y análisis de la distribución de películas y reseñas a lo largo de los años.
4. **Procedimiento de Evaluación:** Implementación de una rutina de evaluación estandarizada para comparar diferentes modelos.
5. **Normalización:** Preprocesamiento del texto de las reseñas, incluyendo lematización y tokenización.
6. **División del Conjunto de Datos:** Separación de los datos en conjuntos de entrenamiento y prueba.
7. **Modelos:** 
   - **Modelo 1 - Constante:** Un modelo base utilizando `DummyClassifier`.
   - **Modelo 2 - NLTK, TF-IDF y LR:** Uso de `TfidfVectorizer` con `LogisticRegression`.
   - **Modelo 3 - spaCy, TF-IDF y LR:** Preprocesamiento avanzado con `spaCy` y clasificación con `LogisticRegression`.
   - **Modelo 4 - spaCy, TF-IDF y LGBMClassifier:** Clasificación con `LGBMClassifier`.
   - **Modelo 9 - BERT:** Uso de `BERT` para embeddings y clasificación.
8. **Evaluación Personalizada:** Aplicación de los modelos a reseñas personales para ver su rendimiento.
9. **Conclusiones:** Comparativa de modelos y análisis de resultados.

Este proyecto demuestra la efectividad de varios enfoques de procesamiento de lenguaje natural (NLP) y algoritmos de clasificación en la detección de la polaridad de reseñas cinematográficas.