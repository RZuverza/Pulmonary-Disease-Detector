# Detector de Enfermedades Pulmonares con IA

Proyecto desarrollado por Ricardo García Z.

## Descripción

Esta aplicación utiliza una red neuronal convolucional (CNN) entrenada con TensorFlow para clasificar radiografías de tórax en cuatro categorías:

- COVID
- LUNG_OPACITY
- NORMAL
- PNEUMONIA

La aplicación fue desplegada mediante Streamlit y permite cargar una imagen para obtener una predicción automática.

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Estructura del proyecto

proyecto_ai/
├── dataset/
├── app.py
├── covid_v2.ipynb
├── modelo_covid_v2.keras
└── requirements

## Resultados

- Accuracy de validación: ~82%
- Clasificación de 4 enfermedades pulmonares

## Autor

Ricardo García Zuverza.
Ingeniero Electromédico