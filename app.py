import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Pulmonary Disease Detector",
    page_icon="🫁",
    layout="centered"
)

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_covid_v2.keras")

modelo = cargar_modelo()

clases = [
    "COVID",
    "LUNG_OPACITY",
    "NORMAL",
    "PNEUMONIA"
]

st.title("🫁 Pulmonary Disease Detector")

st.write(
    "Sube una radiografía de tórax y el modelo estimará la clase más probable."
)

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:

    imagen = Image.open(archivo).convert("RGB")

    st.image(imagen, caption="Imagen cargada")

    imagen = imagen.resize((224, 224))

    x = np.array(imagen) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = modelo.predict(x, verbose=0)

    score = pred[0]

    clase = clases[np.argmax(score)]
    confianza = float(np.max(score) * 100)

    st.subheader("Resultado")

    st.success(f"Predicción más probable: {clase}")
    st.write(f"Confianza máxima: {confianza:.2f}%")

    st.subheader("Probabilidades por categoría")

    for i, nombre in enumerate(clases):
        st.write(f"{nombre}: {score[i]*100:.2f}%")

    datos = {
        "COVID": score[0],
        "LUNG_OPACITY": score[1],
        "NORMAL": score[2],
        "PNEUMONIA": score[3]
    }

    st.bar_chart(datos)
