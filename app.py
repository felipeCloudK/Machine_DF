import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Generador de Retratos Artísticos")
st.sidebar.write("Opciones de Generación")

# Opciones del usuario
num_images = st.sidebar.slider("Número de imágenes", 1, 16, 4)  # Slider para seleccionar número de imágenes
latent_dim = 128  # Dimensión del vector latente

if st.button("Generar Imágenes"):
    # Cargar el modelo ONNX
    ort_session = ort.InferenceSession("generator.onnx")

    # Generar vectores latentes aleatorios
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim)).astype(np.float32)

    # Ejecutar el modelo ONNX con los vectores latentes
    onnx_output = ort_session.run(None, {"input": random_latent_vectors})

    # Visualizar las imágenes generadas
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow((onnx_output[0][i] + 1) / 2)  # Normalizar las imágenes al rango [0, 1]
        ax.axis("off")
    
    # Mostrar las imágenes en Streamlit
    st.pyplot(fig)
