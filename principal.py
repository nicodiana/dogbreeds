from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st 
from openai import OpenAI

import os
import streamlit as st

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory containing the script
directory_files = os.listdir(script_dir)
#st.text("Files in directory: " + ", ".join(directory_files))

def classify_dog(img):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model_path = os.path.join(script_dir, "modelo", "keras_model.h5")
    model = load_model(model_path, compile=False)

    # Load the labels
    labels_path = os.path.join(script_dir, "modelo", "labels.txt")
    class_names = open(labels_path, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    return class_name, confidence_score




# Streamlit App

#st.set_page_config(layout='wide')

st.title("DogBreeds Detector")

st.subheader("""Adjuntá una foto de tu perro y descubrí mucho mas...""")
st.subheader("""Te vas a enterar de cosas que nunca supiste!""")
input_img = st.file_uploader("Elegir imagen", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Determinar la raza y recomendaciones personalizadas"):
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.info("Imagen cargada")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Resultado")
            image_file = Image.open(input_img)

            with st.spinner('Analizando imagen...'):
                label, confidence_score = classify_dog(image_file)

                # Extraer el nombre de la etiqueta sin el número
                label_description = label.split(maxsplit=1)[1]  # Divide la etiqueta por el primer espacio y toma el segundo elemento
                st.session_state['label'] = label_description  # Guarda la descripción en label2

                st.success(st.session_state['label'])  # Muestra la etiqueta sin el número

            
        with col3:
            st.info("Recomendaciones personalizadas")
            if st.session_state['label'] == 'German Sheperd':
                st.write('Que coma 3 veces por dia')
     






