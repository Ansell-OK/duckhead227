import streamlit as st 
import tensorflow as tf 
import numpy as np 
import cv2


model_path = 'kidney.h5'

model = tf.keras.models.load_model(model_path)

CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']

st.write("""

## KIDNEY HEALTH DIAGNOSIS🧑🏽‍⚕️
***
This application aims to predict if a kidney's health status is Normal if it is plagued with a         
         *  Cyst 
         *  Stone
         *  Tumor 
""")

kidney_image = st.file_uploader("Choose Image.....", type=['jpeg', 'png', 'jpg'])
submit = st.button('Predict')

if submit:


    if kidney_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(kidney_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (300,300))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,300,300,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        if np.argmax(Y_pred) == 1:
            st.title("You're all good, Kidney Status is Normal")
        else:
            st.title(str("Need to get yourself check out, Your kidney has a "+CLASS_NAMES[np.argmax(Y_pred)]))
            