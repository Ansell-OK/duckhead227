import streamlit as st 
import tensorflow as tf 
import cv2
import numpy as np

model_path = "model.h5"

model = tf.keras.models.load_model(model_path)

CLASS_NAMES = ['Baked Potato',
 'Burger',
 'Crispy Chicken',
 'Donut',
 'Fries',
 'Hot Dog',
 'Pizza',
 'Sandwich',
 'Taco',
 'Taquito']

st.write("""
## Fast Food Prediction System🍔
         
This is a simple Image classification project to classify the following classes:
         
         *  Fries🍟
         *  Crispy Chicken🍗
         *  Sandwich🥪
         *  Baked Potato🍠
         *  Taquito🌯
         *  Pizza🍕
         *  Burger🍔
         *  Donut🍩
         *  Taco🌮
         *  Hot Dog🌭

This project incorporated transfer learning using a topless EfficientNetB0 model which achieved 89%.
         
""")

food_image = st.file_uploader("Choose Image.....", type=['jpeg', 'png'])
submit = st.button('Predict')

if submit:


    if food_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(food_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (150,150))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,150,150,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("Food is "+CLASS_NAMES[np.argmax(Y_pred)]))
