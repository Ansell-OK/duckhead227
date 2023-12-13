import streamlit as st 
import pickle

model_path = 'model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))

st.write("""
AI Prediction🤖
""")

input_user = st.text_input('Text Here', max_chars=100000)
submit = st.button('Submit')
if submit: 
    result = loaded_model.predict([input_user])
    if result[0] == 0:
        st.write("""Most Likely Human written""")
    else:
        st.write("""Written by AI""")