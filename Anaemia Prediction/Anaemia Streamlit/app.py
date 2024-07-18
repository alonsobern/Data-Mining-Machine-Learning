import streamlit as st
import pandas as pd
from prediction import predict

st.title('Classifying Anaemia Disease')
st.markdown('Demo model to classify whether a patient could tends to have anaemia or not')

st.header('Anaemia Features')
col1, col2 = st.columns(2)

with col1:
    st.text('Categorical characteristics')
    sex = st.radio('Sex',['M','F'])
with col2:
    st.text('Numerical characteristics')
    red_pixel = st.slider('Red Pixel(%)', 0.0, 100.0, 0.5)
    green_pixel = st.slider('Green Pixel(%)', 0.0, 100.0, 0.5)
    blue_pixel = st.slider('Blue Pixel(%)', 0.0, 100.0, 0.5)
    hb = st.slider('Heamoglobin Level (g/dL)', 2.0, 20.0)

result = 0
if st.button('Predict Anaemia'):
    result = predict(pd.DataFrame(data=[[sex, red_pixel, green_pixel, blue_pixel, hb]],
                              columns=['sex','red_pixel','green_pixel','blue_pixel','hb']))[:,1]
    
st.text(f"Anaemia Probability: {round(float(result)*100,4)}%")
