import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prediction import predict

st.title('Predicting Anaemia Disease')
st.markdown('Demo model to predict whether an individual could tend to have anaemia or not')

st.header('Anaemia Features')
col1, col2 = st.columns(2)

with col1:
    st.text('Numerical')
    red_pixel = st.slider('Red Pixel(%)', 0.0, 100.0, 0.5)
    green_pixel = st.slider('Green Pixel(%)', 0.0, 100.0, 0.5)
    blue_pixel = st.slider('Blue Pixel(%)', 0.0, 100.0, 0.5)
    hb = st.slider('Hemoglobin Level (g/dL)', 2.0, 20.0)
with col2:
    st.text('Categorical')
    sex = st.radio('Sex',['M','F'])
    st.text('Sample Image Pixels')
    fig, ax = plt.subplots()
    r = red_pixel / 100
    g = green_pixel / 100
    b = blue_pixel / 100
    color = (r, g, b)
    ax.imshow([[color]])
    ax.axis(False)
    st.pyplot(fig)

result = 0
if st.button('Get Anaemia Prediction'):
    result = predict(pd.DataFrame(data=[[sex, red_pixel, green_pixel, blue_pixel, hb]],
                              columns=['sex','red_pixel','green_pixel','blue_pixel','hb']))[:,1]
    
    result = round(float(result)*100,4)

    if result > 60:
        st.warning("You have high probability to get anaemia.")
    else:
        st.success("You do not have high probability to get anaemia.")
   
    st.text(f"Anaemia Probability: {result}%")