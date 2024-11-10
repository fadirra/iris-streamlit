# code inspired by:
# https://365datascience.com/blog/authors/santiago-viquez/

import streamlit as st # lib for web app interfaces
import pandas as pd
import numpy as np
from prediction import predict # import from a custom module (see prediction.py)

# set title and description
st.title("Iris Flower Classification by Fasilkom/Pusilkom UI")
st.markdown("Classify iris flowers based on their sepal/petal length/width.")

st.header("Iris Flower Features")

# create two side-by-side columns for inputting sepal and petal features
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal length (cm)', min_value=4.0, max_value=8.0, step=0.1)
    sepal_w = st.slider('Sepal width (cm)', min_value=2.0, max_value=5.0, step=0.1)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal length (cm)', min_value=1.0, max_value=7.0, step=0.1)
    petal_w = st.slider('Petal width (cm)', min_value=0.1, max_value=3.0, step=0.1)

# set UI for class prediction
st.text('')
if st.button("Predict class"):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.success(f"Result: {result[0]}")
