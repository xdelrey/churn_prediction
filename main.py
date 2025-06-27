import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Modelo de predicción')

with st.form('formulario:'):
    marca = st.text_input('marca de pc:'),
    cpu = st.text_input('cpu:'),
    gpu = st.text_input('gpu:'),
    ram = st.number_input('ram:', step=1, max_value=10)
    boton = st.form_submit_button('enviar')

if boton:
    st.subheader('valores:')
    st.write('marca:', marca)
    st.write('cpu:', cpu)
    st.write('gpu:', gpu)
    st.write('ram:', ram)
