import streamlit as st
import pandas as pd
import os

st.title("Simple Excel Viewer")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df)