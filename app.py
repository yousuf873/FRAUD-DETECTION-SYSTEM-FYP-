import streamlit as st
import os

st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir("."))
st.write("__file__ path:", os.path.abspath(__file__))
