import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Reciclaje IA-ISC", layout="centered")
st.title("Reciclaje clase de IA-ISC-CAMPUS COMAYAGUA")
st.write("Suba una imagen para clasificarla con el modelo MobileNetV2 entrenado.")
