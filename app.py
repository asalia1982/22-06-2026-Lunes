import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Reciclaje IA-ISC", layout="centered")
st.title("Reciclaje clase de IA-ISC-CAMPUS COMAYAGUA")
st.write("Suba una imagen para clasificarla con el modelo MobileNetV2 entrenado.")

IMG_SIZE = (224, 224)
MODEL_DIR = Path("modelo_reciclaje_mobilenet")
CLASS_PATH = MODEL_DIR / "class_names.json"
MODEL_PATHS = [MODEL_DIR / "waste_mobilenet.keras", MODEL_DIR / "waste_mobilenet.h5"]

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}
