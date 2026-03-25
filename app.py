from ultralytics import YOLO
import streamlit as st
from PIL import Image


model = YOLO("yolov8n.pt")  


st.title("AI pentru diagnosticare radiografii")
uploaded_file = st.file_uploader("Încarcă o radiografie (JPG sau PNG)", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagine încărcată", use_column_width=True)

   
    results = model(img)

   
    results[0].save(save_dir="pred/")
    st.image("pred/image0.jpg", caption="Rezultat AI", use_column_width=True)
