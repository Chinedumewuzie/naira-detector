import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Naira Note Detector")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload a Naira image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    results = model.predict(source=temp_path, save=False)
    result = results[0]

    annotated = result.plot()
    st.image(annotated, caption="Detected Result")

    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"{names[cls_id]} — {conf:.2f}")