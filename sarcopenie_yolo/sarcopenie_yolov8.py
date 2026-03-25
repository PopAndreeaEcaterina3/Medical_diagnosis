import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


DATASET_DIR = 'yolo_dataset'  
MODEL_ARCH = 'yolov8n-seg.pt' 
EPOCHS = 100
IMG_SIZE = 512
PIXEL_SPACING_MM = [0.6, 0.6]  
SEX = 'male'


yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/train
nc: 1
names: ['psoas']
"""

os.makedirs(DATASET_DIR, exist_ok=True)
with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
    f.write(yaml_content)


model = YOLO(MODEL_ARCH)
model.train(data=os.path.join(DATASET_DIR, 'data.yaml'), epochs=EPOCHS, imgsz=IMG_SIZE)

def calc_csa_from_mask(mask, spacing):
    area_pixels = np.sum(mask)
    area_mm2 = area_pixels * spacing[0] * spacing[1]
    area_cm2 = area_mm2 / 100
    return area_cm2

def predict_and_diagnose(image_path, model, spacing_mm, sex):
    results = model(image_path)
    masks = results[0].masks

    if masks is None:
        print("⚠️ Nicio mască detectată.")
        return

    total_area_cm2 = 0
    for i, mask_tensor in enumerate(masks.data):
        mask = mask_tensor.cpu().numpy()
        area = calc_csa_from_mask(mask, spacing_mm)
        print(f"CSA Psoas #{i+1}: {area:.2f} cm²")
        total_area_cm2 += area

    print(f"CSA Total: {total_area_cm2:.2f} cm²")

    threshold = 52.4 if sex == 'male' else 38.5
    if total_area_cm2 < threshold:
        print("⚠️ Posibilă sarcopenie")
    else:
        print("✅ CSA în limite normale")

   
    results[0].show()


predict_and_diagnose("yolo_dataset/images/train/SZILAGYILEVENTEpsoas1.jpg", model, PIXEL_SPACING_MM, SEX)
