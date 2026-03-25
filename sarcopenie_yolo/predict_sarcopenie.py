from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path


model = YOLO("runs/train/sarcopenie-psoas-seg/weights/best.pt")

folder_input = "imagini_noi/"
folder_output = "rezultate/predictii/"
os.makedirs(folder_output, exist_ok=True)


results = model.predict(
    source=folder_input,
    save=True,
    save_txt=False,
    save_conf=True,
    project="rezultate",
    name="predictii",
    imgsz=512,
    conf=0.25
)


PRAG_SARCOPENIE = 3000 
print("\n Analiză sarcopenie:\n")

for i, result in enumerate(results):
    masks = result.masks  
    img_path = result.path
    file_name = Path(img_path).name

    if masks is not None:
        mask_array = masks.data.cpu().numpy()  

        combined_mask = np.any(mask_array, axis=0).astype(np.uint8)

       
        aria_pixeli = np.sum(combined_mask)

      
        verdict = "Sarcopenie detectată " if aria_pixeli < PRAG_SARCOPENIE else "Nu are sarcopenie "

        print(f"{file_name}: Arie estimată = {aria_pixeli} pixeli → {verdict}")
    else:
        print(f"{file_name}: Nicio segmentare detectată ")

print("\n Gata! Verifică folderul 'rezultate/predictii' pentru imagini.")
