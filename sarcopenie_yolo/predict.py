from ultralytics import YOLO
import os

model = YOLO("runs/train/sarcopenie-psoas-seg/weights/best.pt") 


folder_input = "imagini_noi/"
folder_output = "rezultate/"


os.makedirs(folder_output, exist_ok=True)


results = model.predict(
    source=folder_input,
    save=True,
    save_txt=False,
    save_conf=True,
    project=folder_output,
    name="predictii",
    imgsz=512,
    conf=0.25,
    verbose=True
)

print("✅ Inferența s-a terminat. Imaginile segmentate sunt în folderul: rezultate/predictii")
