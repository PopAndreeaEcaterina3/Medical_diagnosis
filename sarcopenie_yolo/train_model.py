from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt")  

model.train(
    data="yolo_dataset/data.yaml",  
    epochs=100,
    imgsz=512,
    batch=8,
    name="sarcopenie-psoas-seg",
    project="runs/train", 
    verbose=True
)
