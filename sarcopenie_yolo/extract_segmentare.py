import cv2
import numpy as np
import os


nume_imagine = "CT_pacient1.jpg"  
input_path = f"yolo_dataset/images/train/{nume_imagine}"
output_path = f"yolo_dataset/labels/train/{nume_imagine.replace('.jpg', '.txt')}"


img = cv2.imread(input_path)
h, w = img.shape[:2]


lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_green, upper_green)


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    for cnt in contours:
        if len(cnt) < 3:
            continue  
        norm_coords = []
        for point in cnt:
            x, y = point[0]
            norm_coords.append(f"{x / w:.6f} {y / h:.6f}")
        f.write("0 " + " ".join(norm_coords) + "\n")

print(f"✅ Segmentarea pe verde a fost salvată în: {output_path}")
