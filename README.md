# 🧠 CT Scan Medical Diagnosis using YOLOv8

This project utilizes the YOLOv8 image classification/detection model to analyze Computed Tomography (CT) scans. The script evaluates medical images and predicts the presence of [Disease Name / Anomalies] with a calculated confidence score.

*⚠️ Important Note: This is an educational and research-oriented Computer Vision project. The model's predictions do not replace an official professional medical diagnosis.*

## 🛠️ Tech Stack
* **Language:** Python 
* **Machine Learning / Computer Vision:** Ultralytics (YOLOv8), OpenCV, PyTorch
* **Data Manipulation:** NumPy, Pandas 

## ✨ Key Features
* ⚙️ **Fast Inference:** Quickly processes CT images and provides a prediction with a confidence score.
* 🎯 **High Accuracy:** The model is fine-tuned on a specific medical dataset to recognize the visual patterns associated with sarcopeny.
* 📂 **Batch Processing Support:** Can evaluate a single image or process an entire folder of CT scans directly via the Command Line Interface (CLI).
* 🖼️ **Result Generation:** Automatically saves the processed images, highlighting the predicted outcomes.

## 🧠 About the Model (YOLOv8)
I chose the YOLOv8 architecture due to its state-of-the-art performance and speed in image analysis. The custom model was trained on a dataset consisting of [Number of images, e.g., 5000+] medical images. 
*The model achieved an accuracy of 92% / an mAP50-95 of 0.85 on the validation set.*

## 🚀 How to Run Locally

1. Clone this repository: 
   ```bash
   git clone your-repository-link-here
   cd your-folder-name
