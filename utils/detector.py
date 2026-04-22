import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path="model/best.pt", labels_path="model/labels.txt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Read class names dynamically
        self.class_names = {}
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                for idx, line in enumerate(f):
                    name = line.strip()
                    if name:
                        self.class_names[idx] = name
        else:
            self.class_names = self.model.names  # Fallback to model's default names

    def predict(self, image, conf=0.5):
        # The model expects RGB image or numpy array
        results = self.model(image, conf=conf, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"Class {class_id}")
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
                
        return detections
