import cv2
import numpy as np
from PIL import Image

def get_color(class_id):
    # Deterministic color generation based on class id
    np.random.seed(class_id * 100)
    return tuple(int(c) for c in np.random.randint(0, 255, size=3))

def draw_boxes(image, detections):
    # Convert PIL Image to OpenCV format if necessary
    if isinstance(image, Image.Image):
        # Convert PIL Image (RGB) to numpy array (RGB)
        image_np = np.array(image)
    else:
        # Assume it is already a numpy array
        image_np = image.copy()
    
    # Adaptive thickness based on image resolution
    height, width = image_np.shape[:2]
    thickness = max(1, int(min(height, width) / 500))
    font_scale = max(0.5, min(height, width) / 1000)
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["confidence"]
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        
        color = get_color(cls_id)
        
        # Draw bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image_np, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        # Draw text
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, max(1, thickness - 1))
        
    return image_np
