import cv2
from PIL import Image
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

def main():
    print("Loading model...")
    model = YOLOModel(model_path="model/best.pt", labels_path="model/labels.txt")
    
    image_path = "assets/demo.png"
    print(f"Loading image {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find {image_path}. Please place an image there.")
        return

    print("Running inference...")
    detections = model.predict(image, conf=0.5)
    
    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f" - {det['class_name']}: {det['confidence']:.2f} at {det['box']}")
        
    print("Drawing boxes...")
    # draw_boxes expects and returns RGB numpy array
    annotated_rgb = draw_boxes(image, detections)
    
    # Convert RGB to BGR for OpenCV writing
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated_bgr)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
