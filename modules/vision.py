from ultralytics import YOLO
import cv2
import numpy as np

class VisionSystem:
    def __init__(self, model_path='yolov8n.pt'):
        # Load pretrained YOLO model
        self.model = YOLO(model_path)

    def detect_objects(self, image_frame):
        """
        Input: Numpy array (Image)
        Output: List of detected object labels
        """
        results = self.model(image_frame, verbose=False)
        detected_items = []
        
        # Extract labels from YOLO results
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                conf = float(box.conf[0])
                
                if conf > 0.20: # Confidence threshold
                    detected_items.append(label)
        
        return list(set(detected_items)) # Return unique items