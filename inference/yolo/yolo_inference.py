from ultralytics import YOLO
import os

class yolo():
    def __init__(self, target_id):
        current_dir = os.getcwd()
        self.model = YOLO(current_dir+'/assets/yolo.pt')
        self.target_id = target_id
        
    def predict(self, image, conf=0.25, iou=0.7, imgsz=640):
        results = self.model(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        detected = []
        results_count = results[0].boxes.cls.shape[0]
        if results_count > 0:
            for i in range(results_count):
                if results[0].boxes.cls[i] == self.target_id:
                    detected.append(results[0].boxes.xywh[i].cpu().numpy())
        return detected
