from ultralytics import YOLO
import sys

def debug_yolo():
    model = YOLO("yolo_runs/train_od2/weights/best.pt")
    img_path = "drishti/Test-20211018T060000Z-001/Test/Images/glaucoma/drishtiGS_001.png"
    
    print(f"Predicting on {img_path}")
    results = model(img_path, conf=0.001) # Very low conf
    
    for r in results:
        print(f"Boxes: {len(r.boxes)}")
        for box in r.boxes:
            print(f"Conf: {box.conf.item():.4f}, Class: {box.cls.item()}")

if __name__ == "__main__":
    debug_yolo()
