from ultralytics import YOLO
import os

def train_yolo():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # We need to pass the absolute path to data.yaml
    data_path = os.path.abspath(r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/yolo_data/data.yaml")
    
    print(f"Training using data at: {data_path}")
    
    results = model.train(data=data_path, epochs=100, imgsz=512, project="yolo_runs", name="train_od")
    
    # Export the model
    success = model.export(format="onnx")

if __name__ == "__main__":
    train_yolo()
