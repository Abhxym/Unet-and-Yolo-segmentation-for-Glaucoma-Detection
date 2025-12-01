import torch
from model import NestedUNet, NestedUNetResNet
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def predict_pipeline(image_path, output_path='pipeline_prediction.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load YOLO Model
    # Find the best model path
    # Check recent runs first
    possible_paths = [
        "yolo_runs/train_od3/weights/best.pt",
        "yolo_runs/train_od2/weights/best.pt", 
        "yolo_runs/train_od/weights/best.pt"
    ]
    
    yolo_weights = None
    for path in possible_paths:
        if os.path.exists(path):
            yolo_weights = path
            break
            
    if yolo_weights is None:
        print("Error: YOLO weights not found in yolo_runs/")
        return

    print(f"Loading YOLO from {yolo_weights}")
    yolo_model = YOLO(yolo_weights)
    
    # 2. Load UNet Model
    # Use the new ResNet-based model
    unet_model = NestedUNetResNet(num_classes=2, input_channels=3, deep_supervision=False)
    # Use the crop-trained model
    checkpoint_path = 'checkpoints/best_model_crop.pth'
    if os.path.exists(checkpoint_path):
        unet_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded UNet checkpoint (Crop + ResNet).")
    else:
        print("Warning: No UNet checkpoint found. Using random weights.")
    
    unet_model = unet_model.to(device)
    unet_model.eval()
    
    # 3. Process Image
    original_image = Image.open(image_path).convert("RGB")
    w, h = original_image.size
    
    # Detect ROI
    results = yolo_model(original_image, conf=0.25)
    
    # Get best box
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No Optic Disc detected by YOLO. Falling back to full image.")
        # Fallback: Use center crop or full image
        # Let's use full image resized to 512x512
        crop_img = original_image.resize((512, 512))
        x1, y1, x2, y2 = 0, 0, w, h # For visualization
    else:
        # Take the box with highest confidence
        best_box = boxes[0]
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        
        # Add margin
        margin = 50
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Crop
        crop_img = original_image.crop((x1, y1, x2, y2))
    
    # Segment Crop
    # Resize crop to 512x512 for UNet
    input_tensor = TF.resize(crop_img, (512, 512))
    input_tensor = TF.to_tensor(input_tensor)
    input_tensor = TF.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = unet_model(input_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        
    pred_od = output[0, 0].cpu().numpy()
    pred_cup = output[0, 1].cpu().numpy()
    
    # Calculate vCDR on Crop
    def get_vertical_diameter(mask_2d):
        rows = np.any(mask_2d, axis=1)
        if not np.any(rows):
            return 0
        ymin, ymax = np.where(rows)[0][[0, -1]]
        return ymax - ymin
    
    pred_od_h = get_vertical_diameter(pred_od)
    pred_cup_h = get_vertical_diameter(pred_cup)
    pred_vcdr = pred_cup_h / pred_od_h if pred_od_h > 0 else 0
    
    # Visualization
    crop_np = np.array(crop_img.resize((512, 512)))
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original with Box
    ax[0].imshow(original_image)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
    ax[0].add_patch(rect)
    ax[0].set_title("YOLO Detection")
    ax[0].axis('off')
    
    # Crop with Segmentation
    ax[1].imshow(crop_np)
    ax[1].imshow(pred_od, alpha=0.4, cmap='Greens')
    ax[1].imshow(pred_cup, alpha=0.4, cmap='Reds')
    ax[1].set_title(f"Segmentation (Crop)\nvCDR: {pred_vcdr:.3f}")
    ax[1].axis('off')
    
    # Full Image Result (Paste back)
    # This is tricky because we resized the crop. 
    # For visualization, showing the crop is usually enough and better.
    # But let's show the crop.
    
    ax[2].imshow(crop_np)
    ax[2].set_title("Focused View")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Pipeline result saved to {output_path}")
    print(f"Calculated vCDR: {pred_vcdr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='pipeline_prediction.png', help='Path to output image')
    args = parser.parse_args()
    
    predict_pipeline(args.image, args.output)
