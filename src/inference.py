import torch
from model import NestedUNet
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import argparse
import os

def predict_single_image(image_path, output_path='prediction.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using random weights.")
        return

    model = model.to(device)
    model.eval()
    
    # Load and Preprocess Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    original_image = Image.open(image_path).convert("RGB")
    # Resize to 512x512 for model
    image = TF.resize(original_image, (512, 512))
    image_tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        
    # Process Output
    pred_od = output[0, 0].cpu().numpy()
    pred_cup = output[0, 1].cpu().numpy()
    
    # Calculate vCDR
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
    img_np = np.array(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_np)
    ax.imshow(pred_od, alpha=0.4, cmap='Greens')
    ax.imshow(pred_cup, alpha=0.4, cmap='Reds')
    ax.set_title(f"Prediction\nvCDR: {pred_vcdr:.3f}")
    ax.axis('off')
    
    plt.savefig(output_path)
    print(f"Prediction saved to {output_path}")
    print(f"Calculated vCDR: {pred_vcdr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='prediction.png', help='Path to output image')
    args = parser.parse_args()
    
    predict_single_image(args.image, args.output)
