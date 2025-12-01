import torch
from torch.utils.data import DataLoader
from dataset import DrishtiDataset
from model import NestedUNet
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Test-20211018T060000Z-001"
    dataset = DrishtiDataset(root_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # Shuffle to get random samples
    
    # Model
    model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using random weights.")
        
    model = model.to(device)
    model.eval()
    
    save_dir = 'visualizations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with torch.no_grad():
        # Get one batch (1 image)
        for i, (image, mask) in enumerate(dataloader):
            if i >= 5: break # Visualize 5 samples
            
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            
            # Prepare for plotting
            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            gt_od = mask[0, 0].cpu().numpy()
            gt_cup = mask[0, 1].cpu().numpy()
            pred_od = output[0, 0].cpu().numpy()
            pred_cup = output[0, 1].cpu().numpy()
            
            # Calculate vCDR
            def get_vertical_diameter(mask_2d):
                rows = np.any(mask_2d, axis=1)
                if not np.any(rows):
                    return 0
                ymin, ymax = np.where(rows)[0][[0, -1]]
                return ymax - ymin
            
            gt_od_h = get_vertical_diameter(gt_od)
            gt_cup_h = get_vertical_diameter(gt_cup)
            pred_od_h = get_vertical_diameter(pred_od)
            pred_cup_h = get_vertical_diameter(pred_cup)
            
            gt_vcdr = gt_cup_h / gt_od_h if gt_od_h > 0 else 0
            pred_vcdr = pred_cup_h / pred_od_h if pred_od_h > 0 else 0
            
            # Plot
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # Original Image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title(f"Original Image\nGT vCDR: {gt_vcdr:.3f}")
            axes[0, 0].axis('off')
            
            # GT OD
            axes[0, 1].imshow(img_np)
            axes[0, 1].imshow(gt_od, alpha=0.5, cmap='Greens')
            axes[0, 1].set_title("GT Optic Disc (Green)")
            axes[0, 1].axis('off')
            
            # GT Cup
            axes[0, 2].imshow(img_np)
            axes[0, 2].imshow(gt_cup, alpha=0.5, cmap='Reds')
            axes[0, 2].set_title("GT Cup (Red)")
            axes[0, 2].axis('off')
            
            # Prediction Image
            axes[1, 0].imshow(img_np)
            axes[1, 0].set_title(f"Prediction\nPred vCDR: {pred_vcdr:.3f}")
            axes[1, 0].axis('off')
            
            # Pred OD
            axes[1, 1].imshow(img_np)
            axes[1, 1].imshow(pred_od, alpha=0.5, cmap='Greens')
            axes[1, 1].set_title("Pred Optic Disc (Green)")
            axes[1, 1].axis('off')
            
            # Pred Cup
            axes[1, 2].imshow(img_np)
            axes[1, 2].imshow(pred_cup, alpha=0.5, cmap='Reds')
            axes[1, 2].set_title("Pred Cup (Red)")
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"vis_{i}.png"))
            plt.close()
            
    print(f"Visualizations saved to {save_dir}")

if __name__ == "__main__":
    visualize_results()
