import torch
from torch.utils.data import DataLoader
from dataset import DrishtiDataset
from model import NestedUNet
from utils import iou_score
import os
import numpy as np
from PIL import Image

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Test-20211018T060000Z-001"
    dataset = DrishtiDataset(root_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Model
    model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
    
    # Load checkpoint if exists, otherwise use random weights (for demo if not trained)
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using random weights.")
        
    model = model.to(device)
    model.eval()
    
    od_ious = []
    cup_ious = []
    
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with torch.no_grad():
        for i, (image, mask) in enumerate(dataloader):
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            
            # Calculate IoU for each class
            # mask shape: (B, 2, H, W)
            # output shape: (B, 2, H, W)
            
            od_iou = iou_score(output[:, 0], mask[:, 0])
            cup_iou = iou_score(output[:, 1], mask[:, 1])
            
            od_ious.append(od_iou.item())
            cup_ious.append(cup_iou.item())
            
            # Save first 5 results
            if i < 5:
                # Convert to numpy
                img_np = image[0].permute(1, 2, 0).cpu().numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                
                gt_od = mask[0, 0].cpu().numpy() * 255
                gt_cup = mask[0, 1].cpu().numpy() * 255
                pred_od = output[0, 0].cpu().numpy() * 255
                pred_cup = output[0, 1].cpu().numpy() * 255
                
                # Create a composite image
                # Row 1: Image, GT OD, GT Cup
                # Row 2: Image, Pred OD, Pred Cup
                
                img_pil = Image.fromarray(img_np)
                gt_od_pil = Image.fromarray(gt_od.astype(np.uint8))
                gt_cup_pil = Image.fromarray(gt_cup.astype(np.uint8))
                pred_od_pil = Image.fromarray(pred_od.astype(np.uint8))
                pred_cup_pil = Image.fromarray(pred_cup.astype(np.uint8))
                
                # Save individually or combined? Let's save individually for simplicity
                img_pil.save(os.path.join(save_dir, f"sample_{i}_image.png"))
                gt_od_pil.save(os.path.join(save_dir, f"sample_{i}_gt_od.png"))
                gt_cup_pil.save(os.path.join(save_dir, f"sample_{i}_gt_cup.png"))
                pred_od_pil.save(os.path.join(save_dir, f"sample_{i}_pred_od.png"))
                pred_cup_pil.save(os.path.join(save_dir, f"sample_{i}_pred_cup.png"))
                
    print(f"Mean OD IoU: {np.mean(od_ious):.4f}")
    print(f"Mean Cup IoU: {np.mean(cup_ious):.4f}")

    # Calculate vCDR
    # vCDR = Vertical Cup Diameter / Vertical Disc Diameter
    
    vcdr_errors = []
    
    print("\nCalculating vCDR...")
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(dataloader):
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            
            # Get masks as numpy arrays
            gt_od = mask[0, 0].cpu().numpy()
            gt_cup = mask[0, 1].cpu().numpy()
            pred_od = output[0, 0].cpu().numpy()
            pred_cup = output[0, 1].cpu().numpy()
            
            def get_vertical_diameter(mask_2d):
                # Find rows with at least one pixel
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
            
            error = abs(gt_vcdr - pred_vcdr)
            vcdr_errors.append(error)
            
            if i < 5:
                print(f"Sample {i}: GT vCDR = {gt_vcdr:.4f}, Pred vCDR = {pred_vcdr:.4f}, Error = {error:.4f}")
                
    print(f"\nMean Absolute Error (MAE) for vCDR: {np.mean(vcdr_errors):.4f}")
    
    # Accuracy (Pixel Accuracy)
    # Correct pixels / Total pixels
    # We can calculate this for OD and Cup separately
    
    print("\nCalculating Pixel Accuracy...")
    od_accs = []
    cup_accs = []
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(dataloader):
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            
            gt_od = mask[:, 0]
            gt_cup = mask[:, 1]
            pred_od = output[:, 0]
            pred_cup = output[:, 1]
            
            # Pixel Accuracy
            od_acc = (pred_od == gt_od).float().mean()
            cup_acc = (pred_cup == gt_cup).float().mean()
            
            od_accs.append(od_acc.item())
            cup_accs.append(cup_acc.item())
            
    print(f"Mean OD Pixel Accuracy: {np.mean(od_accs):.4f}")
    print(f"Mean Cup Pixel Accuracy: {np.mean(cup_accs):.4f}")

if __name__ == "__main__":
    evaluate()
