import torch
from torch.utils.data import DataLoader
from dataset import DrishtiDataset
from model import NestedUNet, NestedUNetResNet
from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

def get_vertical_diameter(mask_2d):
    rows = np.any(mask_2d, axis=1)
    if not np.any(rows):
        return 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    return ymax - ymin

def evaluate_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load YOLO
    yolo_weights = "yolo_runs/train_od3/weights/best.pt"
    if not os.path.exists(yolo_weights):
        yolo_weights = "yolo_runs/train_od2/weights/best.pt"
    
    print(f"Loading YOLO from {yolo_weights}")
    yolo_model = YOLO(yolo_weights)
    
    # 2. Load UNet (Crop)
    # Use the new ResNet-based model
    unet_model = NestedUNetResNet(num_classes=2, input_channels=3, deep_supervision=False)
    checkpoint_path = 'checkpoints/best_model_crop.pth'
    if os.path.exists(checkpoint_path):
        unet_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded UNet checkpoint (Crop).")
    else:
        print("Warning: No UNet checkpoint found.")
        return
        
    unet_model = unet_model.to(device)
    unet_model.eval()
    
    # 3. Data
    # We use the original dataset class to get full images and full masks
    # We will perform the cropping logic manually in the evaluation loop to match inference
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Test-20211018T060000Z-001"
    dataset = DrishtiDataset(root_dir, mode='test')
    # Note: DrishtiDataset resizes to 512x512. 
    # For pipeline evaluation, we ideally want ORIGINAL images to test the "in the wild" performance.
    # But DrishtiDataset already loads and resizes. 
    # Let's use the raw file paths from the dataset to load original images.
    
    image_paths = dataset.image_paths
    gt_dir = dataset.gt_dir
    
    vcdr_errors = []
    vcdr_mapes = []
    ious_od = []
    ious_cup = []
    accuracies_od = []
    accuracies_cup = []
    
    print(f"Evaluating on {len(image_paths)} images...")
    
    for i, img_path in enumerate(tqdm(image_paths)):
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Load Original Image
        original_image = Image.open(img_path).convert("RGB")
        w, h = original_image.size
        
        # Load GT Masks (Original Size)
        gt_folder = os.path.join(gt_dir, base_name, 'SoftMap')
        od_mask_path = os.path.join(gt_folder, f"{base_name}_ODsegSoftmap.png")
        cup_mask_path = os.path.join(gt_folder, f"{base_name}_cupsegSoftmap.png")
        
        if not os.path.exists(od_mask_path): continue
        
        gt_od_mask = np.array(Image.open(od_mask_path).convert("L"))
        gt_cup_mask = np.array(Image.open(cup_mask_path).convert("L"))
        gt_od_mask = (gt_od_mask > 128)
        gt_cup_mask = (gt_cup_mask > 128)
        
        # Calculate GT vCDR
        gt_od_h = get_vertical_diameter(gt_od_mask)
        gt_cup_h = get_vertical_diameter(gt_cup_mask)
        gt_vcdr = gt_cup_h / gt_od_h if gt_od_h > 0 else 0
        
        # --- Pipeline Inference ---
        
        # Detect ROI
        results = yolo_model(original_image, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            # Fallback
            crop_img = original_image.resize((512, 512))
        else:
            best_box = boxes[0]
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            
            margin = 50
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            crop_img = original_image.crop((x1, y1, x2, y2))
            
        # 4. Segment (with TTA)
        # Resize to 512x512
        crop_img = TF.resize(crop_img, (512, 512))
        
        with torch.no_grad():
            # Original
            input_tensor = TF.to_tensor(crop_img).unsqueeze(0).to(device)
            input_tensor = TF.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            output_orig = unet_model(input_tensor)
            prob_orig = torch.sigmoid(output_orig)
            
            # Horizontal Flip TTA
            input_flip = torch.flip(input_tensor, [3])
            output_flip = unet_model(input_flip)
            prob_flip = torch.sigmoid(output_flip)
            prob_flip = torch.flip(prob_flip, [3])
            
            # Average
            output = (prob_orig + prob_flip) / 2.0
            
        pred_od = (output[0, 0] > 0.5).float().cpu().numpy()
        pred_cup = (output[0, 1] > 0.5).float().cpu().numpy()
        
        # Calculate Pred vCDR (on crop, ratio is invariant)
        pred_od_h = get_vertical_diameter(pred_od)
        pred_cup_h = get_vertical_diameter(pred_cup)
        pred_vcdr = pred_cup_h / pred_od_h if pred_od_h > 0 else 0
        
        # Metrics
        error = abs(pred_vcdr - gt_vcdr)
        vcdr_errors.append(error)
        
        # MAPE (avoid division by zero)
        if gt_vcdr > 0:
            mape = (error / gt_vcdr) * 100
            vcdr_mapes.append(mape)
            
        # Pixel Accuracy & IoU (on crop)
        # We need to resize GT to 512x512 to compare with prediction
        gt_od_crop = TF.resize(Image.fromarray(gt_od_mask), (512, 512), interpolation=TF.InterpolationMode.NEAREST)
        gt_cup_crop = TF.resize(Image.fromarray(gt_cup_mask), (512, 512), interpolation=TF.InterpolationMode.NEAREST)
        
        # Need to crop GT first! 
        # Wait, we cropped the original image. We must crop the GT masks using the SAME coordinates.
        # But we didn't save the coordinates in a way to easily crop the loaded full-size GT masks here.
        # Actually, we loaded full size GT masks earlier: gt_od_mask, gt_cup_mask
        
        if len(boxes) > 0:
             # Apply same crop to GT
             gt_od_crop = Image.fromarray(gt_od_mask).crop((x1, y1, x2, y2))
             gt_cup_crop = Image.fromarray(gt_cup_mask).crop((x1, y1, x2, y2))
        else:
             gt_od_crop = Image.fromarray(gt_od_mask).resize((512, 512))
             gt_cup_crop = Image.fromarray(gt_cup_mask).resize((512, 512))
             
        gt_od_crop = TF.resize(gt_od_crop, (512, 512), interpolation=TF.InterpolationMode.NEAREST)
        gt_cup_crop = TF.resize(gt_cup_crop, (512, 512), interpolation=TF.InterpolationMode.NEAREST)
        
        gt_od_np = np.array(gt_od_crop) > 0 # Boolean
        gt_cup_np = np.array(gt_cup_crop) > 0
        
        pred_od_bool = pred_od > 0.5
        pred_cup_bool = pred_cup > 0.5
        
        # IoU
        intersection_od = np.logical_and(gt_od_np, pred_od_bool).sum()
        union_od = np.logical_or(gt_od_np, pred_od_bool).sum()
        iou_od = intersection_od / union_od if union_od > 0 else 0
        ious_od.append(iou_od)
        
        intersection_cup = np.logical_and(gt_cup_np, pred_cup_bool).sum()
        union_cup = np.logical_or(gt_cup_np, pred_cup_bool).sum()
        iou_cup = intersection_cup / union_cup if union_cup > 0 else 0
        ious_cup.append(iou_cup)
        
        # Pixel Accuracy
        acc_od = (gt_od_np == pred_od_bool).mean()
        acc_cup = (gt_cup_np == pred_cup_bool).mean()
        accuracies_od.append(acc_od)
        accuracies_cup.append(acc_cup)
        
    mae = np.mean(vcdr_errors)
    mape = np.mean(vcdr_mapes)
    mean_iou_od = np.mean(ious_od)
    mean_iou_cup = np.mean(ious_cup)
    mean_acc_od = np.mean(accuracies_od)
    mean_acc_cup = np.mean(accuracies_cup)
    
    print(f"\nFinal Results:")
    print(f"vCDR MAE: {mae:.4f}")
    print(f"vCDR MAPE: {mape:.2f}% (Accuracy: {100-mape:.2f}%)")
    print(f"Mean IoU (OD): {mean_iou_od:.4f}")
    print(f"Mean IoU (Cup): {mean_iou_cup:.4f}")
    print(f"Pixel Accuracy (OD): {mean_acc_od*100:.2f}%")
    print(f"Pixel Accuracy (Cup): {mean_acc_cup*100:.2f}%")

if __name__ == "__main__":
    evaluate_pipeline()
