import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class DrishtiDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' or 'test'.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        if self.mode == 'train':
            self.images_dir = os.path.join(root_dir, 'Training', 'Images')
            self.gt_dir = os.path.join(root_dir, 'Training', 'GT')
        elif self.mode == 'test':
            self.images_dir = os.path.join(root_dir, 'Test', 'Images')
            self.gt_dir = os.path.join(root_dir, 'Test', 'Test_GT')
        else:
            raise ValueError("Mode must be 'train' or 'test'")

        # Recursively find all png images in images_dir
        self.image_paths = glob.glob(os.path.join(self.images_dir, '**', '*.png'), recursive=True)
        
        if len(self.image_paths) == 0:
             print(f"Warning: No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0] # e.g., drishtiGS_001
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        
        # Load Masks
        # GT structure: GT/drishtiGS_XXX/SoftMap/drishtiGS_XXX_ODsegSoftmap.png
        #               GT/drishtiGS_XXX/SoftMap/drishtiGS_XXX_cupsegSoftmap.png
        
        gt_folder = os.path.join(self.gt_dir, base_name, 'SoftMap')
        
        od_mask_path = os.path.join(gt_folder, f"{base_name}_ODsegSoftmap.png")
        cup_mask_path = os.path.join(gt_folder, f"{base_name}_cupsegSoftmap.png")
        
        if not os.path.exists(od_mask_path):
            raise FileNotFoundError(f"OD mask not found: {od_mask_path}")
        if not os.path.exists(cup_mask_path):
            raise FileNotFoundError(f"Cup mask not found: {cup_mask_path}")
            
        od_mask = Image.open(od_mask_path).convert("L")
        cup_mask = Image.open(cup_mask_path).convert("L")
        
        # Resize if needed (we will do this in transform usually, but let's ensure consistency)
        # For simplicity, we can do basic resizing here if transform is not provided or part of it.
        # But standard way is to pass transforms.
        
        if self.transform:
            # Apply transforms to both image and masks
            # Note: We need to be careful to apply same random transforms to both.
            # For now, let's assume transform is a function that takes image and masks
            # OR we handle simple resizing here.
            pass

        # Convert to tensor
        # We'll do manual transformation to ensure masks and images are synced
        
        # Resize to 512x512 for now (standard for UNet)
        target_size = (512, 512)
        image = TF.resize(image, target_size)
        od_mask = TF.resize(od_mask, target_size, interpolation=Image.NEAREST)
        cup_mask = TF.resize(cup_mask, target_size, interpolation=Image.NEAREST)
        
        image = TF.to_tensor(image)
        od_mask = TF.to_tensor(od_mask)
        cup_mask = TF.to_tensor(cup_mask)
        
        # Normalize image
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Binarize masks (SoftMap might be probability, but usually it's 0 or 255)
        od_mask = (od_mask > 0.5).float()
        cup_mask = (cup_mask > 0.5).float()
        
        # Combine masks: Channel 0 = OD, Channel 1 = Cup
        mask = torch.cat([od_mask, cup_mask], dim=0)
        
        
        return image, mask

class DrishtiCropDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, margin=50):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.margin = margin
        
        if self.mode == 'train':
            self.images_dir = os.path.join(root_dir, 'Training', 'Images')
            self.gt_dir = os.path.join(root_dir, 'Training', 'GT')
        else:
            self.images_dir = os.path.join(root_dir, 'Test', 'Images')
            self.gt_dir = os.path.join(root_dir, 'Test', 'Test_GT')
            
        # Get all image files
        self.image_paths = []
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(root, file))
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        # Load Masks
        if self.mode == 'train':
            gt_folder = os.path.join(self.gt_dir, base_name, 'SoftMap')
        else:
            gt_folder = os.path.join(self.gt_dir, base_name, 'SoftMap')
            
        od_mask_path = os.path.join(gt_folder, f"{base_name}_ODsegSoftmap.png")
        cup_mask_path = os.path.join(gt_folder, f"{base_name}_cupsegSoftmap.png")
        
        od_mask = Image.open(od_mask_path).convert("L")
        cup_mask = Image.open(cup_mask_path).convert("L")
        
        # Find BBox from OD Mask
        od_np = np.array(od_mask)
        rows = np.any(od_np > 128, axis=1)
        cols = np.any(od_np > 128, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Fallback if empty mask (shouldn't happen often in training)
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            x1 = max(0, xmin - self.margin)
            y1 = max(0, ymin - self.margin)
            x2 = min(w, xmax + self.margin)
            y2 = min(h, ymax + self.margin)
            
        # Crop
        image = image.crop((x1, y1, x2, y2))
        od_mask = od_mask.crop((x1, y1, x2, y2))
        cup_mask = cup_mask.crop((x1, y1, x2, y2))
        
        # Resize to 512x512 (or smaller if we want)
        image = TF.resize(image, (512, 512))
        od_mask = TF.resize(od_mask, (512, 512), interpolation=TF.InterpolationMode.NEAREST)
        cup_mask = TF.resize(cup_mask, (512, 512), interpolation=TF.InterpolationMode.NEAREST)
        
        # Data Augmentation (Train only)
        if self.mode == 'train':
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                od_mask = TF.hflip(od_mask)
                cup_mask = TF.hflip(cup_mask)
                
            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                od_mask = TF.vflip(od_mask)
                cup_mask = TF.vflip(cup_mask)
                
            # Random Rotation
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            od_mask = TF.rotate(od_mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            cup_mask = TF.rotate(cup_mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            
            # Color Jitter (Image only)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                
        # To Tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        od_mask = np.array(od_mask)
        cup_mask = np.array(cup_mask)
        
        od_mask = (od_mask > 128).astype(np.float32)
        cup_mask = (cup_mask > 128).astype(np.float32)
        
        mask = np.stack([od_mask, cup_mask], axis=0)
        mask = torch.from_numpy(mask)
        
        return image, mask
