import torch
from dataset import DrishtiCropDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

def verify_augmentation():
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Training-20211018T055246Z-001"
    # Enable training mode to trigger augmentation
    dataset = DrishtiCropDataset(root_dir, mode='train')
    
    # Get the same image multiple times to see different augmentations
    idx = 0 
    
    if not os.path.exists('aug_vis'):
        os.makedirs('aug_vis')
        
    print(f"Saving augmented samples to aug_vis/...")
    
    for i in range(5):
        image, mask = dataset[idx]
        
        # Denormalize image
        img_np = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        od_mask = mask[0].numpy()
        cup_mask = mask[1].numpy()
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].imshow(img_np)
        ax[0].set_title(f"Augmentation {i+1}")
        ax[0].axis('off')
        
        ax[1].imshow(od_mask, cmap='gray')
        ax[1].set_title("OD Mask")
        ax[1].axis('off')
        
        ax[2].imshow(cup_mask, cmap='gray')
        ax[2].set_title("Cup Mask")
        ax[2].axis('off')
        
        plt.savefig(f'aug_vis/aug_sample_{i}.png')
        plt.close()
        
    print("Verification complete.")

if __name__ == "__main__":
    verify_augmentation()
