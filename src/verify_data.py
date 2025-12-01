import torch
from dataset import DrishtiDataset
import os

def verify():
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Training-20211018T055246Z-001"
    dataset = DrishtiDataset(root_dir, mode='train')
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
        
        # Check Test set
        test_root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Test-20211018T060000Z-001"
        test_dataset = DrishtiDataset(test_root_dir, mode='test')
        print(f"Test Dataset length: {len(test_dataset)}")
        if len(test_dataset) > 0:
             img, mask = test_dataset[0]
             print(f"Test Image shape: {img.shape}")

if __name__ == "__main__":
    verify()
