import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import DrishtiDataset
from model import NestedUNet
from utils import DiceLoss, iou_score
import os

def train_debug():
    # Hyperparameters
    batch_size = 2
    lr = 1e-4
    epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Training-20211018T055246Z-001"
    dataset = DrishtiDataset(root_dir, mode='train')
    # Use only 4 samples for debug
    dataset = Subset(dataset, range(4))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting debug training...")
    for epoch in range(epochs):
        model.train()
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Step {i+1}, Loss: {loss.item():.4f}")
            
    print("Debug training complete.")

if __name__ == "__main__":
    train_debug()
