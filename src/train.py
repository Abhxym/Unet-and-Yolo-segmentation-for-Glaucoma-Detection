import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import DrishtiDataset
from model import NestedUNet
from utils import DiceLoss, iou_score
import os
import time

def train():
    # Hyperparameters
    batch_size = 2
    lr = 1e-4
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Training-20211018T055246Z-001"
    dataset = DrishtiDataset(root_dir, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    best_loss = float('inf')
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Validation / Visualization Data
    test_root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Test-20211018T060000Z-001"
    test_dataset = DrishtiDataset(test_root_dir, mode='test')
    # Use a fixed subset for visualization to see progress on same images
    vis_loader = DataLoader(Subset(test_dataset, range(3)), batch_size=1, shuffle=False)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.exists('training_vis'):
        os.makedirs('training_vis')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        
        start_time = time.time()
        
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_iou += iou_score(outputs, masks).item()
            
            if (i+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        avg_iou = epoch_iou / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{epochs}] Completed. Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}, Time: {time.time()-start_time:.2f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print("Saved best model.")

        # Visualize progress every epoch
        model.eval()
        with torch.no_grad():
            for i, (image, mask) in enumerate(vis_loader):
                image = image.to(device)
                output = model(image)
                output = torch.sigmoid(output)
                output = (output > 0.5).float()
                
                # Plot
                img_np = image[0].permute(1, 2, 0).cpu().numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                
                pred_od = output[0, 0].cpu().numpy()
                pred_cup = output[0, 1].cpu().numpy()
                
                fig, ax = plt.subplots(1, 3, figsize=(10, 3))
                ax[0].imshow(img_np)
                ax[0].set_title("Image")
                ax[0].axis('off')
                
                ax[1].imshow(img_np)
                ax[1].imshow(pred_od, alpha=0.5, cmap='Greens')
                ax[1].set_title(f"Epoch {epoch+1} OD")
                ax[1].axis('off')
                
                ax[2].imshow(img_np)
                ax[2].imshow(pred_cup, alpha=0.5, cmap='Reds')
                ax[2].set_title(f"Epoch {epoch+1} Cup")
                ax[2].axis('off')
                
                plt.savefig(f'training_vis/epoch_{epoch+1}_sample_{i}.png')
                plt.close()

            
    print("Training Complete.")

if __name__ == "__main__":
    train()
