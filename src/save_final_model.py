import torch
import os
from model import NestedUNet

def save_final_model():
    # Define paths
    checkpoint_path = r'c:/Users/Admin/OneDrive/Desktop/Model/checkpoints/best_model.pth'
    final_state_dict_path = r'c:/Users/Admin/OneDrive/Desktop/Model/checkpoints/final_model.pt'
    final_full_model_path = r'c:/Users/Admin/OneDrive/Desktop/Model/checkpoints/final_model_full.pt'

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
    
    # Load state dict
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Successfully loaded state dictionary from checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model = model.to(device)
    model.eval()

    # Save state dictionary (recommended)
    try:
        torch.save(model.state_dict(), final_state_dict_path)
        print(f"Saved model state dictionary to: {final_state_dict_path}")
    except Exception as e:
        print(f"Error saving state dictionary: {e}")

    # Save full model (optional, but requested "overall .pt file")
    try:
        torch.save(model, final_full_model_path)
        print(f"Saved full model object to: {final_full_model_path}")
    except Exception as e:
        print(f"Error saving full model: {e}")

if __name__ == "__main__":
    save_final_model()
