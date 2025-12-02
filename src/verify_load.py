import torch
from model import NestedUNet
import os

def verify_load():
    final_state_dict_path = r'c:/Users/Admin/OneDrive/Desktop/Model/checkpoints/final_model.pt'
    final_full_model_path = r'c:/Users/Admin/OneDrive/Desktop/Model/checkpoints/final_model_full.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Verify state dict load
    print(f"Verifying state dict load from {final_state_dict_path}...")
    try:
        model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
        state_dict = torch.load(final_state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        print("SUCCESS: State dictionary loaded correctly.")
    except Exception as e:
        print(f"FAILURE: Could not load state dictionary. Error: {e}")

    # Verify full model load
    print(f"Verifying full model load from {final_full_model_path}...")
    try:
        model_full = torch.load(final_full_model_path, map_location=device, weights_only=False)
        model_full.eval()
        print("SUCCESS: Full model loaded correctly.")
    except Exception as e:
        print(f"FAILURE: Could not load full model. Error: {e}")

if __name__ == "__main__":
    verify_load()
