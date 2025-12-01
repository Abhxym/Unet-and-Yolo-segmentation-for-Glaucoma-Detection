import torch
from model import NestedUNet

def verify():
    model = NestedUNet(num_classes=2, input_channels=3, deep_supervision=False)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    if y.shape == (1, 2, 512, 512):
        print("Model verification successful!")
    else:
        print("Model verification failed!")

if __name__ == "__main__":
    verify()
