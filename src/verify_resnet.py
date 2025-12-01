import torch
from model import NestedUNetResNet

def verify_resnet_model():
    model = NestedUNetResNet(num_classes=2, input_channels=3, deep_supervision=False)
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    if output.shape == (1, 2, 512, 512):
        print("Verification Successful: Output shape matches expected.")
    else:
        print("Verification Failed: Output shape mismatch.")

if __name__ == "__main__":
    verify_resnet_model()
