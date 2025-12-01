import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class NestedUNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, deep_supervision=False):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

import torchvision

class NestedUNetResNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, deep_supervision=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # ResNet Backbone
        resnet = torchvision.models.resnet34(pretrained=True)
        
        # Encoder Layers
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64, H/2
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1 # 64, H/4
        self.enc3 = resnet.layer2 # 128, H/8
        self.enc4 = resnet.layer3 # 256, H/16
        
        # Filters: [x0_0, x1_0, x2_0, x3_0, x4_0]
        # x0_0 is custom full-res layer
        nb_filter = [64, 64, 64, 128, 256]
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Level 0 (Full Res)
        self.conv0_0 = ConvBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        # Level 1 (H/2)
        # enc1 gives 64
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        
        # Level 2 (H/4)
        # enc2 gives 64
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        
        # Level 3 (H/8)
        # enc3 gives 128
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        # Level 4 (H/16)
        # enc4 gives 256
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # Encoder
        x0_0 = self.conv0_0(input)
        
        x1_0 = self.enc1(input)
        
        x2_0 = self.enc2(self.pool(x1_0))
        x3_0 = self.enc3(x2_0)
        x4_0 = self.enc4(x3_0)
        
        # Decoder
        # Level 0_1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        # Level 1_1
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # Level 0_2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        # Level 2_1
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # Level 1_2
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # Level 0_3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        # Level 3_1
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # Level 2_2
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # Level 1_3
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # Level 0_4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output
