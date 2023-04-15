import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))
    

class UNetModel(nn.Module):
    '''
    UNet model for supervised learning
    '''

    def __init__(self):
        super(UNetModel, self).__init__()

        self.dim = 8
        self.channels = 12
        self.features = self.channels * self.dim ** 2

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_block = self._create_cnn_block(12, 64)
        self.down_block1 = self._create_cnn_block(64, 128)
        self.down_block2 = self._create_cnn_block(128, 256)
        
        self.upscale1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block1 = self._create_cnn_block(256, 128)

        self.upscale2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block2 = self._create_cnn_block(128, 64)

        self.out_block = self._create_out_block(64, 2)

    def forward(self, x):
        # Add skip connections

        # The encoder part
        x1 = self.in_block(x)
        x2 = self.maxpool(x1)
        x3 = self.down_block1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down_block2(x4)
        
        # The first step of the decoder
        x6 = self.upscale1(x5)
        x7 = torch.cat([x6, x3], dim=1) # Skip connection
        x8 = self.up_block1(x7)

        # The second step of the decoder
        x9 = self.upscale2(x8)
        x10 = torch.cat([x9, x1], dim=1) # Skip connection
        x11 = self.up_block2(x10)

        x12 = self.out_block(x11)

        x13 = F.softmax(x12.view(x12.size(0), x12.size(1), -1), dim=2).view(x12.size())

        return x13

    
    def _create_cnn_block(self, in_channels, out_channels):

        self.channels = out_channels
        self.features = self.channels * self.dim ** 2
        
        return nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size=3, padding=1),
            CNNBlock(out_channels, out_channels, kernel_size=3, padding=1)
        )
    

    # This makes no sense
    # The up and down scaling should happen independently
    # to be able to use skip connections
    
    def _create_out_block(self, in_channels, out_channels):
        self.channels = out_channels
        self.features = self.channels * self.dim ** 2

        # The output block is a 1x1 convolution
        # With softmax activation over the pixels
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

