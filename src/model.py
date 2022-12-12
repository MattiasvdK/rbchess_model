import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))


class Softmax(nn.Module):
    
    def __init__(self):
        super(Softmax, self).__init__()
        self.sm = nn.Softmax(dim=1)

    # TODO make this work
    # ?? Implement softmax myself to reach subarrays of 8
    def forward(self, x):
        return x

# TODO Make a more adaptable version to be able to change the model shape
#      on the fly

class TheModel(nn.Module):
    
    def __init__(self):
        super(TheModel, self).__init__()
        
        self.darknet = self._create_conv_layers()
        self.fcl = self._create_fc_layers()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcl(x)
        #return self.fcl(torch.flatten(x, start_dim=1))


    def _create_conv_layers(self):  # TODO how to define the channels
        return nn.Sequential(       # how to decide the nr of channels?
            CNNBlock(12, 18, kernel_size=5),
            # H = W = 8 - (5 - 1) = 4
            CNNBlock(18, 32, kernel_size=3),
            # H = W = 4 - (3 - 1) = 2
            CNNBlock(32, 64, kernel_size=1),
            # H = W = 2 - (1 - 1) = 2
        )

    def _create_fc_layers(self):
        return nn.Sequential(
            nn.Flatten(),           # 2 * 2 * Channels = 2 * 2 * 64 = 256
            nn.Linear(256, 256),    # TODO calculate CNN output dimensions
            nn.LeakyReLU(0.1),  
            nn.Linear(256, 4 * 8),  # TODO see how to get 4 heads of 8 classes
            #Softmax(),             # 4-dimensional softmax
        )


