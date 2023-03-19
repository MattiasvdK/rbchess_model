import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))


# TODO Make a more adaptable version to be able to change the model shape
#      on the fly

class TheModel(nn.Module):
    
    def __init__(self):
        super(TheModel, self).__init__()
        
        self.darknet = self._create_conv_layers()
        self.fcl = self._create_fc_layers()
        
        self.task_one = self._create_task_layer()
        self.task_two = self._create_task_layer()
        self.task_three = self._create_task_layer()
        self.task_four = self._create_task_layer()


    def forward(self, x):
        x = self.darknet(x)
        x = self.fcl(x)
        
        return torch.stack(
            (self.task_one(x),   
            self.task_two(x),   
            self.task_three(x), 
            self.task_four(x)),
            dim=1
        )


    def _create_conv_layers(self):  # TODO how to define the channels
        return nn.Sequential(       # how to decide the nr of channels?
            CNNBlock(12, 18, kernel_size=4),
            # H = W = 8 - (4 - 1) = 5
            CNNBlock(18, 32, kernel_size=3),
            # 5 - (3 - 1) = 3
            CNNBlock(32, 64, kernel_size=2),
            # H = W = 3 - (2 - 1) = 2
            CNNBlock(64, 96, kernel_size=1),
            # H = W = 2 - (1 - 1) = 2
        )

    def _create_fc_layers(self):
        return nn.Sequential(
            nn.Flatten(),           # 2 * 2 * Channels = 2 * 2 * 96 = 384
            nn.Linear(384, 256),    # TODO calculate CNN output dimensions
            nn.LeakyReLU(0.1),  
        )

    def _create_task_layer(self):
        return nn.Sequential(
            nn.Linear(256, 8),
            nn.Softmax(dim=1)       #This can probably be made neater
        )


