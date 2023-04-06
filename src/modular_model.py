import torch
import torch.nn as nn
import json
import numpy as np



class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))




class ModularModel(nn.Module):

    def __init__(self, model_specs):
        super(ModularModel, self).__init__()
        
        self.features = 8 * 8 * 12
        self.dims = 8
        self.channels = 12

        try:
            with open(model_specs) as specs:
                data = json.load(specs)
        except FileNotFoundError:
            print('--ERROR-- FILE NOT FOUND')
            print('Could not initialize model')
            return

        layers = []

        for layer in data['layers']:
            layers.append(self._process(layer))
            
        self.darknet = nn.Sequential(*layers)
        
        self.tasks_one = self._create_task()
        self.tasks_two = self._create_task()
        self.tasks_three = self._create_task()
        self.tasks_four = self._create_task()



    def forward(self, x):
        x = self.darknet(x)
        return torch.stack(
            (self.tasks_one(x),
            self.tasks_two(x),
            self.tasks_three(x),
            self.tasks_four(x)),
            dim=1
        )

    def _process(self, layer: dict):
        
        if layer['type'] == 'conv':
            return self._create_conv(layer)
        elif layer['type'] == 'linear':
            return self._create_linear(layer)
        elif layer['type'] == 'leakyrelu':
            return self._create_leakyrelu(layer)
        elif layer['type'] == 'flatten':
            return nn.Flatten()

        return None


    def _create_conv(self, layer: dict):
        
        try:
            padding = layer['padding']
        except KeyError:
            padding = 0

        self.dims = (self.dims + 2 * padding) - (layer['kernel_size'] - 1)
        
        in_channels = self.channels
        self.channels = layer['out_channels']

        self.features = self.dims ** 2 * self.channels
        return CNNBlock(in_channels,
                        self.channels,
                        kernel_size=layer['kernel_size'],
                        padding=padding
        )


    def _create_linear(self, layer: dict):
        in_features = self.features
        self.features = layer['out_features']
        return nn.Linear(in_features, self.features)


    def _create_leakyrelu(self, layer: dict):
        return nn.LeakyReLU(layer['leak'])


    def _create_task(self):
        return nn.Sequential(
            nn.Linear(self.features, 8),
            nn.Softmax(dim=1)       #This can probably be made neater
        )




