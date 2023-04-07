import torch
import torch.nn as nn
import json

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.leakyre = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))
    

class SelfModularModel(nn.Module):
    
    def __init__(self, model_specs):
        super(SelfModularModel, self).__init__()
        
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

        self.features_darknet = self.features

        self.features = 512
        self.fcl = nn.Linear(self.features_darknet, self.features)
        
        self.tasks_one = self._task_selfsuper()
        self.tasks_two = self._task_selfsuper()
        self.tasks_three = self._task_selfsuper()
        self.tasks_four = self._task_selfsuper()


    def forward(self, x):
        x = self.darknet(x)
        x = self.fcl(x)
        return torch.stack(
            (self.tasks_one(x),
            self.tasks_two(x),
            self.tasks_three(x),
            self.tasks_four(x)),
            dim=1
        )
    
    def switch_task(self, task):
        self.fcl = nn.Linear(self.features_darknet, self.features)

        self.tasks_one = self._task_move()
        self.tasks_two = self._task_move()
        self.tasks_three = self._task_move()
        self.tasks_four = self._task_move()
    
    def _process(self, layer):
        if layer['type'] == 'conv':
            return self._convolutional(layer)
        elif layer['type'] == 'maxpool':
            return self._maxpool(layer)
        elif layer['type'] == 'linear':
            return self._linear(layer)
        elif layer['type'] == 'flatten':
            return self._flatten(layer)
        elif layer['type'] == 'leakyrelu':
            return self._leakyrelu(layer)
        else:
            print('Unknown layer type: ' + layer['type'])
            return None
        
    def _convolutional(self, layer: dict):
        
        kernel_size = layer['kernel_size']
        self.dims = (self.dims + 2 * layer['padding']) - (kernel_size - 1)
        in_channels = self.channels
        self.channels = layer['out_channels']

        self.features = self.dims ** 2 * self.channels

        return CNNBlock(
            in_channels,
            self.channels,
            kernel_size=layer['kernel_size'],
            padding=layer['padding']
        )
    
    # Created by Copilot
    # TODO needs checking for correctness
    def _maxpool(self, layer: dict):
        self.dims = self.dims // layer['kernel_size']
        self.features = self.dims ** 2 * self.channels
        return nn.MaxPool2d(layer['kernel_size'])
    
    def _linear(self, layer: dict):
        in_features = self.features
        self.features = layer['out_features']
        return nn.Linear(in_features, self.features)
    
    def _create_leakyrelu(self, layer: dict):
        return nn.LeakyReLU(layer['leak'])

    # This is not correct
    # TODO make dataloader properly to output one hot encoded class labels
    # for the selfsupervised task
    def _task_selfsuper(self):
        return nn.Sequential(
            nn.Linear(self.features, 4),
            nn.Softmax(dim=1)
        )
    
    def _task_move(self):
        return nn.Sequential(
            nn.Linear(self.features, 8),
            nn.Softmax(dim=1)
        )

