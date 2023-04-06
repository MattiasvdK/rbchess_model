import torch
import os
import numpy as np

from modular_model import ModularModel


def load_model(directory, name):
    path_model = directory + '/' + name + '.pth'
    path_specs = directory + '/' + name + '.json'
    
    model = ModularModel(path_specs)
    loaded = False
    
    if os.path.isfile(path_model):
        model.load_state_dict(torch.load(path_model))
        model.eval()        # Return model for inference
        loaded = True

    return model, loaded


def load_state(directory):
    lines = open(directory + '/log.txt', 'r').readlines()

    return int(lines[0]), float(lines[1])



