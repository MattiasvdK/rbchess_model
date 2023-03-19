import dataloader as dl
import numpy as np
from torch.nn import CrossEntropyLoss
import train
from accuracy import *
import sys


def run():
    if len(sys.argv) != 2:
        print('Please provide the model name')
        exit()

    train.train_model(CrossEntropyLoss(), sys.argv[1])

if __name__ == '__main__':
    run()

