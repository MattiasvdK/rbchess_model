import dataloader as dl
import numpy as np
from torch.nn import CrossEntropyLoss
import train
from accuracy import *

def run():
    train.train_model(CrossEntropyLoss())

if __name__ == '__main__':
    run()

