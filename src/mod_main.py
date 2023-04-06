import train
from torch.nn import CrossEntropyLoss
import sys
import mod_train


def run():
    mod_train.train_modular(CrossEntropyLoss(), sys.argv[1], sys.argv[2]) 

if __name__ == '__main__':
    run()




