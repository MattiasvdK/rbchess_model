import numpy as np
from torch.utils.data import DataLoader, Dataset


class ChessDataset(Dataset):
    
    """Class to load and format the training data for the CNN.

    Class is a generator for the training data and is used to load the
    data from the directories.
    The class also formats the data into a matrix to represent the image
    and will generate both the training image of shape (8, 8, 12) and
    its labels, i.e. the chosen piece and the square to move it to.
    
    """

    def __init__(self, samples):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pass



