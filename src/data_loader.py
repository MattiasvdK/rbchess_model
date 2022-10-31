import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


class ChessDataset(Dataset):
    """Class to load and format the training data for the CNN.

    Class is a generator for the training data and is used to load the
    data from the directories in samples.
    The class also formats the data into a tensor to represent the image
    and will generate both the training image of shape (8, 8, 12) and
    its labels, i.e. the chosen piece and the square to move it to.

    The data is stored in binary files as a list of 64 bytes defining the
    value in the given square. The bytes are written in row major order.
    
    The class formats the data where the position in the list of bytes defines
    the value in the first two dimensions of the vector and the value the
    depth.

    # TODO: Write an example
    
    
    Attributes:
       samples: A list of directories where the data is stored.

    """

    def __init__(self, samples: list):
        """Inits ChessDataset with samples."""
        self.samples = samples

    def __len__(self):
        """Define the length of the class."""
        return len(self.samples) 

    def __getitem__(self, idx: int) -> torch.tensor, torch.tensor:
        """Loads and formats the data stored in the samples.
        
        The function loads and formats the training data and labels.
        
        # TODO: Check if the explanation in the class should be here instead

        Args:
            idx: The index of the data to be loaded from the samples list.
        
        Returns:
            The input tensor and the target vector.
            The input vector has shape (8, 8, 12). The first two dimensions
            represents the chessboard positions, the depth represents the
            piece type.

        """
        
        sample = self.samples[idx]
        
        x = torch.zeros(8, 8, 12)
        
        board = list(map(int, open(sample + '/board', 'rb').read()))
         
        # TODO: Read data and process into tensor

        y = torch.tensor(map(int, open(sample + '/move', 'rb').read()))

        
        return x, y

    def __process(board: list, tensor: torch.tensor)
        """Maps a list of integers to a tensor.

        This function is a helper for __getitem__ to process the list of
        integers that is stored as data and map it to the tensor needed
        for the network to process.
        """
        # TODO: Implement
        pass

