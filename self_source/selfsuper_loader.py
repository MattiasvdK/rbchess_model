import torch
import os
from torch.utils.data import DataLoader, Dataset


def get_self_train_loaders(*, path: str, batch_size: int):
    """Creates the data generator for self supervision.
    
    Args:
        path: the directory where the data is stored.

    Returns:
        DataLoader: the generator that loads the data.

    """
    train_samples = [path + '/train/' + sample
                        for sample in os.listdir(path + '/train')]
    test_samples = [path + '/test/' + sample
                        for sample in os.listdir(path + '/test')]


    train_loader = DataLoader(
        dataset=SelfChessDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=SelfChessDataset(test_samples),
        batch_size=1,
        shuffle=True,
        drop_last=False
    )

    return train_loader, test_loader


def get_self_val_loaders(*, path: str):
    """Creates the data generator for self supervision.
    
    Args:
        path: the directory where the data is stored.

    Returns:
        DataLoader: the generator that loads the data.

    """
    samples = [path + '/val/' + sample 
               for sample in os.listdir(path + '/val')]


    val_loader = DataLoader(
        dataset=SelfChessDataset(samples),
        batch_size=1,
        shuffle=True,
        drop_last=False
    )

    return val_loader



class SelfChessDataset(Dataset):
    """Class to load and format the selfsupervised data for the CNN.

    Class is a generator for the training data and is used to load the
    data from the directories in samples.
    The class also formats the data into a tensor to represent the image
    and will generate both the training image of shape (8, 8, 12).
    The class will then create a jigsaw puzzle for the network to solve.
    The labels will be the index of the piece in the puzzle.

    The data is stored in binary files as a list of 64 bytes defining the
    value in the given square. The bytes are written in row major order.
    
    The class formats the data where the position in the list of bytes defines
    the value in the first two dimensions of the vector and the value the
    depth.
    """
    def __init__(self, samples):
        """Initializes the class.

        Args:
            samples: a list of paths to the data files.

        """
        self.samples = samples

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: the length of the dataset.

        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns the data at the given index.

        Args:
            idx: the index of the data to return.

        Returns:
            torch.Tensor: the data at the given index.
            torch.Tensor: the labels for the data at the given index.

        """

        x = torch.zeros(12, 8, 8)

        board = list(
            map(
                lambda x: x - 1 if x < 126 else x - 244,
                map(int, open(self.samples[idx] + '/board', 'rb').read())
            )
        )

        for idx, piece in enumerate(board):
            if piece != -1:
                try:
                    x[piece][idx // 8][idx % 8] = 1
                except IndexError as err:
                    print(f'--- ERROR: {err}, at index: {idx} ---')
        
        return self._permutation(x)
    
    def _permutation(self, board):
        """Creates a permutation of the board.

        Args:
            board: the chess board to permute.

        Returns:
            torch.Tensor: the permuted board.
            torch.Tensor: the permutation of the board. Functioning as the labels.

        """

        jigsaw = [(0, 0), (0, 1), (1, 0), (1, 1)]

        permutation = torch.randperm(4)

        permuted = torch.zeros_like(board)

        target = torch.zeros((4, 4))

        for idx, perm in enumerate(permutation):
            perm_x = jigsaw[idx][0] * 4
            perm_y = jigsaw[idx][1] * 4
            board_x = jigsaw[perm][0] * 4
            board_y = jigsaw[perm][1] * 4

            target[idx][perm] = 1

            permuted[:, perm_x:perm_x + 4, perm_y:perm_y + 4] = \
                board[:, board_x:board_x + 4, board_y:board_y + 4]
        
        return permuted, target