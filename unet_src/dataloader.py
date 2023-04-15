from torch.utils.data import Dataset, DataLoader
import torch
import os

def get_dataloaders(*, path: str, batch_size: int):
    """Creates the data generator.
    
    Args:
        path: the directory where the data is stored.
        batch_size: the batch sizes to load the data in.

    Returns:
        train_loader: the generator that loads the training data.
        test_loader: the generator that loads the test data.

    """
    train_samples = [path + '/train/' + sample \
                        for sample in os.listdir(path + '/train')]
    test_samples = [path + '/test/' + sample
                        for sample in os.listdir(path + '/test')]


    train_set = ChessDataset(train_samples)
    test_set = ChessDataset(test_samples)


    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,       # Or should this keep the same batch_size?
        shuffle=True,
        drop_last=False
    )

    return train_loader, test_loader


class ChessDataset(Dataset):
    """The dataset for the classification task."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

         # Channels, height, width
        x = torch.zeros(12, 8, 8)
        
        # x is lowered by one to map the pieces to the depth of the tensor.
        # opposing colour is stored as negative numbers. To map those to the
        # to the correct position, 244 is subtracted. Putting the king at
        # depth 6 and the pawns at depth 12.

        board = list(
            map(
                lambda x: x - 1 if x < 126 else x - 244,
                map(int, open(sample + '/board', 'rb').read())
            )
        )
         
        for idx, piece in enumerate(board):
            if piece != -1:
                try:
                    x[piece][idx // 8][idx % 8] = 1
                except IndexError as err:
                    print(f'--- ERROR: {err}, at index: {idx} ---')

        data = torch.tensor(list(map(int, open(sample + '/move', 'rb').read())))

        # y is a one-hot encoded tensor.        
        y = torch.zeros((2, 8, 8))
        
        # The first two numbers are the start position, the last two are the
        # end position.
        try:
            y[0][data[0]][data[1]] = 1
        except IndexError as err:
            print(f'--- ERROR: {err}, at index: {idx}, src square ---')
        
        try:
            y[1][data[2]][data[3]] = 1
        except IndexError as err:
            print(f'--- ERROR: {err}, at index: {idx}, dst square ---')

        return x, y

