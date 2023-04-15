from train import train_model
from model import UNetModel
from dataloader import get_dataloaders
import torch

from metrics import *

def main():

    # Get dataloaders
    loaders = get_dataloaders(
        path='../../datasets/split_all/',
        batch_size=128
    )

    # Initialize model
    model = UNetModel()

    metrics = {
        'square': SquareAccuracy(),
        'move': MoveAccuracy(),
        'piece': PieceSelector()
    }

    # Train model
    train_model(
        model=model,
        loaders=loaders,
        learning_rate=1e-4,
        epochs=100,
        metric_names=['square', 'move', 'piece'],
        metrics=metrics,
        loss_fn=torch.nn.CrossEntropyLoss(),
        directory='../../results/unet/',
        name='unet'
    )

if __name__ == '__main__':
    main()