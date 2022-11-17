import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from model import TheModel
from dataloader import get_chess_loader

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0
EPOCHS = 20


def train_model(loss_fn):
    model = TheModel()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_loader = get_chess_loader(
        path='../../test/all',
        batch_size=BATCH_SIZE
    )

    model = model.to(DEVICE)

    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        batch_losses = []

        for x, y, in train_loader:

            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        writer.add_scalar("training loss", epoch_loss, epoch)
        print(f'training loss: {epoch_loss}, {epoch}')





