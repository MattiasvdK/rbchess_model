import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import BinaryAccuracy

from model import TheModel
from dataloader import get_chess_loader


LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 20


def train_model(loss_fn, metric_fn):
    model = TheModel()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_loader, test_loader = get_chess_loader(
        path='../../test2',
        batch_size=BATCH_SIZE
    )

    model = model.to(DEVICE)

    writer = SummaryWriter()
    metric = BinaryAccuracy().to(DEVICE)    # This is not really it..

    for epoch in range(EPOCHS):
        batch_losses = []
        batch_acc = []

        test_losses = []
        test_acc = []

        preds = []

        # The training loop
        for x, y, in train_loader:

            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)
            accuracy = metric_fn(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_acc.append(accuracy.item())

            preds = predictions


        # The test loop
        for x, y, in test_loader:

            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)
            accuracy = metric_fn(predictions, y)

            test_losses.append(loss.item())
            test_acc.append(accuracy.item())

            preds = predictions
        

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        writer.add_scalar(
            "training loss, accuracy",
            epoch_loss,
            epoch_acc,
            epoch
        )

        test_loss = np.mean(test_losses)
        test_accuracy = np.mean(test_acc)   # Bad naming
        
        print('\n', preds)
        print(f"training loss: {epoch_loss}, ",
              f"training accuracy: {epoch_acc}, ",
              f"{epoch}"
        )
        print(f"test loss: {test_loss}, ",
              f"test accuracy: {test_accuracy}, ",
              f"{epoch}"
        )





