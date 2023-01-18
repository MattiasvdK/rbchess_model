import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy

from model import TheModel
from dataloader import get_chess_loader

from accuracy import *


LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0
EPOCHS = 200


def train_model(loss_fn, metric_fn):
    
    print(f'Using: {DEVICE} as training processor')

    model = TheModel()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_loader, test_loader = get_chess_loader(
        path='/media/mattias/DataDisk/data/thesis/data/',
        #path='../../test2',
        batch_size=BATCH_SIZE
    )

    model = model.to(DEVICE)

    writer = SummaryWriter()

    # Metrics

    metrics = [CorrectDimension(), CorrectSquare(), CorrectMove()]

    for epoch in range(EPOCHS):
        
        print(f'\nCurrent epoch: {epoch}')

        batch_losses = []
        batch_acc = [[], [], []]

        test_losses = []
        test_acc = [[], [], []]

        preds = []

        # The training loop
        for x, y, in train_loader:

            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)

            for idx, metric in enumerate(metrics):
                accuracy = metric(predictions, y)
                batch_acc[idx].append(accuracy.item())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            # batch_acc.append(accuracy.item())

            preds = predictions


        # The test loop
        for x, y, in test_loader:

            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)

            for idx, metric in enumerate(metrics):
                accuracy = metric(predictions, y)
                test_acc[idx].append(accuracy.item())


            test_losses.append(loss.item())
            #test_acc.append(accuracy.item())

            preds = predictions

        epoch_train_acc = np.mean(batch_acc, 1)  # Keep dimension over accuracy
        epoch_test_acc = np.mean(test_acc, 1)    # Keep dimension over accuracy


        epoch_train_loss = np.mean(batch_losses)
        #epoch_train_acc = np.mean(batch_acc)

        epoch_test_loss = np.mean(test_losses)
        #epoch_test_acc = np.mean(test_acc)

        writer.add_scalars(
            "Loss",
            {"Training loss": epoch_train_loss,
             "Test loss": epoch_test_loss},
            epoch
        )

        writer.add_scalars(
            "Class Accuracy",
            {"Training accuracy": epoch_train_acc[0],
             "Test accuracy": epoch_test_acc[0]},
            epoch
        )

        writer.add_scalars(
            "Square Accuracy",
            {"Training accuracy": epoch_train_acc[1],
             "Test accuracy": epoch_test_acc[1]},
            epoch
        )

        writer.add_scalars(
            "Move Accuracy",
            {"Training accuracy": epoch_train_acc[2],
             "Test accuracy": epoch_test_acc[2]},
            epoch
        )
        
    
        print(f"training loss: {epoch_train_loss}, ",
              f"training accuracy: {epoch_train_acc[1]}, ",
              f"{epoch}"
        )
        print(f"test loss: {epoch_test_loss}, ",
              f"test accuracy: {epoch_test_acc[1]}, ",
              f"{epoch}"
        )

    writer.close()





