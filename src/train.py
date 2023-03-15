import numpy as np
import torch
import time

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
EARLY_STOP = 7


def train_model(loss_fn):

    model = TheModel()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_loader, test_loader = get_chess_loader(
        #path='/media/mattias/DataDisk/data/thesis/data/',
        path='../../games_10k',
        #path='../../data/games',
        batch_size=BATCH_SIZE
    )

    model = model.to(DEVICE)

    writer = SummaryWriter()

    # Metrics

    loss_min = -1
    unimproved = 0

    metrics = [CorrectDimension(), CorrectSquare(), CorrectMove()]

    for epoch in range(EPOCHS):
        
        time_start = time.perf_counter()

        print(f'\nCurrent epoch: {epoch}')

        batch_losses = []
        batch_acc = [[], [], []]

        test_losses = []
        test_acc = [[], [], []]

        # The training loop
        for x, y in train_loader:


            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)

            for idx, metric in enumerate(metrics):  
                accuracy = metric(predictions, y)
                batch_acc[idx].append(accuracy.item())
            

            loss = loss_fn(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        time_train = time.perf_counter()
        
        # The test loop
        for x, y in test_loader:

            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)
            

            for idx, metric in enumerate(metrics):
                accuracy = metric(predictions, y)
                test_acc[idx].append(accuracy.item())

            loss = loss_fn(predictions, y)
            
            test_losses.append(loss.item())

        epoch_train_acc = np.mean(batch_acc, 1)  # Keep dimension over accuracy
        epoch_test_acc = np.mean(test_acc, 1)    # Keep dimension over accuracy


        epoch_train_loss = np.mean(batch_losses)
        epoch_test_loss = np.mean(test_losses)
        

        # Stop if there is no more improvement on the test loss
        if epoch_test_loss < loss_min or loss_min < 0:
            loss_min = epoch_test_loss
        else:
            if unimproved == EARLY_STOP:
                print("--- EARLY STOP ---")
                break
            unimproved += 1
        
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
        print(f"Epoch time: {time.perf_counter() - time_start}",
              f"Of which training: {time_train - time_start}"
        )

    writer.close()





