import time
import torch
import numpy as np

from selfsuper_loader import get_self_train_loaders, get_self_val_loaders

import os

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0
EPOCHS = 200
EARLY_STOP = 10 # was 7

CSV_HEADER = 'epoch,train_loss,test_loss,train_acc,test_acc\n'


def train_self_supervised(*, model, loss_fn, acc_fn, directory, name, epochs=0):
    """Trains the model on the self supervised data.
    
    Args:
        model: the model to train.
        loss_fn: the loss function to use.
        acc_fn: the accuracy function to use.
        directory: the directory where the model and csv file are stored.
        name: the name of the model.
        epochs: the number of epochs trained so far
    
    Returns:
        None
    """

    

    csv_path = directory + '/' + name + '.csv'
    model_path = directory + '/' + name + '.pth'

    model = model.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)
    acc_fn = acc_fn.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Get the data loaders
    train_loader, test_loader = get_self_train_loaders(
        path='../../datasets/split_all/',
        batch_size=64
    )

    # Check if csv file exists
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as results:
            results.write(CSV_HEADER)

    loss_min = -1

    print(f'Training \'{name}\' with \'{DEVICE}\'', end='\n\n')

    for epoch in range(epochs, EPOCHS):
        
        loss_train = []
        acc_train = []

        loss_test = []
        acc_test = []

        start_time = time.perf_counter()
        
        model.train()
        for x, y in train_loader:


            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)

            accuracy = acc_fn(predictions, y)

            loss = loss_fn(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())
            acc_train.append(accuracy.item())

        train_time = time.perf_counter() - start_time

        # Testing
        with torch.no_grad():
            model.eval()
            for x, y in test_loader:
                

                x, y = x.to(DEVICE), y.to(DEVICE)
                predictions = model(x)

                accuracy = acc_fn(predictions, y)
                loss = loss_fn(predictions, y)

                loss_test.append(loss.item())
                acc_test.append(accuracy.item())

        epoch_train_acc = np.mean(acc_train)
        epoch_test_acc = np.mean(acc_test)

        epoch_train_loss = np.mean(loss_train)
        epoch_test_loss = np.mean(loss_test)
    
        # Stop if there is no more improvement on the test loss
        if epoch_test_loss < loss_min or loss_min < 0:
            loss_min = epoch_test_loss
            unimproved = 0
            # Save model
            torch.save(model.state_dict(), model_path)
        elif unimproved >= EARLY_STOP:
            print('Early stopping')
            break
        else:
            unimproved += 1

        # Save results
        with open(csv_path, 'a') as results:
            results.write(f'{epoch},\
                          {epoch_train_loss},{epoch_test_loss},\
                          {epoch_train_acc},{epoch_test_acc}\n'
                        )
        
        print(f'Epoch: {epoch}',
                f'Train loss: {epoch_train_loss:.4f}',
                f'Train acc: {epoch_train_acc:.4f}\n',
                f'Test loss : {epoch_test_loss:.4f}',
                f'Test acc : {epoch_test_acc:.4f}',
                f'Epoch time: {time.perf_counter() - start_time:.4f}',
                f'of which training: {train_time:.4f}',
        )


        
        
        
