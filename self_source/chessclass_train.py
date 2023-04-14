# Training function for classification model

import time
import torch
import numpy as np
import os

# Needs dataloader
from chessclass_loader import get_classification_loaders

BATCH_SIZE = 64
EARLY_STOP = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0
EPOCHS = 200

CSV_HEADER = 'epoch,train_loss,test_loss,train_acc_class,test_acc_class,\
              train_acc_square,test_acc_square,\
              train_acc_move,test_acc_move\n'

def train_classification(*, model, loss_fn, acc_fn, directory, name, epochs=0):
    loss_min = -1
    unimproved = 0

    model = model.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)

    # There are more acc functions that need to be used
    acc_fn[0] = acc_fn[0].to(DEVICE)
    acc_fn[1] = acc_fn[1].to(DEVICE)
    acc_fn[2] = acc_fn[2].to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Get the data loaders

    train_loader, test_loader = get_classification_loaders(
        path='../../datasets/split_all/',
        batch_size=BATCH_SIZE
    )

    csv_path = directory + '/' + name + '.csv'
    model_path = directory + '/' + name + '.pth'

    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as results:
            results.write(CSV_HEADER)
    
    print(f'Training \'{name}\' with \'{DEVICE}\'', end='\n\n')

    for epoch in range(epochs, EPOCHS):
        
        loss_train = []
        acc_train = [[], [], []]
        loss_test = []
        acc_test = [[], [], []]

        start_time  = time.perf_counter()

        # Training
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            
            for idx, fn in enumerate(acc_fn):
                acc_train[idx].append(fn(y_pred, y).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

        train_time = time.perf_counter() - start_time

        # Testing
        with torch.no_grad():
            model.eval()
            for x, y in test_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                y_pred = model(x)

                loss = loss_fn(y_pred, y)

                for idx, fn in enumerate(acc_fn):
                    acc_test[idx].append(fn(y_pred, y).item())

                loss_test.append(loss.item())

        epoch_train_loss = np.mean(loss_train)
        epoch_train_acc = [np.mean(acc) for acc in acc_train]
        epoch_test_loss = np.mean(loss_test)
        epoch_test_acc = [np.mean(acc) for acc in acc_test]

        # Save the model if it is the best so far
        if epoch_test_loss < loss_min or loss_min == -1:
            loss_min = epoch_test_loss
            unimproved = 0
            torch.save(model.state_dict(), model_path)
        elif unimproved >= EARLY_STOP:
            print(f'Early stopping at epoch {epoch}')
            break
        else:
            unimproved += 1
        
        # Save the results
        with open(csv_path, 'a') as results:
            results.write(f'{epoch},\
                          {epoch_train_loss},{epoch_test_loss},\
                          {epoch_train_acc[0]},{epoch_test_acc[0]},\
                          {epoch_train_acc[1]},{epoch_test_acc[1]},\
                          {epoch_train_acc[2]},{epoch_test_acc[2]}\n'
                        )
        
        print(f'Epoch: {epoch}')
        print(
            f'Train loss: {epoch_train_loss}',
            f'Train acc: {epoch_train_acc[1]}'
        )
        print(
            f'Test loss: {epoch_test_loss}',
            f'Test acc: {epoch_test_acc[1]}'
        )
        print(
            f'Epoch time: {time.perf_counter() - start_time:.4f}',
            f'of which training: {train_time:.4f}',
        )
        

