
import time
import numpy as np

from torchmetrics.classification import BinaryAccuracy

from model import TheModel
from modular_model import ModularModel
from dataloader import get_chess_loader

from accuracy import *
import load_model as load

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0
EPOCHS = 200
EARLY_STOP = 7
PATH_MODEL = '../../results/models/'
PATH_CSV = '../../results/csv/'
PATH_BACKUP = '../../results/backup/'

CSV_HEADER = 'epoch,train_loss,test_loss,train_class,test_class, \
              train_square,test_square,train_move,test_move\n'

def train_modular(loss_fn, directory, name):
    # TODO: Implement a better way to include loaded models
     
    model, loaded = load.load_model(directory, name)


    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_loader, test_loader = get_chess_loader(
        #path='/media/mattias/DataDisk/data/thesis/data/',
        #path='../../games_10k',
        path='../../datasets/games_all',
        batch_size=BATCH_SIZE
    )

    model = model.to(DEVICE)

    if not loaded:
        with open(directory + '/' + name + '.csv', 'w') as results:
            results.write(CSV_HEADER)

    
    if loaded:
        epochs, loss_min = load.load_state(directory)
    else:
        epochs, loss_min = (0, -1)

    unimproved = 0

    # Metrics
    metrics = [CorrectDimension(), CorrectSquare(), CorrectMove()]
    
    print(f'Training \'{name}\' with \'{DEVICE}\'', end='\n\n')

    for epoch in range(epochs, EPOCHS):
        
        time_start = time.perf_counter()

        print(f'\nCurrent epoch: {epoch}')

        batch_losses = []
        batch_acc = [[], [], []]

        test_losses = []
        test_acc = [[], [], []]

        # The training loop
        for x, y in train_loader:
            
            model.train()

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

            model.eval()
           
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
            unimproved = 0
            # Save model
            torch.save(model.state_dict(), directory + '/' + name + '.pth')
            
            # Backup model, if interupted during first save this one shoud
            # remain intact
            torch.save(model.state_dict(), PATH_BACKUP + name + '.pth')
        else:
            if unimproved == EARLY_STOP:
                print("--- EARLY STOP ---")
                break
            unimproved += 1
    
        
        with open(directory + '/' + name + '.csv', 'a') as results:
            results.write(f'{epoch + 1}, \
                {epoch_train_loss},{epoch_test_loss}, \
                {epoch_train_acc[0]},{epoch_test_acc[0]}, \
                {epoch_train_acc[1]},{epoch_test_acc[1]}, \
                {epoch_train_acc[2]},{epoch_test_acc[2]}\n'
            )

        with open(directory + '/log.txt', 'w') as log:
            log.write(f'{epoch}\n{loss_min}')
            

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
    



