import time
import numpy as np
import torch
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOPPING = 10

def train_model(
        *,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        learning_rate: float,
        loaders: tuple,
        metric_names: tuple,
        metrics: dict,
        directory: str,
        name: str,
        epochs=200,
        loss_min=-1,
        unimproved=0,
        passed_epochs=0,
        ):
    """
    Train a model.

    Args:
        model (torch.nn.Module): The model to train.
        loss_fn (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        loaders (tuple): The data loaders.
        metric_names (tuple): The names of the metrics.
        metrics (dict): The metrics.
        epochs (int): The number of epochs.
        directory (str): The directory to save the model.
        name (str): The name of the model.
        loss_min (float): The minimum loss.
        unimproved (int): The number of epochs without improvement.
        passed_epochs (int): The number of epochs already passed.
    """

    csv_header = ['epoch', 'train_loss', 'test_loss']
    for metric in metric_names:
        csv_header.append(f'train_{metric}')
        csv_header.append(f'test_{metric}')

    train_loader, test_loader = loaders

    loss_fn = loss_fn.to(DEVICE)

    print(f'Training \'{name}\' with \'{DEVICE}\'', end='\n\n')

    # Write csv header if file does not exist
    csv_path = directory + '/' + name + '.csv'

    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as results:
            results.write(','.join(csv_header) + '\n')

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    model = model.to(DEVICE)


    for epoch in range(passed_epochs, epochs):
        
        loss_train = []
        loss_test = []

        # Initialize metric lists
        acc_train = []
        acc_test = []
        for metric in metric_names:
            acc_train.append([])
            acc_test.append([])
        
        time_start = time.perf_counter()

        # Train
        model.train()
        for batch in train_loader:

            optimizer.zero_grad()
            
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            #print(y_pred[0][0])
            #print(y_pred[0][1])

            loss_train.append(loss.item())
            for idx, metric in enumerate(metric_names):
                acc_train[idx].append(metrics[metric](y_pred, y))
        

        time_train = time.perf_counter()

        # Test
        model.eval()
        with torch.no_grad():
            for batch in test_loader:

                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                loss_test.append(loss.item())
                for idx, metric in enumerate(metric_names):
                    acc_test[idx].append(metrics[metric](y_pred, y))

        # Calculate mean loss and metrics
        epoch_loss_train = np.mean(loss_train)
        epoch_loss_test = np.mean(loss_test)

        epoch_acc_train = []
        epoch_acc_test = []
        for idx, metric in enumerate(metric_names):
            epoch_acc_train.append(np.mean(acc_train[idx]))
            epoch_acc_test.append(np.mean(acc_test[idx]))

        # Stop if there is no more improvement on the test loss
        if epoch_loss_test < loss_min or loss_min == -1:
            loss_min = epoch_loss_test
            unimproved = 0
            torch.save(model.state_dict(), directory + '/best_' + name + '.pth')
        else:
            unimproved += 1
            if unimproved == EARLY_STOPPING:
                print('Early stopping')
                break
        
        # Write results to csv
        with open(csv_path, 'a') as results:
            results.write(f'{epoch},{epoch_loss_train},{epoch_loss_test}')
            for metric in metric_names:
                results.write(f',{epoch_acc_train[idx]},{epoch_acc_test[idx]}')
            results.write('\n')

        # Save model and progress
        torch.save(model.state_dict(), directory + '/' + name + '.pth')
        torch.save({
            'epoch': epoch,
            'loss_min': loss_min,
            'unimproved': unimproved,
        }, directory + '/' + name + '_progress.pth')
        
        # Print results
        print(f'Epoch {epoch}')
        print(f'\tTrain loss: {epoch_loss_train}')
        print(f'\tTest loss: {epoch_loss_test}')
        for idx, metric in enumerate(metric_names):
            print(f'\tTrain {metric}: {epoch_acc_train[idx]}')
            print(f'\tTest {metric}: {epoch_acc_test[idx]}')
        print('Time: {:.2f}s'.format(time.perf_counter() - time_start))
        print('Time (train): {:.2f}s'.format(time_train - time_start))




    

