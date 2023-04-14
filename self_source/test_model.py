from selfsuper_loader import get_self_train_loaders
from selfsuper_model import SelfModularModel
from selfsuper_train import train_self_supervised
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy



# Seems to work

def test():
    model = SelfModularModel('../../results/models/selfsupertest/selfsupertest.json')
    train_self_supervised(
        model=model,
        loss_fn=CrossEntropyLoss(),
        acc_fn=MulticlassAccuracy(num_classes=4),
        directory='../../results/models/selflarger',
        name='selflarger',
    )

if __name__ == '__main__':
    test()
