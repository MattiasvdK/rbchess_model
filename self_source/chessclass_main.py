from chessclass_train import train_classification
from selfsuper_model import  SelfModularModel
from accuracy import *
from torch.nn import CrossEntropyLoss


def train():
    model = SelfModularModel(
        #'../../results/models/selfsupertest/selfsupertest.json'
        '../../results/models/classsupertest/classsupertest.json'
    )
    '''
    model.load_state_dict(
        torch.load(
            '../../results/models/selfsupertest/selfsupertest_self.pth'
        )
    )
    '''
    model.switch_task()
    train_classification(
        model=model,
        loss_fn=CrossEntropyLoss(),
        acc_fn=[CorrectDimension(), CorrectSquare(), CorrectMove()],
        directory='../../results/models/classsupertest/',
        name='classsupertest',
    )


if __name__ == '__main__':
    train()

    