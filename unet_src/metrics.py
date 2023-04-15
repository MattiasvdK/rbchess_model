from torchmetrics import Metric
import torch


class SquareAccuracy(Metric):
    '''Calculates hitrate of correct square.

    '''

    full_state_update: bool = True

    is_differentiable: bool = False

    higher_is_better: bool = True
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))


    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.shape == targets.shape

        # It's not pretty, but I think it works
        for preds, targs in zip(predictions, targets):
            # we are now at batch level for tasks
            for pred, targ in zip(preds, targs):
                if torch.argmax(pred) == torch.argmax(targ):
                    self.correct += 1
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total
    

class MoveAccuracy(Metric):
    '''Calculates hitrate of correct move.

    '''

    full_state_update: bool = True

    is_differentiable: bool = False

    higher_is_better: bool = True
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))


    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.shape == targets.shape

        # It's not pretty, but I think it works
        for preds, targs in zip(predictions, targets):
            # we are now at batch level for tasks
            if torch.argmax(preds[0]) == torch.argmax(targs[0]) \
            and torch.argmax(preds[1]) == torch.argmax(targs[1]):
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total
    

class PieceSelector(Metric):
    full_state_update: bool = True

    is_differentiable: bool = False

    higher_is_better: bool = True
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))


    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.shape == targets.shape

        # It's not pretty, but I think it works
        for preds, targs in zip(predictions, targets):
            # we are now at batch level for tasks
            if torch.argmax(preds[0]) == torch.argmax(targs[0]):
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total
    

