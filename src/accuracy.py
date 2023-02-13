from torchmetrics import Metric
import torch

'''
Note to self: 
More heads should make calculating accuracy easier. Instead of going over
the array we can just compare the different heads to the target arrays,
this should result in direct comparisons and nicer loops.

'''


class CorrectDimension(Metric):
    '''Calculates hitrate of correct dimension (row or col).
    
    '''
    
    full_state_update: bool = True

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



class CorrectSquare(Metric):
    '''Calculates hitrate of correct square.
    
    '''

    full_state_update: bool = True
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))


    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.shape == targets.shape

        # It's not pretty, but I think it works
        for preds, targs in zip(predictions, targets):
            count = 0
            second = False
            for pred, targ in zip(preds, targs):
                # Check if the coordinate is predicted correctly
                # and count
                if torch.argmax(pred) == torch.argmax(targ):
                    count += 1
                
                # If this is the second coordinate, check if both were
                # correct to see if the square was correct.
                if second:
                    self.correct += 1 if count == 2 else 0
                    self.total += 1
                    count = 0
                
                second = not second

    def compute(self):
        return self.correct.float() / self.total



class CorrectMove(Metric):
    '''Calculates hitrate of complete move.
    
    '''

    full_state_update: bool = True
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))
    
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.shape == targets.shape

        # It's not pretty, but I think it works
        for preds, targs in zip(predictions, targets):
            count = 0

            for pred, targ in zip(preds, targs):
                # Check if the coordinate is predicted correctly
                # and count
                if torch.argmax(pred) == torch.argmax(targ):
                    count += 1

            if count == 4:
                self.correct += 1
            self.total += 1


    def compute(self):
        return self.correct.float() / self.total


