from torchmetrics import Metric
import torch

class CorrectDimension(Metric):
    '''Calculates hitrate
    
    '''
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        # It's not pretty, but I think it works
        for pred, targ in zip(preds, target):
            for idx in range(0, len(pred), 8):
                self.correct += 1 \
                    if torch.argmax(pred[idx:idx+8]) == torch.argmax(targ[idx:idx+8]) \
                    else 0
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total



class CorrectSquare(Metric):
    '''Calculates hitrate
    
    '''
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        # It's not pretty, but I think it works
        for pred, targ in zip(preds, target):
            for idx in range(0, len(pred), 16):
                if torch.argmax(pred[idx:idx+8]) == torch.argmax(targ[idx:idx+8]) \
                and torch.argmax(pred[idx+8:idx+16]) == torch.argmax(targ[idx+8:idx+16]):
                    self.correct += 1
                
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total

class CorrectMove(Metric):
    '''Calculates hitrate
    
    '''
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        for pred, targ in zip(preds, target):
            count = 0
            for idx in range(0, len(pred), 8):
                if torch.argmax(pred[idx:idx+8]) == torch.argmax(targ[idx:idx+8]):
                    count += 1

            if count == 4:
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total


