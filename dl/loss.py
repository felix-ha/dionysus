import torch 
import torch.nn.functional as F


class CrossEntropyLoss():
    """
    Class is just used for testing. 
    """
    def __init__(self, reduction="mean"):
        match reduction:
            case "sum":
                self.reduction_func = torch.sum
            case "mean":
                self.reduction_func = torch.mean
            case _:
                raise NotImplementedError(f"Case {reduction} not implemented")
            
    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        if len(target.shape) == 1:
            target = F.one_hot(target, num_classes=input.shape[-1])
        sm = F.softmax(input, dim=-1)
        negative_log = -torch.log(sm)
        return self.reduction_func(torch.sum(negative_log * target, dim=-1))
