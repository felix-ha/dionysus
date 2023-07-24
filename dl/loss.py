import torch
import torch.nn as nn
import torch.nn.functional as F


class DistilationLoss:
    """
    Class is just used for testing.
    """

    def __init__(self, teacher, temperature, alpha):
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(input)

        teacher_sm = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_sm = F.softmax(input, dim=-1)

        loss_kd = self.temperature**2 * F.kl_div(
            teacher_sm, student_sm, log_target=True, reduction="sum"
        )
        loss_ce = self.loss_func(input, target)

        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        return loss


class CrossEntropyLoss:
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
