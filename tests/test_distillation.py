import torch
import torch.nn.functional as F

from src.dionysus.loss import DistillationLoss


class Teacher:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return torch.tensor([[4, 2, 2]], dtype=torch.float32)

    def eval(self):
        pass


def test_distillation_loss():
    T = 2
    alpha = 0.5

    x = None

    teacher = Teacher()

    logits_teacher = teacher(x)
    sm_teacher_log = F.log_softmax(logits_teacher / T, dim=1)

    logits_student = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    sm_student = F.softmax(logits_student / T, dim=1)

    loss_kl = F.kl_div(sm_teacher_log, sm_student, reduction="batchmean")
    loss_ce = F.cross_entropy(logits_student, torch.tensor([0]))
    loss_expected = alpha * loss_ce + (1 - alpha) * T**2 * loss_kl

    loss_func_distillation = DistillationLoss(teacher, T, alpha)
    loss_actual = loss_func_distillation(logits_student, torch.tensor([0]), batch=x)

    assert loss_expected == loss_actual
