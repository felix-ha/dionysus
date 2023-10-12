import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.dionysus.loss as loss


def eps():
    return 1e-10


def test_init_distillation_loss():
    teacher = nn.Linear(10, 10)
    temperature = 1.0
    alpha = 0.5

    loss_func = loss.DistillationLoss(teacher, temperature, alpha)

    assert loss_func.teacher == teacher
    assert loss_func.temperature == temperature
    assert loss_func.alpha == alpha
    assert isinstance(loss_func.loss_func, nn.CrossEntropyLoss)


def test_single_cross_entropy_loss():
    logits = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    y_true = torch.tensor([[0, 0, 1]], dtype=torch.float)

    loss_func = nn.CrossEntropyLoss()
    loss_expected = loss_func(logits, y_true)

    loss_actual = -torch.log(torch.exp(logits[0, 2]) / torch.sum(torch.exp(logits)))

    assert torch.abs(loss_expected - loss_actual) < eps()


def test_batched_cross_entropy_loss():
    logits = torch.tensor([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
    y_true = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float)

    loss_func = nn.CrossEntropyLoss(reduction="sum")
    loss_expected = loss_func(logits, y_true)

    loss_actual = torch.sum(
        torch.sum(-torch.log(F.softmax(logits, dim=-1)) * y_true, dim=-1)
    )

    assert torch.abs(loss_expected - loss_actual) < eps()


@pytest.mark.parametrize("reduction", [("mean"), ("sum")])
def test_custom_class_cross_entropy_loss_sum(reduction):
    torch.manual_seed(0)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)

    loss_func_torch = nn.CrossEntropyLoss(reduction=reduction)
    loss_expected = loss_func_torch(input, target)

    loss_func = loss.CrossEntropyLoss(reduction=reduction)
    loss_actual = loss_func(input, target)

    assert torch.abs(loss_expected - loss_actual) < eps()


@pytest.mark.parametrize("reduction", [("mean"), ("sum")])
def test_custom_class_cross_entropy_loss_int(reduction):
    torch.manual_seed(0)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)

    loss_func_torch = nn.CrossEntropyLoss(reduction=reduction)
    loss_expected = loss_func_torch(input, target)

    loss_func = loss.CrossEntropyLoss(reduction=reduction)
    loss_actual = loss_func(input, target)

    assert torch.abs(loss_expected - loss_actual) < eps()
