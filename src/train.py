import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    num_correct = 0
    num_samples = 0

    for images, labels in dataloader:
        images = images.to(device)  # [batch_size, 3, height, width]
        labels = labels.to(device)  # [batch_size]

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)  # [batch_size, num_classes]
        loss = criterion(logits, labels)  # scalar

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)  # [batch_size]
        is_correct = predictions == labels  # [batch_size]

        num_correct += is_correct.sum().item()
        num_samples += batch_size

    average_loss = total_loss / num_samples
    accuracy = num_correct / num_samples

    return average_loss, accuracy


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    num_correct = 0
    num_samples = 0

    for images, labels in dataloader:
        images = images.to(device)  # [batch_size, 3, height, width]
        labels = labels.to(device)  # [batch_size]

        logits = model(images)  # [batch_size, num_classes]
        loss = criterion(logits, labels)  # scalar

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)  # [batch_size]
        is_correct = predictions == labels  # [batch_size]

        num_correct += is_correct.sum().item()
        num_samples += batch_size

    average_loss = total_loss / num_samples
    accuracy = num_correct / num_samples

    return average_loss, accuracy