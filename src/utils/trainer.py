"""
Contains the training and validation loop
"""

from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

def train_one_epoch(model, train_loader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): The opimizer for the model.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    # wrap the DataLoader with tqdm for progress bar
    progress_bar = tqdm(train_loader, desc="Training", unit="batch", leave=False)

    for i, (images, targets) in enumerate(progress_bar):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        # update the progress bar with the current loss
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """
    Validates the model on the validation dataset

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        dict: A dictionary containing the mean average precision (mAP) for the validation dataset.
    """
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    progress_bar = tqdm(val_loader, desc="Validation", unit="batch", leave=False)

    with torch.no_grad():
        for i, (images, targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            metric.update(outputs, targets)

            progress_bar.set_postfix(i=i)
    
    return metric.compute()