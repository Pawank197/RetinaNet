from src.models.retinaNet import create_model
from src.Dataset.dataset import StanfordAerialPedestrianDataset
from src.Dataset.transforms import train_transforms, val_transforms
from torch.utils.data import DataLoader
from src.utils.collateFunction import collate_fn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from config import CONFIG
from src.utils.trainer import train_one_epoch, validate
import csv, os, pandas as pd

def main():
    # Fix the random seed for reproducibility
    torch.manual_seed(42)
    print(f"Using device: {CONFIG['DEVICE']}")

    # Load the dataset
    train_dataset = StanfordAerialPedestrianDataset(
        split='train',
        data_dir='data/',
        annotations_file='data/annotations/train_annotations.csv',
        label_file='data/labels.csv',
        transforms=train_transforms
    )

    val_dataset = StanfordAerialPedestrianDataset(
        split='val',
        data_dir='data/',
        annotations_file='data/annotations/val_annotations.csv',
        label_file='data/labels.csv',
        transforms=val_transforms
    )

    # Create the DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        collate_fn=collate_fn
    )

    # Load the model
    model = create_model(CONFIG['NUM_CLASSES'])

    # Move the model to the device
    model.to(CONFIG['DEVICE'])

    # Define the optimzer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['LEARNING_RATE'],
        weight_decay=CONFIG['WEIGHT_DECAY']
    )

    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=CONFIG['NUM_EPOCHS'])


    # TRAINING + RESUMING 
    print("Starting training...")
    start_epoch = 0
    best_mAP = 0.0
    train_losses_history = []
    val_mAP_history = []
    CHECKPOINT_FILE = CONFIG['CHECKPOINT_PATH']

    # check if the best model exists
    if CONFIG['RESUME_TRAINING'] and os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming training from checkpoint: {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint['best_mAP']
        
        print(f"Resuming from Epoch {start_epoch}. Best mAP so far: {best_mAP:.4f}")
    else:
        print("Starting training from scratch...")
        with open(CONFIG['METRICS_FILE'], 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'avg_train_loss', 'val_mAP'])


    for epoch in range(start_epoch, CONFIG['NUM_EPOCHS']):

        # Train the model for one epoch
        print(f"Epoch: [{epoch+1}/{CONFIG['NUM_EPOCHS']}] Training")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG['DEVICE'])

        # print the average training loss
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validate the model
        print(f"Epoch: [{epoch+1}/{CONFIG['NUM_EPOCHS']}] Validation")

        mAP = validate(model, val_loader, CONFIG['DEVICE'])
        # print the mAP 
        print(f"mAP: {mAP['map']:.4f}")

        # Step the scheduler
        scheduler.step()
        
        # Append the metrics to the history
        train_losses_history.append(avg_train_loss)
        val_mAP_history.append(mAP['map'].item())

        # Write the metrics to the CSV file
        with open(CONFIG['METRICS_FILE'], 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, mAP['map'].item()])

        # If the best mAP, then save this model
        if mAP['map'] > best_mAP:
            best_mAP = mAP['map']
            print(f"New best mAP: {best_mAP:.4f}, Saving model...")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_map': best_mAP
            }
            torch.save(checkpoint, CONFIG['CHECKPOINT_PATH'])

    print("\nTraining Complete!/n")
    print(f"Best mAP: {best_mAP:.4f}")


if __name__ == "__main__":
    main()