import torch

"""
Configuration file for the RetinaNet model training.
"""

CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_EPOCHS": 50,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "NUM_WORKERS": 2,
    "NUM_CLASSES": 2,
    "TRAINABLE_BACKBONE_LAYERS": 3,
    "METRICS_FILE": "metrics.csv",
    "CHECKPOINT_PATH": "checkpoint.pth",
    "RESUME_TRAINING": False # Set to True to resume
}