from src.Dataset.transformsVisualization import visualize_bbox_augmentations
from src.Dataset.transforms import train_transforms
import albumentations as A
import cv2
import os
import pandas as pd
import numpy as np
import random

random.seed(42)

# Take 15 randoms images from data/imgs/train directory
sample_images = 15
sample_image_paths = random.sample(
    [f"data/imgs/train/{img}" for img in os.listdir("data/imgs/train") if img.endswith(('.jpg', '.png'))],
    sample_images
)
train_transform = A.Compose([
    A.RandomCrop(width=450, height=450, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

for i in range(sample_images):
    # Load the images and their corresponding bounding boxes and class labels
    image_path = sample_image_paths[i]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # strip "data/" from the image path for annotations
    image_path = os.path.relpath(image_path, 'data/')
    print(image_path)
    # We get bounding boxes from the annotations file
    ann_file = 'data/annotations/train_annotations.csv'
    df = pd.read_csv(ann_file,header=None,names=['image_path','xmin','ymin','xmax','ymax','label'])
    
    # Get all the bounding boxes for the current image in numpy format
    bboxes = df[df['image_path'] == image_path][['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
    
    # class labels are all pedestrian
    class_labels = df[df['image_path'] == image_path]['label'].to_numpy()

    # Define the transform (must include bbox_params with correct format and label_fields)
    train_transform = train_transforms

    # Visualize
    visualize_bbox_augmentations(image, bboxes, class_labels, train_transform, samples=5)
