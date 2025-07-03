import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

train_transforms = A.Compose([
    A.RandomSizedBBoxSafeCrop(height=800, width=800, erosion_rate=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=2),
        A.Blur(blur_limit=3)
    ], p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc',
                             label_fields=['class_labels'],
                             min_area=0.0,
                             min_visibility=0.5))
val_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc',
                             label_fields=['class_labels']))

test_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc',
                             label_fields=['class_labels']))