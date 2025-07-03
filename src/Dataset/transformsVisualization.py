import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A

def draw_bboxes(image_np, bboxes, labels, class_name_map=None, color=(0, 255, 0), thickness=2):
    img_res = image_np.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Convert to list if necessary
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    for bbox, label in zip(bboxes, labels):
        try:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
        except Exception as e:
            print(f"Warning: Could not convert bbox coords to int: {bbox}, Error: {e}")
            continue
        cv2.rectangle(img_res, (x_min, y_min), (x_max, y_max), color, thickness)
        label_name = str(label) if class_name_map is None else class_name_map.get(label, str(label))
        (text_width, text_height), baseline = cv2.getTextSize(label_name, font, font_scale, font_thickness)
        text_y = y_min - baseline if y_min - baseline > text_height else y_min + text_height
        cv2.putText(img_res, label_name, (x_min, text_y), font, font_scale, color, font_thickness)
    return img_res

def visualize_bbox_augmentations(image, bboxes, labels, transform, samples=5):
    """
    Visualizes original image and several augmentations with bounding boxes.
    """
    # Convert to lists if needed
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    # Remove Normalize and ToTensorV2 for visualization
    vis_transform_list = [
        t for t in transform if not isinstance(t, (A.Normalize, A.pytorch.transforms.ToTensorV2))
    ] if isinstance(transform, A.Compose) else []
    bbox_params = transform.processors['bboxes'].params if isinstance(transform, A.Compose) and 'bboxes' in transform.processors else None
    vis_transform = A.Compose(vis_transform_list, bbox_params=bbox_params) if bbox_params else transform

    figure, ax = plt.subplots(1, samples + 1, figsize=(15, 5))

    # Draw original
    original_drawn = draw_bboxes(image, bboxes, labels)
    ax[0].imshow(original_drawn)
    ax[0].set_title("Original")
    ax[0].axis("off")

    # Draw augmented samples
    for i in range(samples):
        try:
            # Prepare label fields for Albumentations
            label_fields = bbox_params.label_fields if bbox_params else ['class_labels']
            label_args = {field: labels for field in label_fields}
            augmented = vis_transform(image=image, bboxes=bboxes, **label_args)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented[label_fields[0]] if label_fields else labels
            augmented_drawn = draw_bboxes(aug_image, aug_bboxes, aug_labels)
            ax[i+1].imshow(augmented_drawn)
            ax[i+1].set_title(f"Augmented {i+1}")
        except Exception as e:
            print(f"Error during augmentation sample {i+1}: {e}")
            ax[i+1].imshow(image)
            ax[i+1].set_title(f"Aug Error {i+1}")
        finally:
            ax[i+1].axis("off")

    plt.tight_layout()
    plt.show()
