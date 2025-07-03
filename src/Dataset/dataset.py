import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class StanfordAerialPedestrianDataset(Dataset):
    """
    This class represents the Aerial Pedestrian Dataset.
    
    The annotations file is in the format:  [image_path,xmin,ymin,xmax,ymax,label] (pascal_voc format)
                                                    
    """
    def __init__(self, split, data_dir, annotations_file, label_file, transforms=None):
        """
        Initializes the dataset with the given arguments:

        Args:
            split (str): The dataset split, e.g., 'train', 'val', 'test'.
            data_dir (str): The directory where the dataset is stored. (Here, it should equal to data/)
            annotations_file (str): The path to the annotations file.
            label_file (str): The path to the label file.
            transforms: A function which implements a set of transformations on the dataset.
        """
        self.split = split
        self.data_dir = data_dir
        self.transforms = transforms

        # loading annotations and labels
        self.img_labels = pd.read_csv(annotations_file,header=None,names=['image_path','xmin','ymin','xmax','ymax','label'])
        self.class_labels = pd.read_csv(label_file,header=None,names=['label', 'id'])

        # Merge the annotations with class labels
        self.img_labels = self.img_labels.merge(self.class_labels, on='label')

        # Group annotations by image path
        self.img_labels = self.img_labels.groupby('image_path')

        # Create a list of unique image paths
        self.image_paths = list(self.img_labels.groups.keys())


    def __len__(self):
        """
        Returns the number of samples in the split dataset.
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image and its corresponding label(or labels).
        """
        # Get the image path and annotations for the given index
        img_path = self.image_paths[idx]
        annotations = self.img_labels.get_group(img_path)

        # Load the image
        img_path = os.path.join(self.data_dir, img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract bounding boxes and labels as lists
        boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        labels = annotations['id'].values.tolist()
        image_id = torch.tensor([idx], dtype=torch.int64)

        # Apply transforms. We assume that transforms is always called and not putting an else statement
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            raw_boxes = transformed['bboxes']
            raw_labels = transformed['class_labels']
        
        # Convert them into approriate tensors:
        if len(raw_boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0, ), dtype=torch.int64)
        else:
            boxes = torch.tensor(raw_boxes, dtype=torch.float32)
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
            labels = torch.tensor(raw_labels, dtype=torch.int64)

        # Create the target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id
        }
        
        return image, target
    