"""
We load the RetinaNet model from torchvision.

Details:
- Backbone: ResNet50
- Neck: FPN (Feature Pyramid Network)
- Anchors: Default: [32, 64, 128, 256, 512]
           But, in our dataset, the pedestrians are quite small so the anchors are set to:
           New: [16, 32, 64, 128, 256]
- Loss Function: Focal Loss         
"""

import torchvision
import torch
from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.anchor_utils import AnchorGenerator

def create_anchors():
    """
    We want to have a new set of anchors, different from the default ones.

    Returns:
        anchor_generator (torchvision.models.detection.anchor_utils.AnchorGenerator): An AnchorGenerator object with custom anchor sizes and aspect ratios.
    """
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [16, 32, 64, 128, 256])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator

def create_model(num_classes=2):
    """
    Loads this model: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html

    Args:
        num_classes (int): Number of classes in the dataset. We are setting it to 2, since we have only one class and background.
    
    Returns:
        model (torchvision.models.detection.RetinaNet): A RetinaNet model with a ResNet50 backbone and FPN neck, customized for the given number of classes.
    """

    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    )

    model.anchor_generator=create_anchors()
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )

    return model
