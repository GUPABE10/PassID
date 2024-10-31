from typing import Optional, Union, Tuple
import numpy as np
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from collections import OrderedDict

def get_finetuned_model(cfg):
    """
    Load and return a finetuned Mask R-CNN model with specified configuration.

    Parameters:
    - cfg: Detectron2 configuration object.

    Returns:
    - DefaultPredictor instance with loaded finetuned weights.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    weight_name = "MaskRCNN_finetuned_2epochs_noda.pth"
    ruta_modelo = os.path.join(script_dir, weight_name)

    state_dict = torch.load(ruta_modelo)
    # Adjust keys in state_dict to remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load the adjusted state_dict into the model
    cfg.MODEL.WEIGHTS = ruta_modelo
    return DefaultPredictor(cfg)

class MaskRCNN:
    def __init__(self, model, device: Optional[str] = None):
        """
        Initialize a Mask R-CNN instance with specified model and device.

        Parameters:
        - model: String specifying model type (e.g., 'pretrained' or 'finetuned').
        - device: Device for computation ('cpu' or 'cuda').
        """
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(device)
        self.device = device
        self.isTuned = "finetuned" in model

        # Configure Detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Load pretrained or finetuned weights
        if model == "MaskRCNN_pretrained":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model == "MaskRCNN_finetuned":
            self.model = get_finetuned_model(cfg)
                
        cfg.MODEL.DEVICE = self.device
        self.model = DefaultPredictor(cfg)

    def predict(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image.

        Parameters:
        - img (Union[str, np.ndarray]): Path to image or image as a numpy array.
        - conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Detected boxes, scores, and labels.
        """
        
        # If input is a path, load and convert the image
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a format suitable for Detectron2
        img = img[:, :, ::-1]  # Convert from BGR to RGB
        outputs = self.model(img)
        
        # Extract boxes, scores, and labels from prediction results
        instances = outputs["instances"]
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        labels = instances.pred_classes

        # Filter results based on finetuned model or general model classes
        if self.isTuned:  # Class 1 for person, class 2 for ball
            desired_classes = (labels == 1) | (labels == 2)
        else:
            # Filter for general model classes: 0 for "person", 32 for "sports ball"
            desired_classes = (labels == 0) | (labels == 32)
        
        # Apply confidence threshold
        selected_indices = torch.where(desired_classes & (scores >= conf_threshold))[0]
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        labels = labels[selected_indices]
        
        # Replace label 32 with 2 for ball
        labels[labels == 32] = 2

        return boxes, scores, labels
