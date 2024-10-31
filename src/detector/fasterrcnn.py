from typing import Optional, Union, Tuple
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import os
from collections import OrderedDict

def get_finetuned_model(backbone):
    """
    Load and return a finetuned Faster R-CNN model with specified backbone.

    Parameters:
    - backbone: String indicating the model backbone to use.

    Returns:
    - Finetuned Faster R-CNN model.
    """
    if backbone == "mobilenetHigh":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    else:  # Default to FasterRCNN with ResNet backbone
        print("Finetuned ResNet v2")
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    
    # Configure classifier with specified number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 3  # Adjust this based on the application needs
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load pre-trained weights
    script_dir = os.path.dirname(os.path.realpath(__file__))
    weight_name = "FasterRCNN_finetuned_2epochs_noda.pth"
    ruta_modelo = os.path.join(script_dir, weight_name)
    state_dict = torch.load(ruta_modelo)

    # Adjust keys in state_dict for compatibility
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix if present
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model

class FasterRCNN:
    def __init__(self, model, backbone, device: Optional[str] = None):
        """
        Initialize a FasterRCNN instance with specified model and backbone.

        Parameters:
        - model: String specifying model type (e.g., pretrained or finetuned).
        - backbone: Model backbone to use.
        - device: Device for computation ('cpu' or 'cuda').
        """
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(device)
        self.device = device
        self.isTuned = "finetuned" in model

        # Load the model with appropriate settings
        try:
            if backbone == "mobilenetHigh":
                print("MobileNet HD")
                if model == "FasterRCNN_pretrained":
                    self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
                elif model == "FasterRCNN_finetuned":
                    self.model = get_finetuned_model(backbone)
            else:  # Default to resnet50v2
                print("ResNet50 v2")
                if model == "FasterRCNN_pretrained":
                    self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
                elif model == "FasterRCNN_finetuned":
                    self.model = get_finetuned_model(backbone)
                
            self.model = self.model.to(self.device)
            self.model.eval()
        except:
            raise Exception("Failed to load the pretrained Faster R-CNN model.")

    def predict(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image.

        Parameters:
        - img (Union[str, np.ndarray]): Path to image or numpy array.
        - conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Detected boxes, scores, and labels.
        """
        
        # If the input is a path, load and convert the image
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert image to PyTorch tensor
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Run model inference
        with torch.no_grad():
            prediction = self.model([img_tensor])
        
        boxes = prediction[0]["boxes"]
        scores = prediction[0]["scores"]
        labels = prediction[0]["labels"]

        # Apply filtering for specific classes
        if self.isTuned:  # Class 1 for person, class 2 for ball
            desired_classes = (labels == 1) | (labels == 2)
        else:
            # For general detection, use classes 1 (person) and 37 (sports ball)
            desired_classes = (labels == 1) | (labels == 37)
        
        # Apply confidence threshold
        selected_indices = torch.where(desired_classes & (scores >= conf_threshold))[0]
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        labels = labels[selected_indices]
        
        # Convert label 37 to 2 if detected
        labels[labels == 37] = 2

        return boxes, scores, labels
