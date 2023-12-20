from typing import Optional, Union, Tuple
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

class FasterRCNN:
    def __init__(self, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device  # <-- Asignar el dispositivo aquí

        # load model
        try:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            self.model = self.model.to(self.device)
            self.model.eval()
        except:
            raise Exception("Failed to load the pretrained Faster R-CNN model.")


  
def predict(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image.

        Args:
            img (Union[str, np.ndarray]): Image path or numpy array.
            conf_threshold (float, optional): Confidence threshold. Defaults to 0.25.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns detected boxes and their scores.
        """
        
        # If the input is a string (path), load the image
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a PyTorch tensor
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Run the model
        with torch.no_grad():
            prediction = self.model([img_tensor])
        
        boxes = prediction[0]["boxes"]
        scores = prediction[0]["scores"]
        labels = prediction[0]["labels"] # New

        # Apply confidence threshold
        # selected_indices = scores >= conf_threshold
        # Filter for desired classes: 1 for "person", 37 for "sports ball"
        selected_indices = torch.where(((labels == 1) | (labels == 37)) & (scores >= conf_threshold))[0]
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]

        return boxes, scores, labels
    
    
