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
    if backbone == "mobilenetHigh":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    else: # FasterRCNN
        print("finetuned resnet v2")
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # Obtener el número de características de entrada del clasificador
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 3
    # Reemplazar la cabeza del clasificador con una nueva
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    weight_name = "FasterRCNN_finetuned_2epochs_noda.pth"
    
    ruta_modelo = os.path.join(script_dir, weight_name)
    # model.load_state_dict(torch.load(ruta_modelo))
    
    state_dict = torch.load(ruta_modelo)
    # Ajustar las claves en el state_dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Elimina el prefijo 'module.' de cada clave
        new_state_dict[name] = v
    
    # Cargar el state_dict ajustado
    model.load_state_dict(new_state_dict)

    return model

class FasterRCNN:
    def __init__(self, model, backbone, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:2" if torch.cuda.is_available() else "cpu"
        
        print(device)

        self.device = device  # <-- Asignar el dispositivo aquí
        if "finetuned" in model:
            self.isTuned = True
        else:
            self.isTuned = False

        # load model
        try:
            if backbone == "mobilenetHigh":
                print("MobileNet HD")
                
                if model == "FasterRCNN_pretrained":
                    self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
                elif model == "FasterRCNN_finetuned":
                    self.model = get_finetuned_model(backbone)
            else: # resnet50v2
                print("ResNet50 v2")
                if model == "FasterRCNN_pretrained":
                    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image.

        Args:
            img (Union[str, np.ndarray]): Image path or numpy array.
            conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.

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
    

        if self.isTuned:
            desired_classes = (labels == 1) | (labels == 2)
        else:
            # Filter for desired classes: 1 for "person", 37 for "sports ball"
            desired_classes = (labels == 1) | (labels == 37)
        
        # if not desired_classes.any():
            # print(labels)
            # raise Exception("The model is not detecting the desired classes.")

        # Apply confidence threshold
        selected_indices = torch.where(desired_classes & (scores >= conf_threshold))[0]
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        labels = labels[selected_indices]

        return boxes, scores, labels
    
    
