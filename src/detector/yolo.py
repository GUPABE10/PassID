from ultralytics import YOLO
import cv2
import torch
from typing import Tuple, Union
import numpy as np
import os

class MyYOLODetector:
    def __init__(self, model_name: str, device: str = 'cpu'):
        
        model = model_name.split('_')[0]
        
        if "pretrained" in model_name:
            self.isTuned = False
            self.model = YOLO(model+".pt")
        elif "finetuned" in model_name:
            self.isTuned = True
            weight_name = model
            script_dir = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(script_dir, weight_name+"fine.pt")
            self.model = YOLO(model_path)
        self.device = device
    
    def predict(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image using YOLO.

        Args:
            img (Union[str, np.ndarray]): Image path or numpy array.
            conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns detected boxes, their scores, and labels.
        """
        
        # Si la entrada es una ruta (string), cargar la imagen
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ejecutar el modelo
        results = self.model(img, visualize = False, device = self.device, verbose=False)
        
        # results = model.predict('path/to/your/image.jpg', verbose=False)
        
        # Extraer cajas delimitadoras, confianzas y etiquetas
        boxes = results[0].boxes.xyxy  # Cajas en formato xyxy
        confs = results[0].boxes.conf  # Confianzas
        labels = results[0].boxes.cls  # Etiquetas
        
        
        
        # Convertir las etiquetas a un tensor si es necesario
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        if self.isTuned: # 0: player, 1:ball
            desired_classes = (labels == 0) | (labels == 1)
        else:
            # Filtrar para clases deseadas: 1 para "persona", 37 para "sports ball"
            desired_classes = (labels == 1) | (labels == 37)
        
        

        # Aplicar el filtro de confianza
        conf_mask = confs >= conf_threshold
        combined_mask = conf_mask & desired_classes
        
        filtered_boxes = boxes[combined_mask]
        filtered_scores = confs[combined_mask]
        filtered_labels = labels[combined_mask]
        
        # Cambiar etiquetas para que 1: player y 2: ball
        if self.isTuned:
            filtered_labels[filtered_labels == 1] = 2
            filtered_labels[filtered_labels == 0] = 1
        else:
            # Reemplazar la etiqueta 37 con 2
            filtered_labels[filtered_labels == 37] = 2

        return filtered_boxes, filtered_scores, filtered_labels