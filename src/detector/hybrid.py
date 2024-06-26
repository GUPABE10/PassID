import os
import cv2
import torch
import numpy as np
from typing import Tuple, Union
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

class HybridDetector:
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.device = device

        if "hybrid" in model_name:
            self.isHybrid = True
            # Cargar modelo YOLO finetuned
            yolo_weight_name = "yolov8fine.pt"
            script_dir = os.path.dirname(os.path.realpath(__file__))
            yolo_model_path = os.path.join(script_dir, yolo_weight_name)
            self.yolo_model = YOLO(yolo_model_path)

            # Cargar modelo Faster R-CNN preentrenado
            self.faster_rcnn_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            self.faster_rcnn_model = self.faster_rcnn_model.to(self.device)
            self.faster_rcnn_model.eval()
        else:
            raise ValueError("Unsupported model name. Use 'hybrid' for this detector.")

    
    def is_green_background(self, image, bbox, padding=10):
        x1, y1, x2, y2 = bbox
        height, width, _ = image.shape

        x1_exp = max(0, x1 - padding)
        y1_exp = max(0, y1 - padding)
        x2_exp = min(width, x2 + padding)
        y2_exp = min(height, y2 + padding)
        
        expanded_box = np.zeros_like(image[y1_exp:y2_exp, x1_exp:x2_exp])
        cv2.rectangle(expanded_box, (x1 - x1_exp, y1 - y1_exp), (x2 - x1_exp, y2 - y1_exp), (255, 255, 255), thickness=-1)
        
        expanded_region = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        mask = cv2.inRange(expanded_box, (255, 255, 255), (255, 255, 255))
        surrounding_region = cv2.bitwise_and(expanded_region, expanded_region, mask=mask)

        hsv = cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_ratio = np.sum(green_mask) / (green_mask.size * 255)
        
        return green_ratio > 0.5
    
    
    def predict(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image using YOLO for person detection and Faster R-CNN for ball detection.

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

        # Detección de personas con YOLO
        yolo_results = self.yolo_model(img, visualize=False, device=self.device, verbose=False)
        yolo_boxes = yolo_results[0].boxes.xyxy  # Cajas en formato xyxy
        yolo_confs = yolo_results[0].boxes.conf  # Confianzas
        yolo_labels = yolo_results[0].boxes.cls  # Etiquetas

        if isinstance(yolo_labels, np.ndarray):
            yolo_labels = torch.tensor(yolo_labels)

        person_mask = (yolo_labels == 0) & (yolo_confs >= conf_threshold)
        person_boxes = yolo_boxes[person_mask]
        person_scores = yolo_confs[person_mask]
        person_labels = yolo_labels[person_mask]
        person_labels[:] = 1  # Reemplazar etiqueta por 1 para "person"

        # Detección de pelotas con Faster R-CNN
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            rcnn_prediction = self.faster_rcnn_model([img_tensor])
        
        rcnn_boxes = rcnn_prediction[0]["boxes"]
        rcnn_scores = rcnn_prediction[0]["scores"]
        rcnn_labels = rcnn_prediction[0]["labels"]

        ball_mask = (rcnn_labels == 37) & (rcnn_scores >= conf_threshold)  # 37 es el label de "sports ball"
        ball_boxes = rcnn_boxes[ball_mask]
        ball_scores = rcnn_scores[ball_mask]
        ball_labels = rcnn_labels[ball_mask]
        ball_labels[:] = 2  # Reemplazar etiqueta por 2 para "ball"


        valid_ball_indices = []
        for i, bbox in enumerate(ball_boxes):
            if self.is_green_background(img, bbox.cpu().numpy().astype(int), padding=10):
                valid_ball_indices.append(i)

        valid_ball_indices = torch.tensor(valid_ball_indices)
        ball_boxes = ball_boxes[valid_ball_indices]
        ball_scores = ball_scores[valid_ball_indices]
        ball_labels = ball_labels[valid_ball_indices]


        # Combinar resultados
        combined_boxes = torch.cat((person_boxes, ball_boxes), dim=0)
        combined_scores = torch.cat((person_scores, ball_scores), dim=0)
        combined_labels = torch.cat((person_labels, ball_labels), dim=0)

        return combined_boxes, combined_scores, combined_labels
