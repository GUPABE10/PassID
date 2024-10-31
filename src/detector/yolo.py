from ultralytics import YOLO
import cv2
import torch
from typing import Tuple, Union
import numpy as np
import os

class MyYOLODetector:
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the YOLO detector with specified model and device.

        Parameters:
        - model_name: Name of the model file (e.g., 'yolov5_pretrained' or 'yolov5_finetuned').
        - device: Computation device ('cpu' or 'cuda').
        """
        
        # Determine if model is pretrained or finetuned and load accordingly
        model = model_name.split('_')[0]
        
        if "pretrained" in model_name:
            self.isTuned = False
            self.model = YOLO(model + ".pt")  # Load pretrained model
        elif "finetuned" in model_name:
            self.isTuned = True
            weight_name = model
            script_dir = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(script_dir, weight_name + "fine.pt")
            self.model = YOLO(model_path)  # Load finetuned model
        self.device = device

    def predict(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image using YOLO.

        Parameters:
        - img (Union[str, np.ndarray]): Path to image or image as a numpy array.
        - conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Detected boxes, their scores, and labels.
        """
        
        # If input is a path, load the image
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run the YOLO model
        results = self.model(img, visualize=False, device=self.device, verbose=False)
        
        # Extract bounding boxes, confidences, and labels from results
        boxes = results[0].boxes.xyxy  # Bounding boxes in xyxy format
        confs = results[0].boxes.conf  # Confidence scores
        labels = results[0].boxes.cls  # Class labels
        
        # Convert labels to tensor if necessary
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        # Filter for specific classes based on finetuned vs. pretrained model
        if self.isTuned:  # Class 0 for player, class 1 for ball
            desired_classes = (labels == 0) | (labels == 1)
        else:
            # General model class filtering: 1 for "person", 32 for "sports ball"
            desired_classes = (labels == 0) | (labels == 32)

        # Apply confidence threshold
        conf_mask = confs >= conf_threshold
        combined_mask = conf_mask & desired_classes
        
        # Filter boxes, scores, and labels based on confidence and class
        filtered_boxes = boxes[combined_mask]
        filtered_scores = confs[combined_mask]
        filtered_labels = labels[combined_mask]
        
        # Convert labels for consistency: 1 for player and 2 for ball
        if self.isTuned:
            filtered_labels[filtered_labels == 1] = 2
            filtered_labels[filtered_labels == 0] = 1
        else:
            # Replace label 32 with 2
            filtered_labels[filtered_labels == 0] = 1
            filtered_labels[filtered_labels == 32] = 2

        return filtered_boxes, filtered_scores, filtered_labels
