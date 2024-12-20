import os
import cv2
import torch
import numpy as np
from typing import Tuple, Union
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt

class HybridDetector:
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the hybrid detector combining YOLO for person detection and Faster R-CNN for ball detection.

        Parameters:
        - model_name: String indicating the type of model, should be 'hybrid' for this class.
        - device: Device to perform computations ('cpu' or 'cuda').
        """
        self.device = device

        if "hybrid" in model_name:
            self.isHybrid = True
            # Load YOLO finetuned model
            yolo_weight_name = "yolov8fine.pt"
            script_dir = os.path.dirname(os.path.realpath(__file__))
            yolo_model_path = os.path.join(script_dir, yolo_weight_name)
            self.yolo_model = YOLO(yolo_model_path)

            # Load Faster R-CNN pretrained model
            self.faster_rcnn_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            self.faster_rcnn_model = self.faster_rcnn_model.to(self.device)
            self.faster_rcnn_model.eval()
        else:
            raise ValueError("Unsupported model name. Use 'hybrid' for this detector.")

    def visualize_bbox(self, image, bbox, padding):
        """
        Visualizes the bounding box region with optional padding for analysis.

        Parameters:
        - image: Input image.
        - bbox: Bounding box coordinates (x1, y1, x2, y2).
        - padding: Amount of padding to expand around the bounding box.
        """
        x1, y1, x2, y2 = bbox
        height, width, _ = image.shape

        # Expand the bounding box by padding within image bounds
        x1_exp = max(0, x1 - padding)
        y1_exp = max(0, y1 - padding)
        x2_exp = min(width, x2 + padding)
        y2_exp = min(height, y2 + padding)

        expanded_region = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        # Create a mask for the bounding box region
        mask = np.zeros(expanded_region.shape[:2], dtype=np.uint8)
        mask = cv2.rectangle(mask, (x1 - x1_exp, y1 - y1_exp), (x2 - x1_exp, y2 - y1_exp), 255, thickness=-1)
        surrounding_region = cv2.bitwise_and(expanded_region, expanded_region, mask=mask)

        # Display original image, expanded region, and masked surrounding region
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(expanded_region, cv2.COLOR_BGR2RGB))
        plt.title("Expanded Region")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2RGB))
        plt.title("Surrounding Region")
        plt.savefig('ball_detection.png')
        plt.show()

    def is_green_background(self, image, bbox, padding=10):
        """
        Checks if the surrounding region of a bounding box has a green background.

        Parameters:
        - image: Input image.
        - bbox: Bounding box coordinates (x1, y1, x2, y2).
        - padding: Amount of padding to expand around the bounding box.

        Returns:
        - Boolean indicating if the background around the bounding box is green.
        """
        x1, y1, x2, y2 = bbox
        height, width, _ = image.shape

        # Expand bounding box by padding, limited by image bounds
        x1_exp = max(0, x1 - padding)
        y1_exp = max(0, y1 - padding)
        x2_exp = min(width, x2 + padding)
        y2_exp = min(height, y2 + padding)
        
        # Create mask for surrounding region
        expanded_box = np.zeros_like(image[y1_exp:y2_exp, x1_exp:x2_exp])
        cv2.rectangle(expanded_box, (x1 - x1_exp, y1 - y1_exp), (x2 - x1_exp, y2 - y1_exp), 255, thickness=-1)
        
        expanded_region = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        mask = cv2.inRange(expanded_box, 255, 255)
        surrounding_region = cv2.bitwise_and(expanded_region, expanded_region, mask=mask)

        hsv = cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2HSV)
        
        # Define HSV range for green and create mask
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_ratio = np.sum(green_mask) / (green_mask.size * 255)

        return green_ratio > 0.1

    def predict(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
        padding: int = 10,
        green_ratio_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on the given image using YOLO for person detection and Faster R-CNN for ball detection.

        Parameters:
        - img (Union[str, np.ndarray]): Path to image or image as a numpy array.
        - conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.
        - padding (int, optional): Padding size for green background check. Defaults to 10.
        - green_ratio_threshold (float, optional): Threshold for green ratio in background. Defaults to 0.5.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Detected boxes, scores, and labels.
        """
        
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run YOLO model for person detection
        yolo_results = self.yolo_model(img, visualize=False, device=self.device, verbose=False)
        yolo_boxes = yolo_results[0].boxes.xyxy.float().to(self.device)
        yolo_confs = yolo_results[0].boxes.conf.float().to(self.device)
        yolo_labels = yolo_results[0].boxes.cls.long().to(self.device)

        if isinstance(yolo_labels, np.ndarray):
            yolo_labels = torch.tensor(yolo_labels, dtype=torch.long).to(self.device)

        # Filter YOLO results for person class
        person_mask = (yolo_labels == 0) & (yolo_confs >= conf_threshold)
        person_boxes = yolo_boxes[person_mask]
        person_scores = yolo_confs[person_mask]
        person_labels = yolo_labels[person_mask]
        person_labels[:] = 1  # Set label to 1 for "person"

        # Prepare image for Faster R-CNN model
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Run Faster R-CNN model for ball detection
        with torch.no_grad():
            rcnn_prediction = self.faster_rcnn_model([img_tensor])
        
        rcnn_boxes = rcnn_prediction[0]["boxes"].float().to(self.device)
        rcnn_scores = rcnn_prediction[0]["scores"].float().to(self.device)
        rcnn_labels = rcnn_prediction[0]["labels"].long().to(self.device)

        # Filter for sports ball class in Faster R-CNN results
        ball_mask = (rcnn_labels == 37) & (rcnn_scores >= conf_threshold)
        ball_boxes = rcnn_boxes[ball_mask]
        ball_scores = rcnn_scores[ball_mask]
        ball_labels = rcnn_labels[ball_mask]
        ball_labels[:] = 2  # Set label to 2 for "ball"

        # Filter balls based on green background
        valid_ball_indices = []
        for i, bbox in enumerate(ball_boxes):
            if self.is_green_background(img, bbox.cpu().numpy().astype(int), padding):
                valid_ball_indices.append(i)

        valid_ball_indices = torch.tensor(valid_ball_indices, dtype=torch.long).to(self.device)
        ball_boxes = ball_boxes[valid_ball_indices]
        ball_scores = ball_scores[valid_ball_indices]
        ball_labels = ball_labels[valid_ball_indices]

        # Combine person and ball detections
        combined_boxes = torch.cat((person_boxes, ball_boxes), dim=0).float()
        combined_scores = torch.cat((person_scores, ball_scores), dim=0).float()
        combined_labels = torch.cat((person_labels, ball_labels), dim=0).long()

        return combined_boxes, combined_scores, combined_labels
