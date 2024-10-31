import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

class PlayerClassifier:
    def __init__(self, model_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device='cpu'):
        """
        Initialize the PlayerClassifier with a configuration for Mask R-CNN model from Detectron2.
        
        Parameters:
        - model_config: The configuration file for the Mask R-CNN model.
        - device: The device for running the model ('cpu' or 'cuda').
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection confidence threshold
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)  # Load model weights
        self.predictor = DefaultPredictor(self.cfg)

    @staticmethod
    def calculate_histogram(segment_mask, image):
        """
        Calculate the HSV histogram of a segmented region in the image.
        
        Parameters:
        - segment_mask: Binary mask for the segmented region.
        - image: Original image in which the mask is applied.
        
        Returns:
        - Normalized histogram for the HSV values of the segmented area.
        """
        mask = segment_mask.astype(np.uint8) * 255
        segment = cv2.bitwise_and(image, image, mask=mask)  # Segment the region of interest
        hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
        hist = cv2.calcHist([hsv_segment], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize the histogram
        return hist

    def classify(self, image, y_true):
        """
        Classify players into clusters based on their histogram features.
        
        Parameters:
        - image: Input image for detecting and segmenting players.
        - y_true: Ground truth labels representing the actual teams of players.
        
        Returns:
        - Predicted cluster labels and evaluation metrics (ARI and AMI).
        """
        # Model inference to detect players
        outputs = self.predictor(image)
        instances = outputs["instances"]

        # Filter for person class only
        person_instances = instances[instances.pred_classes == 0]

        # Calculate histograms for each detected player
        histograms = []
        for i in range(len(person_instances)):
            mask = person_instances.pred_masks[i].cpu().numpy()
            hist = self.calculate_histogram(mask, image)
            histograms.append(hist)

        # Clustering using HDBSCAN on the histograms
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        y_pred = clusterer.fit_predict(histograms)

        # Compute evaluation metrics comparing predictions with ground truth
        ari = adjusted_rand_score(y_true, y_pred)  # Adjusted Rand Index
        ami = adjusted_mutual_info_score(y_true, y_pred)  # Adjusted Mutual Information
        # homogeneity = homogeneity_score(y_true, y_pred)
        # completeness = completeness_score(y_true, y_pred)
        # v_measure = v_measure_score(y_true, y_pred)

        # Print the evaluation metrics
        print(f"ARI: {ari}")
        print(f"AMI: {ami}")
        # print(f"Homogeneity: {homogeneity}")
        # print(f"Completeness: {completeness}")
        # print(f"V-Measure: {v_measure}")
        
        return y_pred, ari, ami  #, homogeneity, completeness, v_measure

# Example usage
classifier = PlayerClassifier()
image = cv2.imread('ruta/a/la/imagen.jpg')
y_true = [0, 0, 1, 1, 0]  # True labels representing real teams
classifier.classify(image, y_true)
