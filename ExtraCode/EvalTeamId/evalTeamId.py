import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

class PlayerClassifier:
    def __init__(self, model_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device='cpu'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        self.predictor = DefaultPredictor(self.cfg)

    @staticmethod
    def calculate_histogram(segment_mask, image):
        mask = segment_mask.astype(np.uint8) * 255
        segment = cv2.bitwise_and(image, image, mask=mask)
        hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_segment], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def classify(self, image, y_true):
        # Inference
        outputs = self.predictor(image)
        instances = outputs["instances"]

        # Filtrar solo las detecciones de personas
        person_instances = instances[instances.pred_classes == 0]

        # Calcular histogramas para cada persona detectada
        histograms = []
        for i in range(len(person_instances)):
            mask = person_instances.pred_masks[i].cpu().numpy()
            hist = self.calculate_histogram(mask, image)
            histograms.append(hist)

        # Clusterización con HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        y_pred = clusterer.fit_predict(histograms)

        # Comparación con las etiquetas verdaderas y cálculo de métricas
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        # homogeneity = homogeneity_score(y_true, y_pred)
        # completeness = completeness_score(y_true, y_pred)
        # v_measure = v_measure_score(y_true, y_pred)

        # Mostrar las métricas
        print(f"ARI: {ari}")
        print(f"AMI: {ami}")
        # print(f"Homogeneity: {homogeneity}")
        # print(f"Completeness: {completeness}")
        # print(f"V-Measure: {v_measure}")
        
        return y_pred, ari, ami#, homogeneity, completeness, v_measure

# Ejemplo de uso
classifier = PlayerClassifier()
image = cv2.imread('ruta/a/la/imagen.jpg')
y_true = [0, 0, 1, 1, 0]  # Etiquetas verdaderas (equipos reales)
classifier.classify(image, y_true)
