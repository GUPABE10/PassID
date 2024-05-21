import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
import hdbscan

# Función para calcular el histograma de una región segmentada
def calculate_histogram(segment_mask, image):
    mask = segment_mask.astype(np.uint8) * 255  # Convertir la máscara a uint8 y escalarla
    segment = cv2.bitwise_and(image, image, mask=mask)
    hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_segment], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def classify_players(image_path):
    print("Classify_players")
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Configurar logger para detectron2 (para información de debug)
    setup_logger()

    # Mask RCNN Model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'  # 'cuda'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Umbral para la detección
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    # Inference
    outputs = predictor(image)
    instances = outputs["instances"]

    # Filtrar solo las detecciones de personas
    person_instances = instances[instances.pred_classes == 0]
    
    # Visualizar resultados
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(person_instances.to("cpu"))

    result_image = out.get_image()[:, :, ::-1]
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen con las segmentaciones y clusters
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image_rgb)
    plt.axis('off')
    plt.show()

    # Calcular histogramas para cada persona detectada
    histograms = []
    for i in range(len(person_instances)):
        mask = person_instances.pred_masks[i].cpu().numpy()
        hist = calculate_histogram(mask, image)
        histograms.append(hist)
        
        
        # Aplicar HDBSCAN a los histogramas
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    labels = clusterer.fit_predict(histograms)
    

    colors = plt.cm.get_cmap('tab10', np.max(labels) + 1)

    bounding_box_image = image.copy()
    for i in range(len(person_instances)):
        box = person_instances.pred_boxes.tensor[i].cpu().numpy().astype(int)
        label = labels[i]
        color = colors(label)[:3] if label != -1 else (0.5, 0.5, 0.5)  # Color gris para ruido
        color = [int(c * 255) for c in color]
        cv2.rectangle(bounding_box_image, (box[0], box[1]), (box[2], box[3]), color, 5)
        cv2.putText(bounding_box_image, str(label), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Mostrar la imagen con bounding boxes y números
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Bounding Box Image with Cluster Colors')
    # plt.show()
    plt.savefig('result_image_with_clusters.png')