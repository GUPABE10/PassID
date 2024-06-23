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

from skimage.metrics import structural_similarity as ssim

class PlayerClassifier:
    def __init__(self, model_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device='cpu'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        self.predictor = DefaultPredictor(self.cfg)
        setup_logger()

    @staticmethod
    def calculate_histogram(segment_mask, image):
        mask = segment_mask.astype(np.uint8) * 255
        segment = cv2.bitwise_and(image, image, mask=mask)
        hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_segment], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def classify(self, image_path=None, image=None, tracked_objects=None, match=None, visualize=False):

        # Lectura de imagen por path
        if image_path:
            print("Image path")
            image = cv2.imread(image_path)
            print(image.shape)
            print(image.dtype, image.min(), image.max())
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('image_path.png')
            plt.show()
        elif image is not None:
            print("Image cv2")

            cv2.imwrite("tempImage.jpg", image)
            image = cv2.imread("tempImage.jpg")

            # print(image.shape)
            # print(image.dtype, image.min(), image.max())
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image)
            # # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.title('Original Image')
            # plt.savefig('image_analized.png')
            # plt.show()
            # # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # image_pt = cv2.imread("frame_7.jpg")
            # print(image_pt.dtype, image_pt.min(), image_pt.max())

            # if image.shape != image_pt.shape:
            #     raise ValueError("Las imágenes deben tener el mismo tamaño y número de canales")

            # if np.array_equal(image, image_pt):
            #     print("SON IGUALES")
            # else:
            #     print("NO SON IGUALES")

            # if np.allclose(image, image_pt, atol=1e-5):
            #     print("MEDIO IGUALES")
            # else:
            #     print("MEDIO NO SON IGUALES")

            # difference = cv2.absdiff(image, image_pt)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.title('Diferencias entre imágenes')
            # plt.savefig('Diferencias.png')
            # plt.show()

            # hist1 = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            # hist2 = cv2.calcHist([image_pt], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            # similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            # print(f"El índice de similitud de los histogramas es {similarity}")

            # image1_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image2_gray = cv2.cvtColor(image_pt, cv2.COLOR_BGR2GRAY)
            # score, diff = ssim(image1_gray, image2_gray, full=True)
            # diff = (diff * 255).astype("uint8")

            # plt.figure(figsize=(10, 10))
            # plt.imshow(diff, cmap='gray')
            # plt.axis('off')
            # plt.title('Diferencias usando SSIM')
            # plt.savefig('Diferencias_SSIM.png')
            # plt.show()
            # print(f"El índice SSIM entre las imágenes es {score}")

            # # image = cv2.imread("frame_7.jpg")

            # colors = ('b', 'g', 'r')
            # plt.figure(figsize=(20, 10))

            # for i, color in enumerate(colors):
            #     hist1 = cv2.calcHist([image], [i], None, [256], [0, 256])
            #     hist2 = cv2.calcHist([image_pt], [i], None, [256], [0, 256])
            #     plt.subplot(1, 3, i+1)
            #     plt.plot(hist1, color=color, label='Image 1')
            #     plt.plot(hist2, color=color, linestyle='dashed', label='Image 2')
            #     plt.title(f'Histogram for {color.upper()} channel')
            #     plt.xlabel('Pixel value')
            #     plt.ylabel('Frequency')
            #     plt.legend()
            # plt.savefig('Histogramas.png')
            # plt.tight_layout()
            # plt.show()

            # def compare_histograms_detailed(image1, image2):
            #     colors = ('b', 'g', 'r')
            #     plt.figure(figsize=(20, 10))
                
            #     for i, color in enumerate(colors):
            #         hist1 = cv2.calcHist([image1], [i], None, [256], [0, 256])
            #         hist2 = cv2.calcHist([image2], [i], None, [256], [0, 256])
                    
            #         diff_hist = hist1 - hist2
                    
            #         plt.subplot(1, 3, i+1)
            #         plt.plot(hist1, color=color, label='Image 1')
            #         plt.plot(hist2, color=color, linestyle='dashed', label='Image 2')
            #         plt.plot(diff_hist, color='black', linestyle='dotted', label='Difference')
            #         plt.title(f'Histogram Comparison for {color.upper()} channel')
            #         plt.xlabel('Pixel value')
            #         plt.ylabel('Frequency')
            #         plt.legend()
                
            #     plt.tight_layout()
            #     plt.savefig('Histogramas_detallado.png')
            #     plt.show()
            # # Ejemplo de uso
            # compare_histograms_detailed(image, image_pt)


            # def highlight_differences(image1, image2, threshold=30):
            #     # Calcula la diferencia absoluta entre las imágenes
            #     difference = cv2.absdiff(image1, image2)
            #     gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                
            #     # Aplica un umbral para resaltar las diferencias significativas
            #     _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
                
            #     # Colorea las diferencias en la imagen original
            #     diff_highlight = image1.copy()
            #     diff_highlight[mask != 0] = [0, 0, 255]  # Resaltar diferencias en rojo
            #     return diff_highlight, mask

            # # Ejemplo de uso
            # diff_highlight, mask = highlight_differences(image, image_pt)

            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(diff_highlight, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.title('Diferencias resaltadas en rojo')
            # plt.savefig('Dif_rojo.png')
            # plt.show()

            # plt.figure(figsize=(10, 10))
            # plt.imshow(mask, cmap='gray')
            # plt.axis('off')
            # plt.title('Máscara de diferencias')
            # plt.savefig('Dif_mask.png')
            # plt.show()
        else:
            raise ValueError("Se debe proporcionar una imagen o una ruta de imagen.")
        
        

        # Inference
        outputs = self.predictor(image)
        instances = outputs["instances"]

        # Filtrar solo las detecciones de personas
        person_instances = instances[instances.pred_classes == 0]
        
        if visualize:
            self.visualize_instances(image, person_instances)

        # Calcular histogramas para cada persona detectada
        histograms = []
        for i in range(len(person_instances)):
            mask = person_instances.pred_masks[i].cpu().numpy()
            hist = self.calculate_histogram(mask, image)
            histograms.append(hist)
            
        # Aplicar HDBSCAN a los histogramas
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        labels = clusterer.fit_predict(histograms)
        print(labels)
        
        # Encontrar los dos clusters más grandes
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        top_two_clusters = unique_labels[sorted_indices[:2]]

        if visualize:
            self.visualize_clusters(image, person_instances, labels)

        if tracked_objects and match:
            return self.assign_clusters_to_tracked_objects(person_instances, labels, tracked_objects, match, top_two_clusters)
        
        

    def visualize_instances(self, image, person_instances):
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(person_instances.to("cpu"))
        result_image = out.get_image()[:, :, ::-1]
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image_rgb)
        plt.axis('off')
        plt.savefig('Segmentacion.png')
        plt.show()

    def assign_clusters_to_tracked_objects(self, person_instances, labels, tracked_objects, match, top_two_clusters):
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.estimate.flatten().tolist()
            obj_box = [x1, y1, x2, y2]
            max_iou = 0
            assigned_cluster = -1
            
            for i in range(len(person_instances)):
                box = person_instances.pred_boxes.tensor[i].cpu().numpy().astype(int)
                iou = self.calculate_iou(obj_box, box)
                if iou > max_iou:
                    max_iou = iou
                    assigned_cluster = labels[i]
            
            if assigned_cluster in top_two_clusters:
                match.add_player(player_id=obj.id, team=assigned_cluster)
            else: 
                match.add_extra_people(extra_id=obj.id)

        return match

    def visualize_clusters(self, image, person_instances, labels):
        colors = plt.cm.get_cmap('tab10', np.max(labels) + 1)
        bounding_box_image = image.copy()
        for i in range(len(person_instances)):
            box = person_instances.pred_boxes.tensor[i].cpu().numpy().astype(int)
            label = labels[i]
            color = colors(label)[:3] if label != -1 else (0.5, 0.5, 0.5)
            color = [int(c * 255) for c in color]
            cv2.rectangle(bounding_box_image, (box[0], box[1]), (box[2], box[3]), color, 5)
            cv2.putText(bounding_box_image, str(label), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Bounding Box Image with Cluster Colors')
        plt.savefig('result_image_with_clusters.png')
        plt.show()

# Ejemplo de uso de la clase
# classifier = PlayerClassifier()
# image_path = 'ruta/a/la/imagen.jpg'
# image = cv2.imread('ruta/a/la/imagen.jpg')  # Alternativamente, una imagen directamente
# tracked_objects = [...]  # Lista de objetos seguidos con atributos id, x1, y1, x2, y2, label
# match = Match()  # Instancia de la clase Match
# classifier.classify(image_path=image_path, tracked_objects=tracked_objects, match=match, visualize=True)
# classifier.classify(image=image, tracked_objects=tracked_objects, match=match, visualize=False)
