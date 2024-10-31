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

from collections import defaultdict, Counter

from sklearn.neighbors import KNeighborsClassifier

class PlayerClassifier:
    def __init__(self, model_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device='cpu'):
        """
        Initialize the PlayerClassifier with Detectron2 model for player detection.

        Parameters:
        - model_config: Path to the Detectron2 model configuration.
        - device: Device to perform computations ('cpu' or 'cuda').
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        self.predictor = DefaultPredictor(self.cfg)
        setup_logger()

        # Stores initial histograms of clusters for team classification
        self.initial_clusters_histograms = {}

        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.initial_histograms = []
        self.initial_labels = []

        self.histograms = []
        self.labels = []

    @staticmethod
    def calculate_histogram(segment_mask, image):
        """
        Calculate the HSV histogram of a segmented region of the image.

        Parameters:
        - segment_mask: Binary mask for the segment.
        - image: Original image.

        Returns:
        - Flattened normalized histogram.
        """
        mask = segment_mask.astype(np.uint8) * 255
        segment = cv2.bitwise_and(image, image, mask=mask)
        hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_segment], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    @staticmethod
    def calculate_iou(boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        - boxA, boxB: Bounding boxes in format [x1, y1, x2, y2].

        Returns:
        - IoU value.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def classify(self, image_path=None, image=None, tracked_objects=None, match=None, visualize=False, missing_ids=set(), frame_number=0, firstFrame = False):
        """
        Classify players by analyzing the image and clustering their histograms.

        Parameters:
        - image_path: Path to the input image.
        - image: Input image as a numpy array.
        - tracked_objects: List of tracked objects.
        - match: Match instance to store classification.
        - visualize: Boolean to enable visualization.
        - missing_ids: Set of IDs for players that are missing.
        - frame_number: Current frame number.
        - firstFrame: Boolean indicating if this is the first frame.
        """
        # Load image from path if provided
        if image_path:
            print("Image path")
            image = cv2.imread(image_path)
            print(image.shape)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.title('Original Image')
            plt.savefig('image_path.png')
            plt.show()
        elif image is not None:
            cv2.imwrite("tempImage.jpg", image)
            image = cv2.imread("tempImage.jpg")
        else:
            raise ValueError("An image or an image path must be provided.")
        
        # Run Detectron2 model inference
        outputs = self.predictor(image)
        instances = outputs["instances"]

        # Filter detections for only players (person class)
        person_instances = instances[instances.pred_classes == 0]
        
        if visualize:
            self.visualize_instances(image, person_instances)
            self.visualize_instances(image, instances)

        # Calculate histograms for each detected player
        histograms = []
        for i in range(len(person_instances)):
            mask = person_instances.pred_masks[i].cpu().numpy()
            hist = self.calculate_histogram(mask, image)
            histograms.append(hist)

        # Wait until at least 8 players are detected to initialize
        if firstFrame and len(histograms) < 8:
            return match

        # Store initial histograms and labels for clustering in the first frame
        if firstFrame:
            self.initial_histograms = histograms
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
            labels = clusterer.fit_predict(histograms)
            self.initial_labels = labels.tolist()
            
            self.histograms = histograms
            self.labels = labels.tolist()

        else:
            # Predict clusters with k-NN for new frames
            labels = self.knn.predict(histograms)

            if isinstance(self.labels, np.ndarray):
                self.labels = self.labels.tolist()
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()

            self.histograms.extend(histograms)
            self.labels.extend(labels)

        # Train k-NN model with all histograms and labels
        self.knn.fit(self.histograms, self.labels)
        
        # Find top two clusters with most detections, excluding outliers
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_indices = unique_labels != -1
        unique_labels = unique_labels[valid_indices]
        counts = counts[valid_indices]
        sorted_indices = np.argsort(counts)[::-1]
        top_two_clusters = unique_labels[sorted_indices[:2]]

        if visualize:
            self.visualize_clusters(image, person_instances, labels, frame_number)
            self.visualize_tracked_objects(image, tracked_objects, frame_number)

        # Assign clusters to tracked objects if available
        if tracked_objects and match:
            return self.assign_clusters_to_tracked_objects(person_instances, labels, tracked_objects, match, top_two_clusters, missing_ids)
    
    def visualize_instances(self, image, person_instances):
        """
        Visualize player instances on the image.

        Parameters:
        - image: Input image.
        - person_instances: Detected person instances from Detectron2.
        """
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(person_instances.to("cpu"))
        result_image = out.get_image()[:, :, ::-1]
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image_rgb)
        plt.axis('off')
        plt.savefig('Segmentacion.png')
        plt.show()

    def totallyNewPlayers(self, match, tracked_objects):
        """
        Check if no tracked players currently match any previously known players.

        Parameters:
        - match: Match instance.
        - tracked_objects: List of tracked objects.

        Returns:
        - Boolean indicating if all players are new.
        """
        player_ids = set(match.players.keys())

        for obj in tracked_objects:
            if obj.id in player_ids:
                return False
            
        return True
    
    def few_players_in_team(self, match, tracked_objects, top_two_clusters):
        """
        Check if there are fewer than a minimum number of players in any team.

        Parameters:
        - match: Match instance.
        - tracked_objects: List of tracked objects.
        - top_two_clusters: List of cluster labels for top two teams.

        Returns:
        - Boolean indicating if any team has too few players.
        """
        team_counts = Counter()
        player_ids = set(match.players.keys())
        
        for obj in tracked_objects:
            if obj.id in player_ids:
                player_team = match.players[obj.id].team
                if player_team in top_two_clusters:
                    team_counts[player_team] += 1
        
        for team in top_two_clusters:
            if team_counts[team] < 3:
                return True
        
        return False

    def assign_clusters_to_tracked_objects(self, person_instances, labels, tracked_objects, match, top_two_clusters, missing_ids):
        """
        Assign clusters to tracked objects based on detected clusters and player teams.

        Parameters:
        - person_instances: Detected player instances.
        - labels: Predicted cluster labels.
        - tracked_objects: List of tracked objects.
        - match: Match instance for player information.
        - top_two_clusters: List of cluster labels for top two teams.
        - missing_ids: Set of missing player IDs.

        Returns:
        - Updated match instance with assigned clusters.
        """
        player_ids = set(match.players.keys())
        totallyNewPlayers = self.totallyNewPlayers(match, tracked_objects)

        if totallyNewPlayers:
            if len(tracked_objects) < 6:
                return match

            print("This should only print once unless players stop being detected.")
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
        else:
            # Mapping clusters to existing teams
            cluster_to_team_counts = defaultdict(Counter)
            existing_teams = set()

            for player_id in player_ids:
                player = match.players[player_id]
                team_number = player.team
                existing_teams.add(team_number)
                assigned_cluster = self.get_cluster_for_player(player_id, labels, tracked_objects, person_instances, top_two_clusters)
                if assigned_cluster is not None:
                    cluster_to_team_counts[assigned_cluster][team_number] += 1

            # Assign majority team to each cluster
            cluster_to_team = {}
            for cluster, team_counts in cluster_to_team_counts.items():
                if len(team_counts) > 1 and team_counts.most_common(2)[0][1] == team_counts.most_common(2)[1][1]:
                    continue
                most_common_team = team_counts.most_common(1)[0][0]
                cluster_to_team[cluster] = most_common_team

            # Assign remaining team to remaining cluster if only one team is assigned
            assigned_teams = set(cluster_to_team.values())
            if len(assigned_teams) == 1 and len(existing_teams) == 2:
                remaining_team = list(existing_teams - assigned_teams)[0]
                remaining_cluster = list(set(top_two_clusters) - set(cluster_to_team.keys()))[0]
                cluster_to_team[remaining_cluster] = remaining_team

            if missing_ids:
                for obj in tracked_objects:
                    if obj.id in missing_ids and obj.label == 1:
                        assigned_cluster = self.get_cluster_for_object(obj, labels, tracked_objects, person_instances)
                        if assigned_cluster in cluster_to_team:
                            team_number = cluster_to_team[assigned_cluster]
                            match.add_player(player_id=obj.id, team=team_number)
                        elif assigned_cluster != -1 or assigned_cluster is None:
                            match.add_extra_people(extra_id=obj.id)

        return match

    def get_cluster_for_player(self, player_id, labels, tracked_objects, person_instances, top_two_clusters):
        """
        Get the cluster assigned to an existing player using IoU matching.

        Parameters:
        - player_id: ID of the player.
        - labels: Predicted cluster labels.
        - tracked_objects: List of tracked objects.
        - person_instances: Detected person instances.
        - top_two_clusters: List of cluster labels for top two teams.

        Returns:
        - Assigned cluster for the player.
        """
        max_iou = 0
        assigned_cluster = None

        for obj in tracked_objects:
            if obj.id == player_id:
                x1, y1, x2, y2 = obj.estimate.flatten().tolist()
                obj_box = [x1, y1, x2, y2]
                for i in range(len(person_instances)):
                    box = person_instances.pred_boxes.tensor[i].cpu().numpy().astype(int)
                    iou = self.calculate_iou(obj_box, box)
                    if iou > max_iou:
                        max_iou = iou
                        potential_cluster = labels[i]
                        if potential_cluster in top_two_clusters:
                            assigned_cluster = potential_cluster
        return assigned_cluster

    def get_cluster_for_object(self, obj, labels, tracked_objects, person_instances):
        """
        Get the cluster assigned to a new object using IoU matching.

        Parameters:
        - obj: Tracked object.
        - labels: Predicted cluster labels.
        - tracked_objects: List of tracked objects.
        - person_instances: Detected person instances.

        Returns:
        - Assigned cluster for the object.
        """
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
        return assigned_cluster

    def visualize_tracked_objects(self, image, tracked_objects, frame_number):
        """
        Visualize bounding boxes and IDs of tracked objects on the image.

        Parameters:
        - image: Input image.
        - tracked_objects: List of tracked objects.
        - frame_number: Current frame number.
        """
        bounding_box_image = image.copy()

        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj.estimate.flatten().tolist())
            color = (0, 255, 0)
            cv2.rectangle(bounding_box_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(bounding_box_image, f'ID: {obj.id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        bounding_box_image_rgb = cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(bounding_box_image_rgb)
        plt.axis('off')
        plt.title('Tracked Objects with IDs')
        plt.savefig(f"{frame_number}_tracked_objects_with_ids.png")
        plt.show()

    def visualize_clusters(self, image, person_instances, labels, frame_number):
        """
        Visualize bounding boxes and cluster labels for each detected person.

        Parameters:
        - image: Input image.
        - person_instances: Detected player instances.
        - labels: Cluster labels for each instance.
        - frame_number: Current frame number.
        """
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
        plt.savefig(f'{frame_number}_result_image_with_clusters.png')
        plt.show()


# Ejemplo de uso de la clase
# classifier = PlayerClassifier()
# image_path = 'ruta/a/la/imagen.jpg'
# image = cv2.imread('ruta/a/la/imagen.jpg')  # Alternativamente, una imagen directamente
# tracked_objects = [...]  # Lista de objetos seguidos con atributos id, x1, y1, x2, y2, label
# match = Match()  # Instancia de la clase Match
# classifier.classify(image_path=image_path, tracked_objects=tracked_objects, match=match, visualize=True)
# classifier.classify(image=image, tracked_objects=tracked_objects, match=match, visualize=False)
