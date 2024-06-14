import numpy as np
import torch
from typing import List

from draw import center, draw

from norfair import AbsolutePaths, Paths, Tracker, Video, Detection
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance, iou

from norfair.tracker import Detection, TrackedObject

import os
import cv2

DISTANCE_THRESHOLD_CENTROID: float = 0.08

class BaseTracker:
    def __init__(self):
        self.coord_transformations = None
        self.paths_drawer = None
        self.fix_paths = True

    def get_parent_folder_name(self, full_path):
        path_components = full_path.split(os.sep)
        return path_components[-2] if len(path_components) > 1 else 'a'

    def load_images_or_video(self, input_path, isVideo):
        if isVideo:
            video_images = Video(input_path=input_path)
            height = video_images.input_height
            width = video_images.input_width
        else:
            files = os.listdir(input_path)
            video_images = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
            video_images = sorted(video_images, key=lambda x: int(os.path.splitext(x)[0]))
            first_image_path = os.path.join(input_path, video_images[0])
            image = cv2.imread(first_image_path)
            height, width = image.shape[:2]
        return video_images, height, width

    def initialize_motion_estimator(self):
        transformations_getter = HomographyTransformationGetter()
        return MotionEstimator(max_points=500, min_distance=7, transformations_getter=transformations_getter)

    def set_distance_function(self, distance_function, height, width, distance_threshold):
        if distance_function == "iou":
            distance_function = "iou"
        elif distance_function == "scalar":
            distance_function = create_normalized_mean_euclidean_distance(height, width)
            distance_threshold = DISTANCE_THRESHOLD_CENTROID
        else:
            Warning("Distance function not recognized. Using default: scalar")
            distance_function = create_normalized_mean_euclidean_distance(height, width)
            distance_threshold = DISTANCE_THRESHOLD_CENTROID
        return distance_function, distance_threshold

    def initialize_tracker(self, distance_function, distance_threshold):
        return Tracker(distance_function=distance_function, distance_threshold=distance_threshold)

    def initialize_paths_drawer(self):
        return AbsolutePaths(max_history=40, thickness=2)

    def process_frame(self, input_path, frame_image, isVideo):
        if not isVideo:
            frame_number = str(int(os.path.splitext(frame_image)[0]))
            frame = cv2.imread(os.path.join(input_path, frame_image))
        else:
            frame = frame_image
            frame_number = None
        return frame_number, frame

    def my_detections_to_norfair_detections(self, my_boxes: torch.Tensor, my_scores: torch.Tensor, my_labels: torch.Tensor, track_points: str = "centroid") -> List[Detection]:
        norfair_detections: List[Detection] = []
        if track_points == "centroid":
            for box, score, label in zip(my_boxes, my_scores, my_labels):
                centroid = np.array(
                    [
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                    ]
                )
                scores = np.array([score.item(), score.item()])
                norfair_detections.append(Detection(points=centroid, scores=scores, label=label.item()))
        elif track_points == "bbox":
            for box, score, label in zip(my_boxes, my_scores, my_labels):
                bbox = np.array(
                    [
                        [box[0].item(), box[1].item()],
                        [box[2].item(), box[3].item()],
                    ]
                )
                scores = np.array([score.item(), score.item()])
                norfair_detections.append(Detection(points=bbox, scores=scores, label=label.item()))
        return norfair_detections

    def draw_frame(self, track_points, frame, detections, tracked_objects):
        return draw(self.paths_drawer, track_points, frame, detections, tracked_objects, self.coord_transformations, self.fix_paths)
