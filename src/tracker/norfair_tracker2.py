import os
import cv2
import numpy as np
import torch
from typing import List

from draw import center, draw
from norfair import AbsolutePaths, Paths, Tracker, Video, Detection
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance, iou

DISTANCE_THRESHOLD_CENTROID: float = 0.08

class NorfairTracker:
    def __init__(self):
        self.coord_transformations = None
        self.paths_drawer = None
        self.fix_paths = True

    def track(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str, drawing: bool, evalFile: bool, isVideo: bool, outputDir: str, track_points: str = None):
        full_path = os.path.abspath(input_path)
        parent_folder_name = self.get_parent_folder_name(full_path)
        output_file = os.path.join(outputDir, parent_folder_name + '.txt')

        video_images, height, width = self.load_images_or_video(input_path, isVideo)
        motion_estimator = self.initialize_motion_estimator()
        distance_function, distance_threshold = self.set_distance_function(distance_function, height, width, distance_threshold)
        tracker = self.initialize_tracker(distance_function, distance_threshold)

        if drawing:
            self.paths_drawer = self.initialize_paths_drawer()

        for frame_image in video_images:
            frame_number, frame = self.process_frame(input_path, frame_image, isVideo)
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            self.coord_transformations = motion_estimator.update(frame, mask)
            detections = self.rcnn_detections_to_norfair_detections(model_boxes, model_scores, track_points)
            tracked_objects = tracker.update(detections=detections, coord_transformations=self.coord_transformations)   

            if evalFile:
                self.write_to_file(tracked_objects, frame_number, output_file)

            if drawing and isVideo:
                frame = self.draw_frame(track_points, frame, detections, tracked_objects)
                video_images.write(frame)
                
                
            """
            if tracked_objects:
                frame_inicado = True
            if detectando_pases and frame_inicado: # Si estoy detectando pases y aparte ya empezó
                identificar pelota
                Jugador más cercano
                Distancia cercana?
            
            Checar ChatGPT
            
            """

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

    def draw_frame(self, track_points, frame, detections, tracked_objects):
        return draw(self.paths_drawer, track_points, frame, detections, tracked_objects, self.coord_transformations, self.fix_paths)

    def write_to_file(self, tracked_objects, frame_number, output_file):
        with open(output_file, 'a') as f:
            for obj in tracked_objects:
                id = obj.id
                x1, y1, x2, y2 = obj.estimate.flatten().tolist()
                width = x2 - x1
                height = y2 - y1
                line = [str(frame_number), str(id), str(x1), str(y1), str(width), str(height)] + ['-1', '-1', '-1', '-1']
                f.write(','.join(line) + '\n')

    def rcnn_detections_to_norfair_detections(self, rcnn_boxes: torch.Tensor, rcnn_scores: torch.Tensor, track_points: str = "centroid") -> List[Detection]:
        norfair_detections: List[Detection] = []
        if track_points == "centroid":
            for box, score in zip(rcnn_boxes, rcnn_scores):
                centroid = np.array(
                    [
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                    ]
                )
                scores = np.array([score.item(), score.item()])
                norfair_detections.append(Detection(points=centroid, scores=scores))
        elif track_points == "bbox":
            for box, score in zip(rcnn_boxes, rcnn_scores):
                bbox = np.array(
                    [
                        [box[0].item(), box[1].item()],
                        [box[2].item(), box[3].item()],
                    ]
                )
                scores = np.array([score.item(), score.item()])
                norfair_detections.append(Detection(points=bbox, scores=scores))
        return norfair_detections
