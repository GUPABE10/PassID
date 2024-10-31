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

class NorfairTracker():
    def __init__(self):
        pass
    
    # Function to write tracking data to file
    def write_to_file(self, tracked_objects, frame_number, output_file):
        """
        Writes tracking data to a specified file.

        Parameters:
        - tracked_objects: List of objects being tracked.
        - frame_number: Current frame number.
        - output_file: Path to the output file.
        """
        with open(output_file, 'a') as f:
            for obj in tracked_objects:
                id = obj.id  # Assumes each tracked object has an 'id' attribute
                x1, y1, x2, y2 = obj.estimate.flatten().tolist()

                width = x2 - x1
                height = y2 - y1
                line = [str(frame_number), str(id), str(x1), str(y1), str(width), str(height)] + ['-1', '-1', '-1', '-1']
                f.write(','.join(line) + '\n')

    def track(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str, drawing: bool, evalFile: bool, isVideo: bool, outputDir: str, track_points: str = None):
        """
        Main tracking function that handles object detection and tracking over frames.

        Parameters:
        - input_path: Path to video or image directory.
        - model: Object detection model.
        - model_threshold: Detection confidence threshold.
        - distance_threshold: Threshold to decide if objects match across frames.
        - distance_function: Function for distance measurement ('iou' or 'scalar').
        - drawing: Boolean to enable drawing on frames.
        - evalFile: Boolean to enable writing tracking data to a file.
        - isVideo: Boolean to indicate if input is a video.
        - outputDir: Directory to store output files.
        - track_points: Indicates if tracking is based on 'centroid' or 'bbox'.
        """
        coord_transformations = None
        paths_drawer = None
        fix_paths = True  # Fixed path setting for tracking visuals

        # Prepare output file path
        full_path = os.path.abspath(input_path)
        path_components = full_path.split(os.sep)
        parent_folder_name = path_components[-2] if len(path_components) > 1 else 'a'
        output_file = os.path.join(outputDir, parent_folder_name + '.txt')
        
        # Load video or images from the input path
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
        
        # Setup for camera motion compensation and motion estimation
        transformations_getter = HomographyTransformationGetter()
        motion_estimator = MotionEstimator(
            max_points=500, min_distance=7, transformations_getter=transformations_getter
        )

        # Configure distance function for tracking
        if distance_function == "iou":
            distance_function = "iou"
        elif distance_function == "scalar":
            distance_function = create_normalized_mean_euclidean_distance(
                height, width
            )
            distance_threshold = DISTANCE_THRESHOLD_CENTROID
        else:
            Warning("Distance function not recognized. Using default: scalar")
            distance_function = create_normalized_mean_euclidean_distance(
                height, width
            )
            distance_threshold = DISTANCE_THRESHOLD_CENTROID

        tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
        )
        
        # Initialize drawing paths if enabled
        if drawing:
            paths_drawer = Paths(center, attenuation=0.01)
            if fix_paths:
                paths_drawer = AbsolutePaths(max_history=40, thickness=2)
            
        for frame_image in video_images:
            if not isVideo:
                frame_number = str(int(os.path.splitext(frame_image)[0]))
                frame = cv2.imread(os.path.join(input_path, frame_image)) 
            else:
                frame = frame_image
                        
            # Object detection on the current frame
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)

            # Mask and update transformations for motion estimation
            mask = np.ones(frame.shape[:2], frame.dtype)
            coord_transformations = motion_estimator.update(frame, mask)

            # Convert model detections to Norfair-compatible detections
            detections = self.rcnn_detections_to_norfair_detections(
                model_boxes, model_scores, track_points
            )

            # Update tracker with current detections
            tracked_objects = tracker.update(
                detections=detections, coord_transformations=coord_transformations
            )
            
            # Write tracking data to file if enabled
            if evalFile:
                self.write_to_file(tracked_objects, frame_number, output_file)
            
            # Draw on the frame if drawing is enabled and input is a video
            if drawing and isVideo:
                frame = draw(
                    paths_drawer,
                    track_points,
                    frame,
                    detections,
                    tracked_objects,
                    coord_transformations,
                    fix_paths,
                )
                video_images.write(frame)
            
    def rcnn_detections_to_norfair_detections(self, 
        rcnn_boxes: torch.Tensor,
        rcnn_scores: torch.Tensor,
        track_points: str = "centroid"
    ) -> List[Detection]:
        """
        Converts model detections to Norfair Detection objects for tracking.

        Parameters:
        - rcnn_boxes: Bounding boxes from the detection model.
        - rcnn_scores: Confidence scores for each detection.
        - track_points: Specifies if tracking uses 'bbox' or 'centroid'.

        Returns:
        - List of Norfair Detection objects.
        """
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
