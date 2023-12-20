import numpy as np
import torch
from typing import List

from draw import center, draw

from norfair import AbsolutePaths, Paths, Tracker, Video, Detection
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance

DISTANCE_THRESHOLD_CENTROID: float = 0.08


class NorfairTracker():
    def __init__(self):
        pass
    

    def track(self, input_video: str, model, model_threshold, track_points):
        
        coord_transformations = None
        paths_drawer = None
        fix_paths = True
        
        video = Video(input_path=input_video)
        
        transformations_getter = HomographyTransformationGetter()
        
        motion_estimator = MotionEstimator(
            max_points=500, min_distance=7, transformations_getter=transformations_getter
        )

        distance_function = create_normalized_mean_euclidean_distance(
            video.input_height, video.input_width
        )
        distance_threshold = DISTANCE_THRESHOLD_CENTROID

        tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
        )
        
        paths_drawer = Paths(center, attenuation=0.01)

        if fix_paths:
            paths_drawer = AbsolutePaths(max_history=40, thickness=2)
            
        for frame in video:
            rcnn_boxes, rcnn_scores, rcnn_labels = model(frame, conf_threshold=model_threshold)  # Change the model call here

            mask = np.ones(frame.shape[:2], frame.dtype)

            coord_transformations = motion_estimator.update(frame, mask)

            detections = self.rcnn_detections_to_norfair_detections(
                rcnn_boxes, rcnn_scores, track_points=track_points  # Change the conversion function call here
            )

            tracked_objects = tracker.update(
                detections=detections, coord_transformations=coord_transformations
            )

            frame = draw(
                paths_drawer,
                #track_points,
                frame,
                detections,
                tracked_objects,
                coord_transformations,
                fix_paths,
            )
            video.write(frame)
            
    def rcnn_detections_to_norfair_detections(
        rcnn_boxes: torch.Tensor,
        rcnn_scores: torch.Tensor,
        track_points: str = "centroid"  # bbox or centroid
    ) -> List[Detection]:

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
