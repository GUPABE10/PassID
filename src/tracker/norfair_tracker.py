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

# DISTANCE_TRESHOLD = 100

class NorfairTracker():
    def __init__(self):
        pass
    
    
    # Función para escribir los datos en el archivo
    def write_to_file(self, tracked_objects, frame_number, output_file):
        with open(output_file, 'a') as f:
            # print("frame: {}, num_obj: {}".format(frame_number, len(tracked_objects)))
            for obj in tracked_objects:
                id = obj.id  # Suponiendo que cada objeto rastreado tiene un atributo 'id'
                
                x1, y1, x2, y2 = obj.estimate.flatten().tolist()

                width = x2 - x1
                height = y2 - y1
                
                # bb = model_boxes[obj.id].tolist()  # Asumiendo que model_boxes es un tensor y que tiene el mismo índice que los IDs de tracked_objects
                line = [str(frame_number), str(id), str(x1), str(y1), str(width), str(height)] + ['-1', '-1', '-1', '-1']
                f.write(','.join(line) + '\n')


    def track(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str , drawing: bool, evalFile: bool, isVideo: bool,outputDir: str, track_points: str = None) :
        
        coord_transformations = None
        paths_drawer = None
        # fix_paths = False
        fix_paths = True


        # Obtén la ruta completa hacia 'img1'
        full_path = os.path.abspath(input_path)
        
        # Separa la ruta en sus componentes para obtener el nombre del directorio padre
        path_components = full_path.split(os.sep)
        parent_folder_name = path_components[-2] if len(path_components) > 1 else 'a'

        output_file = os.path.join(outputDir, parent_folder_name + '.txt')

        
        if isVideo:
            video_images = Video(input_path=input_path)
            height = video_images.input_height
            width = video_images.input_width
        else:
            # print(input_path)
            files = os.listdir(input_path)
            # print(files)
            # Filtra los archivos para obtener solo las imágenes
            video_images = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
            video_images = sorted(video_images, key=lambda x: int(os.path.splitext(x)[0]))
            # print(video_images)

            first_image_path = os.path.join(input_path, video_images[0])
            image = cv2.imread(first_image_path)
            # Obtiene las dimensiones de la imagen
            height, width = image.shape[:2]
        
        transformations_getter = HomographyTransformationGetter()
        
        motion_estimator = MotionEstimator(
            max_points=500, min_distance=7, transformations_getter=transformations_getter
        )

        # distance_function = create_normalized_mean_euclidean_distance(
        #     video.input_height, video.input_width
        # )
        
        if distance_function == "iou":
            distance_function = "iou"
        elif "scalar":
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
        
        if drawing:
            paths_drawer = Paths(center, attenuation=0.01)

            if fix_paths:
                paths_drawer = AbsolutePaths(max_history=40, thickness=2)
            
        for frame_image in video_images:
            
            if not isVideo:
                # frame_number = os.path.splitext(frame_image)[0]
                frame_number = str(int(os.path.splitext(frame_image)[0]))
                frame = cv2.imread(os.path.join(input_path, frame_image)) 
            else:
                frame = frame_image
                        
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)  # Change the model call here

            mask = np.ones(frame.shape[:2], frame.dtype)

            coord_transformations = motion_estimator.update(frame, mask)

            detections = self.rcnn_detections_to_norfair_detections(
                model_boxes, model_scores, track_points  # Change the conversion function call here
            )

            tracked_objects = tracker.update(
                detections=detections, coord_transformations=coord_transformations
            )
            
            if evalFile:
                
                self.write_to_file(tracked_objects, frame_number, output_file)
            
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
