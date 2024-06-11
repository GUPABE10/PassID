"""
Escribiré un método simple que resuma todo

Recibe: 
- directorio de videos, Imagenes
- Frame_rate necesario para imagenes, opcional para videos para obtener las imagenes

1. Inicializo Clase video con Frame_rate
    AL incializar 


"""
import numpy as np
from tracker.base_tracker import BaseTracker

class PassDetection(BaseTracker):
    def __init__(self):
        super().__init__()

    def detect_passes(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str, isVideo: bool, track_points: str = None):
        video_images, height, width = self.load_images_or_video(input_path, isVideo)
        motion_estimator = self.initialize_motion_estimator()
        distance_function, distance_threshold = self.set_distance_function(distance_function, height, width, distance_threshold)
        tracker = self.initialize_tracker(distance_function, distance_threshold)

        for frame_image in video_images:
            
            ########### Error: Labels. Son diferentes labels si uso FasterRCNN y YOLO
            
            frame_number, frame = self.process_frame(input_path, frame_image, isVideo)
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            self.coord_transformations = motion_estimator.update(frame, mask)
            detections = self.rcnn_detections_to_norfair_detections(model_boxes, model_scores, track_points)
            tracked_objects = tracker.update(detections=detections, coord_transformations=self.coord_transformations)

            # Aquí puedes agregar la lógica específica para la detección de pases
            self.detect_pass_logic(tracked_objects, frame_number)

    def detect_pass_logic(self, tracked_objects, frame_number):
        # Implementa la lógica específica para detectar pases aquí
        
        
        
        pass
