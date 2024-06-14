"""
Escribiré un método simple que resuma todo

Recibe: 
- directorio de videos, Imagenes
- Frame_rate necesario para imagenes, opcional para videos para obtener las imagenes


"""
import numpy as np
from tracker.base_tracker import BaseTracker
from utils.match_info import VideoInfo, Match

class PassDetection(BaseTracker):
    def __init__(self):
        super().__init__()

    def detect_passes(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str, isVideo: bool, track_points: str = None):
        video_images, height, width = self.load_images_or_video(input_path, isVideo)
        motion_estimator = self.initialize_motion_estimator()
        distance_function, distance_threshold = self.set_distance_function(distance_function, height, width, distance_threshold)
        tracker = self.initialize_tracker(distance_function, distance_threshold)
        
        # Incializar Clase Video
        VideoInfo = VideoInfo(video_path = input_path)
        
        # Inicializar Clase Partido
        Match = Match()
        
        # Bandera de frame de inicio
        isDetectionStarted = False

        for frame_image in video_images:
            
            ########### Error: Labels. Son diferentes labels si uso FasterRCNN y YOLO
            
            # Sección de tracking original
            frame_number, frame = self.process_frame(input_path, frame_image, isVideo)
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            self.coord_transformations = motion_estimator.update(frame, mask)
            detections = self.my_detections_to_norfair_detections(model_boxes, model_scores, model_labels, track_points)
            tracked_objects = tracker.update(detections=detections, coord_transformations=self.coord_transformations)

            # Sección de detección de pases
            # aqui está pendiete saber si solo con esta condicion es suficiente para el frame de inicialización
            if tracked_objects:
                print("Frame de inicializacion: ")
                print(tracked_objects)
                isDetectionStarted = True
                
            if isDetectionStarted:
                self.detect_pass_logic(tracked_objects, frame_number)

    def detect_pass_logic(self, tracked_objects, frame_number):
        # Aqui ya cuento con jugadores detectados y balón
        # Aplicar Cluster
        
        
        pass


"""
for obj in tracked_objects:
    id = obj.id
    x1, y1, x2, y2 = obj.estimate.flatten().tolist()
    label = obj.label # esta se obtiene de DEtection si se guardó previamente

"""