"""
Escribiré un método simple que resuma todo

Recibe: 
- directorio de videos, Imagenes
- Frame_rate necesario para imagenes, opcional para videos para obtener las imagenes


"""
import numpy as np
from tracker.base_tracker import BaseTracker
from utils.match_info import VideoInfo, Match
from tasks.team_id import PlayerClassifier

class PassDetection(BaseTracker):
    def __init__(self):
        super().__init__()
        
        self.isDetectionStarted = False
        self.match = Match()

    def detect_passes(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str, isVideo: bool, track_points: str = None):
        video_images, height, width = self.load_images_or_video(input_path, isVideo)
        motion_estimator = self.initialize_motion_estimator()
        distance_function, distance_threshold = self.set_distance_function(distance_function, height, width, distance_threshold)
        tracker = self.initialize_tracker(distance_function, distance_threshold)
        
        # Incializar Clase Video
        self.videoInfo = VideoInfo(video_path = input_path)
        

        for frame_image in video_images: 
            
            #### Frame_image 
            # It returns regular OpenCV frames which enables the usage of the huge number of tools OpenCV provides to modify images.
            
            # Sección de tracking original
            frame_number, frame = self.process_frame(input_path, frame_image, isVideo)
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            self.coord_transformations = motion_estimator.update(frame, mask)
            detections = self.my_detections_to_norfair_detections(model_boxes, model_scores, model_labels, track_points)
            tracked_objects = tracker.update(detections=detections, coord_transformations=self.coord_transformations)

            # Sección de detección de pases
            # aqui está pendiete saber si solo con esta condicion es suficiente para el frame de inicialización
            if tracked_objects and not self.isDetectionStarted:
                print("Frame de inicializacion: ")
                print(tracked_objects)
                self.initFrame(frame_image, tracked_objects)
                
                
                isDetectionStarted = True
                
                
            if isDetectionStarted:
                self.detect_pass_logic(frame_image, tracked_objects, frame_number)
                
    def initFrame(self, image, tracked_objects):
        classifier = PlayerClassifier()
        classifier.classify(image=image, tracked_objects=tracked_objects, match=self.match, visualize=False)

    def detect_pass_logic(self, image, match, tracked_objects, frame_number):
        # Aqui ya cuento con jugadores detectados y balón
        # Fase de inicialización
        
        pass
        
        
        
        


"""
for obj in tracked_objects:
    id = obj.id
    x1, y1, x2, y2 = obj.estimate.flatten().tolist()
    label = obj.label # esta se obtiene de DEtection si se guardó previamente

"""