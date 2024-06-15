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
        self.classifier = PlayerClassifier()

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
                self.assign_objects(frame_image, tracked_objects)
                
                
                self.isDetectionStarted = True
                
                
            if self.isDetectionStarted:
                self.detect_pass_logic(frame_image, tracked_objects, frame_number)
                
    def assign_objects(self, image, tracked_objects):
        self.match = self.classifier.classify(image=image, tracked_objects=tracked_objects, match = self.match, visualize=False)
        self.match.assign_ball_to_match(tracked_objects)
        
        
        
    

    def detect_pass_logic(self, image, tracked_objects, frame_number):
        
        """
        Aqui ya cuento con jugadores detectados y balón
        Tengo que tomar en cuenta que
        tracked_objects puede tener a todas las personas
        pero match solo tiene los ids de los jugadores con equipo
        
        """
        # Fase de inicialización
        
        # Verificar que todos los tracked objects estén en match, incluyendo el balón.
        # Para esto puedo verificar tanto en players como en extra
        if not self.verify_tracked_objects(tracked_objects):
            # Aqui puede haber un error si un jugador previamente se asignó a un equipo y después el cluster lo canbia por otro
            self.assign_objects(image, tracked_objects)
            
        
        # 
        
        
        pass
        
    # Esta funcion es para verificar que todos los tracked_objects esten en match, y verificar si hay nuevos
    def verify_tracked_objects(self,tracked_objects):
        # Obtenemos los ids de players, extras y ball (si no es None)
        player_ids = set(self.match.players.keys())
        extra_ids = self.match.extras
        ball_id = self.match.ball.id if self.match.ball is not None else None

        # Verificamos que cada id en tracked_objects esté en players, extras o ball
        for obj in tracked_objects:
            obj_id = obj.id
            if obj_id not in player_ids and obj_id not in extra_ids and obj_id != ball_id:
                return False
        return True
        


"""
for obj in tracked_objects:
    id = obj.id
    x1, y1, x2, y2 = obj.estimate.flatten().tolist()
    label = obj.label # esta se obtiene de DEtection si se guardó previamente

"""