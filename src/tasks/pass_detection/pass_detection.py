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

from detector.fasterrcnn import FasterRCNN
from detector.yolo import MyYOLODetector
from detector.maskrcnn import MaskRCNN

class PassDetection(BaseTracker):
    def __init__(self, input_path: str, 
                 conf_threshold,  
                 track_points,
                 distance_threshold,
                 distance_function,
                 backboneModel,
                 isVideo: bool,
                 device,
                 detector,
                 testMode
    ):
        super().__init__()
        
        self.isDetectionStarted = False
        self.match = Match()
        self.classifier = PlayerClassifier()

        self.input_path = input_path
        self.conf_threshold = conf_threshold
        self.track_points = track_points
        self.distance_threshold = distance_threshold
        self.distance_function = distance_function
        self.backboneModel = backboneModel
        self.isVideo = isVideo
        self.device = device
        # self.detector = detector
        self.testMode = testMode

        if "FasterRCNN" in detector:
            self.model = FasterRCNN(detector, backbone = backboneModel, device = device)
        elif "yolo" in detector:
            self.model = MyYOLODetector(detector, device)
        elif "MaskRCNN" in detector:
            self.model = MaskRCNN(detector, device=device)
        else:
            print("Unknown model")

        
        # Mismos metodos que Tracker
        self.video_images, height, width = self.load_images_or_video(self.input_path, self.isVideo)
        
        self.motion_estimator = self.initialize_motion_estimator()
        
        self.distance_function, self.distance_threshold = self.set_distance_function(self.distance_function, height, width, distance_threshold)
        
        self.tracker = self.initialize_tracker(self.distance_function, distance_threshold)
        
        # Incializar Clase Video
        self.videoInfo = VideoInfo(video_path = input_path)



    def detect_passes(self):

        print("Detecting Passes")

        for frame_image in self.video_images: 
            
            #### Frame_image 
            # It returns regular OpenCV frames which enables the usage of the huge number of tools OpenCV provides to modify images.
            
            # Sección de tracking original
            frame_number, frame = self.process_frame(self.input_path, frame_image, self.isVideo)
            model_boxes, model_scores, model_labels = self.model.predict(frame, conf_threshold=self.conf_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            coord_transformations = self.motion_estimator.update(frame, mask)
            detections = self.my_detections_to_norfair_detections(model_boxes, model_scores, model_labels, self.track_points)
            tracked_objects = self.tracker.update(detections=detections, coord_transformations=coord_transformations)

            # Sección de detección de pases

            # Sección de frame de inicializacion
            # aqui está pendiete saber si solo con esta condicion es suficiente para el frame de inicialización
            # print(tracked_objects)
            if tracked_objects and not self.isDetectionStarted:
                print("Frame de inicializacion: ")
                print(tracked_objects)
                # ! Test: visualize_cluster = True
                # Visualize cluster me permitirá ver que se hace bien la clusterización
                self.assign_objects(frame_image, tracked_objects, self.testMode)
                
                self.isDetectionStarted = True
                
                
            if self.isDetectionStarted:
                print("Detect Pass Logic")
                self.detect_pass_logic(frame_image, tracked_objects, frame_number)
                break
        
        print("Finalizado")
                
    def assign_objects(self, image, tracked_objects, visualize_cluster):
        # Aqui se puede hacer una prueba de funcionamiento para verificar que se hace bien
        self.match = self.classifier.classify(image=image, tracked_objects=tracked_objects, match = self.match, visualize=visualize_cluster)
        print("After Classify")
        print(self.match)

        self.match.assign_ball_to_match(tracked_objects)
        print("After assign Ball")
        print(self.match)
        

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

        if self.testMode:
            print("Mismos objetos? Debería ser True:")
            print(self.verify_tracked_objects(tracked_objects))
            return
        
        missing_ids = self.verify_tracked_objects(tracked_objects)

        if missing_ids:
            # Aqui puede haber un error si un jugador previamente se asignó a un equipo y después el cluster lo canbia por otro
            # return
            # self.assign_objects(image, tracked_objects)
            pass
            
        
    # Esta función es para verificar que todos los tracked_objects estén en match, y verificar si hay nuevos
    def verify_tracked_objects(self, tracked_objects):
        # Obtenemos los ids de players, extras y ball (si no es None)
        player_ids = set(self.match.players.keys())
        extra_ids = self.match.extras
        ball_id = self.match.ball.id if self.match.ball is not None else None

        # Conjunto para almacenar los IDs que no están presentes
        missing_ids = set()

        # Verificamos que cada id en tracked_objects esté en players, extras o ball
        for obj in tracked_objects:
            obj_id = obj.id
            if obj_id not in player_ids and obj_id not in extra_ids and obj_id != ball_id:
                missing_ids.add(obj_id)

        return missing_ids

        


"""
for obj in tracked_objects:
    id = obj.id
    x1, y1, x2, y2 = obj.estimate.flatten().tolist()
    label = obj.label # esta se obtiene de DEtection si se guardó previamente

"""