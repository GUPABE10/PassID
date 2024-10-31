"""
This class defines a simple method to summarize the pass detection pipeline.

Inputs:
- input_path: Path to videos or images directory.
- conf_threshold: Confidence threshold for detection.
- track_points: Points to use for tracking.
- distance_threshold: Threshold for distance in tracking.
- distance_function: Distance function to apply in tracking.
- backboneModel: Backbone model for the detector.
- isVideo: Boolean indicating if input_path is a video.
- device: Device to run the model on.
- detector: Type of detector to use (e.g., FasterRCNN, YOLO).
- testMode: Enables test mode for visualization.
"""

import os
import numpy as np
import cv2
import random
from tracker.base_tracker import BaseTracker
from utils.match_info import VideoInfo, Match
from tasks.team_id import PlayerClassifier

from detector.fasterrcnn import FasterRCNN
from detector.yolo import MyYOLODetector
from detector.maskrcnn import MaskRCNN
from detector.hybrid import HybridDetector

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

        # Get file name without extension and set output for passes CSV
        file_name = os.path.splitext(os.path.basename(self.input_path))[0]
        self.outPasses = f"{file_name}_passes.csv"

        self.conf_threshold = conf_threshold
        self.track_points = track_points
        self.distance_threshold = distance_threshold
        self.distance_function = distance_function
        self.backboneModel = backboneModel
        self.isVideo = isVideo
        self.device = device
        self.testMode = testMode

        # Initialize visualization drawer if in test mode
        if self.testMode:
            self.paths_drawer = self.initialize_paths_drawer()

        # Initialize the model based on specified detector type
        if "FasterRCNN" in detector:
            self.model = FasterRCNN(detector, backbone=backboneModel, device=device)
        elif "yolo" in detector:
            self.model = MyYOLODetector(detector, device)
        elif "MaskRCNN" in detector:
            self.model = MaskRCNN(detector, device=device)
        elif "hybrid" in detector:
            self.model = HybridDetector(detector, device)
        else:
            print("Unknown model")

        # Initialize tracking and video processing components
        self.video_images, height, width = self.load_images_or_video(self.input_path, self.isVideo)
        self.motion_estimator = self.initialize_motion_estimator()
        self.distance_function, self.distance_threshold = self.set_distance_function(self.distance_function, height, width, distance_threshold)
        self.tracker = self.initialize_tracker(self.distance_function, distance_threshold)
        self.videoInfo = VideoInfo(video_path=input_path)
        
        self.stop = False
        self.team_colors = {}

        # Define contrasting colors for team visualization
        self.contrast_colors = [
            (0, 0, 255),   # Red
            (0, 255, 0),   # Green
            (0, 255, 255), # Yellow
            (255, 0, 255), # Magenta
            (255, 255, 0), 
        ]
        self.color_idx = 0  # Index to cycle through colors

    def detect_passes(self):
        """
        Main function to detect passes in each frame.
        """
        print("Detecting Passes")
        frame_number = 1

        for frame_image in self.video_images: 
            
            # Original tracking section
            _, frame = self.process_frame(self.input_path, frame_image, self.isVideo)
            model_boxes, model_scores, model_labels = self.model.predict(frame, conf_threshold=self.conf_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            # Update transformations and detections
            self.coord_transformations = self.motion_estimator.update(frame, mask)
            detections = self.my_detections_to_norfair_detections(model_boxes, model_scores, model_labels, self.track_points)
            tracked_objects = self.tracker.update(detections=detections, coord_transformations=self.coord_transformations)

            # Initialization frame processing
            if tracked_objects and not self.isDetectionStarted:
                print("Initialization Frame: ")
                missing_ids = self.verify_tracked_objects(tracked_objects)
                self.assign_objects(frame_image, tracked_objects, self.testMode, missing_ids, frame_number)
                self.isDetectionStarted = True
                print(self.match)
                
            # Main pass detection logic
            if self.isDetectionStarted:
                self.detect_pass_logic(frame_image, tracked_objects, frame_number)

            # Visualization in test mode
            if self.testMode:
                frame = self.draw_ball_possession(frame, tracked_objects)
                frame = self.draw_teams(frame, tracked_objects)
                frame = self.draw_pass_info(frame)
                self.video_images.write(frame)

            if self.stop:
                break
                
            frame_number += 1
        
        print("Process Finished")
                
    def assign_objects(self, image, tracked_objects, visualize_cluster, missing_ids, frame_number):
        """
        Assign clusters and players to tracked objects.

        Parameters:
        - image: Input image.
        - tracked_objects: List of tracked objects.
        - visualize_cluster: Enable cluster visualization.
        - missing_ids: Set of IDs that are missing.
        - frame_number: Current frame number.
        """
        self.match = self.classifier.classify(image=image, tracked_objects=tracked_objects, match=self.match, visualize=False, missing_ids=missing_ids, frame_number=frame_number, firstFrame=not self.isDetectionStarted)
        self.match.assign_ball_to_match(tracked_objects, image)

    def detect_pass_logic(self, image, tracked_objects, frame_number):
        """
        Main logic to detect passes based on tracked objects and possession.

        Parameters:
        - image: Input image.
        - tracked_objects: List of tracked objects.
        - frame_number: Current frame number.
        """
        missing_ids = self.verify_tracked_objects(tracked_objects)

        # Assign missing objects if needed
        if missing_ids:
            self.assign_objects(image, tracked_objects, self.testMode, missing_ids, frame_number)

        # Update ball possession and pass information
        self.match.update_ball_possession(tracked_objects, self.videoInfo, frame_number, self.outPasses) 

    def verify_tracked_objects(self, tracked_objects):
        """
        Check for new objects in tracked_objects that are not in the match.

        Parameters:
        - tracked_objects: List of tracked objects.

        Returns:
        - Set of missing IDs.
        """
        player_ids = set(self.match.players.keys())
        extra_ids = self.match.extras
        ball_id = self.match.ball.id if self.match.ball is not None else None

        missing_ids = set()
        for obj in tracked_objects:
            obj_id = obj.id
            label = obj.label
            if obj_id not in player_ids and obj_id not in extra_ids and obj_id != ball_id:
                missing_ids.add(obj_id)

        return missing_ids

    def draw_ball_possession(self, frame, tracked_objects):
        """
        Draw ball possession information on the frame.

        Parameters:
        - frame: Current frame.
        - tracked_objects: List of tracked objects.

        Returns:
        - Modified frame with possession information.
        """
        if self.match.ball is not None:
            if self.match.ball.inPossession and self.match.lastPlayerWithBall is not None:
                for obj in tracked_objects:
                    if obj.id == self.match.lastPlayerWithBall.id:
                        x1, y1, x2, y2 = map(int, obj.estimate.flatten().tolist())
                        font_scale = frame.shape[0] / 800
                        cv2.putText(frame, "EN POSESION", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
                        break
        return frame

    def draw_teams(self, frame, tracked_objects):
        """
        Draw team information for players and ball in the frame.

        Parameters:
        - frame: Current frame.
        - tracked_objects: List of tracked objects.

        Returns:
        - Modified frame with team information.
        """
        ball_color = (255, 0, 0)  # Color for the ball
        font_scale = frame.shape[0] / 1000

        for obj in tracked_objects:
            id = obj.id
            x1, y1, x2, y2 = map(int, obj.estimate.flatten().tolist())

            if id in self.match.players:
                player = self.match.players[id]
                
                # Assign contrasting color if team doesn't have one
                if player.team not in self.team_colors:
                    self.team_colors[player.team] = self.contrast_colors[self.color_idx]
                    self.color_idx = (self.color_idx + 1) % len(self.contrast_colors)

                color = self.team_colors[player.team]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Player {player.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            elif self.match.ball is not None and id == self.match.ball.id:
                cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ball_color, 2)

        return frame
    
    def draw_pass_info(self, frame):
        """
        Draw pass information on the frame.

        Parameters:
        - frame: Current frame.

        Returns:
        - Modified frame with pass information.
        """
        if self.match.newPass is None:
            return frame

        pass_instance = self.match.newPass

        # Define the rectangle for pass info
        top_left_corner = (frame.shape[1] - 250, 10)
        bottom_right_corner = (frame.shape[1] - 10, 190)
        
        # Draw rectangle background for text
        cv2.rectangle(frame, top_left_corner, bottom_right_corner, (255, 255, 255), -1)

        # Set positions and prepare text
        text_start_x = top_left_corner[0] + 10
        text_start_y = top_left_corner[1] + 30
        line_height = 25
        pass_text = [
            f"Passer ID: {pass_instance.initPlayer.id}",
            f"Receiver ID: {pass_instance.finalPlayer.id}",
            f"Start Time: {pass_instance.secondInitPass:.2f} s",
            f"Duration: {pass_instance.secondFinalPass - pass_instance.secondInitPass:.2f} s",
            f"End Time: {pass_instance.secondFinalPass:.2f} s",
            f"Valid: {'Yes' if pass_instance.valid else 'No'}"
        ]

        # Write each line of text on frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(pass_text):
            y = text_start_y + i * line_height
            cv2.putText(frame, line, (text_start_x, y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return frame


"""
for obj in tracked_objects:
    id = obj.id
    x1, y1, x2, y2 = obj.estimate.flatten().tolist()
    label = obj.label # esta se obtiene de DEtection si se guardó previamente

"""