import numpy as np
import os
from tracker.base_tracker import BaseTracker

class NorfairTracker(BaseTracker):
    def __init__(self):
        super().__init__()

    def track(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str, drawing: bool, evalFile: bool, isVideo: bool, outputDir: str, track_points: str = None):
        """
        Main tracking function that processes frames and applies tracking.

        Parameters:
        - input_path: Path to the input video or image directory.
        - model: Object detection model to use.
        - model_threshold: Detection confidence threshold.
        - distance_threshold: Threshold to determine if objects match across frames.
        - distance_function: Function for distance calculation ('iou' or 'scalar').
        - drawing: Boolean to enable drawing on frames.
        - evalFile: Boolean to enable saving tracking data to a file.
        - isVideo: Boolean indicating if input is a video.
        - outputDir: Directory to store output files.
        - track_points: Specifies if tracking is based on 'centroid' or 'bbox'.
        """
        # Prepare output file path
        full_path = os.path.abspath(input_path)
        parent_folder_name = self.get_parent_folder_name(full_path)
        output_file = os.path.join(outputDir, parent_folder_name + '.txt')

        # Load video frames or images
        video_images, height, width = self.load_images_or_video(input_path, isVideo)
        
        # Initialize motion estimator and tracker settings
        motion_estimator = self.initialize_motion_estimator()
        distance_function, distance_threshold = self.set_distance_function(distance_function, height, width, distance_threshold)
        tracker = self.initialize_tracker(distance_function, distance_threshold)

        # Initialize paths drawer for visualization if drawing is enabled
        if drawing:
            self.paths_drawer = self.initialize_paths_drawer()

        # Process each frame in the input
        for frame_image in video_images:
            frame_number, frame = self.process_frame(input_path, frame_image, isVideo)

            # Detect objects in the frame
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            # Update transformations and convert detections for Norfair compatibility
            self.coord_transformations = motion_estimator.update(frame, mask)
            detections = self.my_detections_to_norfair_detections(model_boxes, model_scores, model_labels, track_points)
            tracked_objects = tracker.update(detections=detections, coord_transformations=self.coord_transformations)   

            # Write tracking data to file if evalFile is enabled
            if evalFile:
                self.write_to_file(tracked_objects, frame_number, output_file)

            # Draw on frame if drawing is enabled and input is a video
            if drawing and isVideo:
                frame = self.draw_frame(track_points, frame, detections, tracked_objects)
                video_images.write(frame)

    def write_to_file(self, tracked_objects, frame_number, output_file):
        """
        Writes tracking data to the specified file.

        Parameters:
        - tracked_objects: List of objects being tracked.
        - frame_number: Current frame number.
        - output_file: Path to the output file.
        """
        with open(output_file, 'a') as f:
            for obj in tracked_objects:
                id = obj.id
                x1, y1, x2, y2 = obj.estimate.flatten().tolist()
                width = x2 - x1
                height = y2 - y1
                line = [str(frame_number), str(id), str(x1), str(y1), str(width), str(height)] + ['-1', '-1', '-1', '-1']
                f.write(','.join(line) + '\n')
