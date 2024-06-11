import numpy as np
import os
from tracker.base_tracker import BaseTracker

class NorfairTracker(BaseTracker):
    def __init__(self):
        super().__init__()

    def track(self, input_path: str, model, model_threshold, distance_threshold, distance_function: str, drawing: bool, evalFile: bool, isVideo: bool, outputDir: str, track_points: str = None):
        full_path = os.path.abspath(input_path)
        parent_folder_name = self.get_parent_folder_name(full_path)
        output_file = os.path.join(outputDir, parent_folder_name + '.txt')

        video_images, height, width = self.load_images_or_video(input_path, isVideo)
        motion_estimator = self.initialize_motion_estimator()
        distance_function, distance_threshold = self.set_distance_function(distance_function, height, width, distance_threshold)
        tracker = self.initialize_tracker(distance_function, distance_threshold)

        if drawing:
            self.paths_drawer = self.initialize_paths_drawer()

        for frame_image in video_images:
            frame_number, frame = self.process_frame(input_path, frame_image, isVideo)
            model_boxes, model_scores, model_labels = model.predict(frame, conf_threshold=model_threshold)
            mask = np.ones(frame.shape[:2], frame.dtype)

            self.coord_transformations = motion_estimator.update(frame, mask)
            detections = self.rcnn_detections_to_norfair_detections(model_boxes, model_scores, track_points)
            tracked_objects = tracker.update(detections=detections, coord_transformations=self.coord_transformations)   

            if evalFile:
                self.write_to_file(tracked_objects, frame_number, output_file)

            if drawing and isVideo:
                frame = self.draw_frame(track_points, frame, detections, tracked_objects)
                video_images.write(frame)

    def write_to_file(self, tracked_objects, frame_number, output_file):
        with open(output_file, 'a') as f:
            for obj in tracked_objects:
                id = obj.id
                x1, y1, x2, y2 = obj.estimate.flatten().tolist()
                width = x2 - x1
                height = y2 - y1
                line = [str(frame_number), str(id), str(x1), str(y1), str(width), str(height)] + ['-1', '-1', '-1', '-1']
                f.write(','.join(line) + '\n')
