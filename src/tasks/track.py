from detector.fasterrcnn import FasterRCNN
from tracker.norfair_tracker import NorfairTracker


def track(input_video, conf_threshold, track_points, distance_threshold, distance_function, backboneModel):
    print("Track")
    
    modelDetector = FasterRCNN(backbone = backboneModel)  # Change the model initialization here
    
    modelTracker = NorfairTracker()  # Change the model initialization here

    # If the model for tracking is changed, the track function must be changed
    modelTracker.track(
        input_video=input_video, 
        model=modelDetector, 
        model_threshold=conf_threshold, 
        track_points=track_points,
        distance_threshold = distance_threshold,
        distance_function = distance_function
    )