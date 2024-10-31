"""
    Initial module for the tracking task
"""

# Importing detectors
# Import custom detector implementations here
from detector.fasterrcnn import FasterRCNN
from detector.yolo import MyYOLODetector
from detector.maskrcnn import MaskRCNN
from detector.hybrid import HybridDetector

# Import module for Norfair Tracker
from tracker.norfair_tracker import NorfairTracker

def track(input_video, conf_threshold, track_points, distance_threshold, distance_function, backboneModel, draw, evalFile, isVideo, device, detector, outputDir):
    """
    Executes the tracking task using a specified detector and tracker.

    Parameters:
    - input_video: Path to the input video or images folder.
    - conf_threshold: Confidence threshold for object detection.
    - track_points: Defines if tracking is based on 'centroid' or 'bbox'.
    - distance_threshold: Threshold for determining if two points are the same object.
    - distance_function: Distance calculation method, e.g., scalar or IoU.
    - backboneModel: Backbone model for the detector.
    - draw: Boolean to determine if tracking visuals are drawn.
    - evalFile: Boolean to generate evaluation file.
    - isVideo: Boolean indicating if the input path is a video.
    - device: Device to run the model, e.g., 'cuda' or 'cpu'.
    - detector: Name of the object detector to use.
    - outputDir: Directory to save output files.

    """
    print("Track")

    # Initialize the detector based on the specified type
    if "FasterRCNN" in detector:
        modelDetector = FasterRCNN(detector, backbone=backboneModel, device=device)
    elif "yolo" in detector:
        modelDetector = MyYOLODetector(detector, device)
    elif "MaskRCNN" in detector:
        modelDetector = MaskRCNN(detector, device)
    elif "hybrid" in detector:
        modelDetector = HybridDetector(detector, device)
    else:
        print("Unknown model")

    # Initialize Norfair Tracker
    modelTracker = NorfairTracker()

    # Run tracking with the initialized detector and specified parameters
    modelTracker.track(
        input_path=input_video, 
        model=modelDetector, 
        model_threshold=conf_threshold, 
        track_points=track_points,
        distance_threshold=distance_threshold,
        distance_function=distance_function,
        drawing=draw,
        evalFile=evalFile,
        isVideo=isVideo,
        outputDir=outputDir
    )
