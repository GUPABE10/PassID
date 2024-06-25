from detector.fasterrcnn import FasterRCNN
from detector.yolo import MyYOLODetector
from detector.maskrcnn import MaskRCNN
from tracker.norfair_tracker import NorfairTracker
from detector.hybrid import HybridDetector


def track(input_video, conf_threshold, track_points, distance_threshold, distance_function, backboneModel, draw, evalFile, isVideo, device, detector, outputDir):
    print("Track")
    
    if "FasterRCNN" in detector:
        modelDetector = FasterRCNN(detector, backbone = backboneModel, device = device)
    elif "yolo" in detector:
        modelDetector = MyYOLODetector(detector, device)
    elif "MaskRCNN" in detector:
        modelDetector = MaskRCNN(detector, device)
    elif "hybrid" in detector:
        modelDetector = HybridDetector(detector, device)
    else:
        print("Unknown model")
    
    modelTracker = NorfairTracker()  # Change the model initialization here

    # If the model for tracking is changed, the track function must be changed
    modelTracker.track(
        input_path=input_video, 
        model=modelDetector, 
        model_threshold=conf_threshold, 
        track_points=track_points,
        distance_threshold = distance_threshold,
        distance_function = distance_function,
        drawing = draw,
        evalFile = evalFile,
        isVideo = isVideo,
        outputDir = outputDir
    )
    