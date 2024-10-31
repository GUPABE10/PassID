"""
Author: Benjamin Gutierrez Padilla - A01732079
Instituto Tecnológico y de Estudios Superiores de Monterrey, Campus Monterrey
Master's Thesis Project in Computational Sciences.
""" 

# Import libraries and modules
import argparse
from tasks.track import track
from tasks.team_id import PlayerClassifier
from tasks.pass_detection.pass_detection import PassDetection

def setup_track_parser(subparsers):
    """Configure the subparser for the 'track' task"""
    parser_track = subparsers.add_parser('track', help='Track objects in video files or image folders')
    parser_track.add_argument('--files', type=str, help='Video files to process or folder path with images')
    parser_track.add_argument("--conf-threshold", type=float, default=0.8, help="Object confidence threshold")
    parser_track.add_argument("--track-points", type=str, default="bbox", help="Tracking points: 'centroid' or 'bbox'")
    parser_track.add_argument("--distance-threshold", type=float, default=1.0, help="Distance threshold for tracking")
    parser_track.add_argument("--distance-function", type=str, default="scalar", help="Distance function: 'scalar' or 'iou'")
    parser_track.add_argument("--backbone", type=str, default="resnet50v2", help="Backbone for object detector")
    parser_track.add_argument("--draw", type=bool, default=False, help="Generate video with visual tracking")
    parser_track.add_argument("--evalFile", type=bool, default=False, help="Generate evaluation file")
    parser_track.add_argument("--isVideo", type=bool, default=False, help="Indicates if the path is a video")
    parser_track.add_argument('--outputDir', type=str, default="outputFiles", help='Directory for outputs in eval mode')
    parser_track.add_argument('--device', type=str, default="cuda:0", help='CUDA device')
    parser_track.add_argument('--detector', type=str, default="FasterRCNN_pretrained", help='Object detector model to use')

def setup_player_classification_parser(subparsers):
    """Configure the subparser for the 'player_classification' task"""
    parser_player_class = subparsers.add_parser('player_classification', help='Identify the team of each player')
    parser_player_class.add_argument('--file', type=str, help='Input image for team clustering')

def setup_pass_detect(subparsers):
    """Configure the subparser for the 'pass_detection' task"""
    parser_pass_detect = subparsers.add_parser('pass_detection', help='Main task: Detect passes')
    parser_pass_detect.add_argument('--files', type=str, help='Video files to process or folder path with images')
    parser_pass_detect.add_argument("--conf-threshold", type=float, default=0.8, help="Object confidence threshold")
    parser_pass_detect.add_argument("--track-points", type=str, default="bbox", help="Tracking points: 'centroid' or 'bbox'")
    parser_pass_detect.add_argument("--distance-threshold", type=float, default=0.04, help="Distance threshold for tracking")
    parser_pass_detect.add_argument("--distance-function", type=str, default="scalar" , help="Distance function: 'scalar' or 'iou'")
    parser_pass_detect.add_argument("--backbone", type=str, default="resnet50v2", help="Backbone for object detector")
    parser_pass_detect.add_argument("--isVideo", type=bool, default=False, help="Indicates if the path is a video")
    parser_pass_detect.add_argument('--device', type=str, default="cuda:0", help='CUDA device')
    parser_pass_detect.add_argument('--detector', type=str, default="FasterRCNN_pretrained", help='Object detector model to use')
    parser_pass_detect.add_argument("--testMode", type=bool, default=False, help="Enable test mode for debugging and saving images")

def main():
    parser = argparse.ArgumentParser(description='Choose a task to run')
    subparsers = parser.add_subparsers(dest='task', help='Task to execute')

    # Configure subparsers for each task
    setup_track_parser(subparsers)
    setup_player_classification_parser(subparsers)
    setup_pass_detect(subparsers)
    subparsers.add_parser('default_task', help='Default task')

    args = parser.parse_args()

    # Execute the selected task
    if args.task == 'track':
        track(
            args.files, 
            args.conf_threshold,  
            args.track_points,
            args.distance_threshold,
            args.distance_function,
            args.backbone,
            args.draw,
            args.evalFile,
            args.isVideo,
            args.device,
            args.detector,
            args.outputDir,
        )
    elif args.task == 'player_classification':
        classifier = PlayerClassifier()
        # Run classification task to identify player teams
        classifier.classify(image_path=args.file, visualize=True)
    elif args.task == 'pass_detection':
        print('Running Pass Detection')
        pass_detection = PassDetection(
            args.files,
            args.conf_threshold,  
            args.track_points,
            args.distance_threshold,
            args.distance_function,
            args.backbone,
            args.isVideo,
            args.device,
            args.detector,
            args.testMode,
        )
        pass_detection.detect_passes()
    else:
        # Display help if no valid task is provided
        parser.print_help()

if __name__ == "__main__":
    main()
