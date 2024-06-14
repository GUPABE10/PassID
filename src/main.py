# Autor: Benjamin Gutierrez Padilla

import argparse
from tasks.track import track
from tasks.team_id import PlayerClassifier
from tasks.pass_detection.pass_detection import PassDetection

def setup_track_parser(subparsers):
    """Configura el subparser para la tarea 'track'."""
    parser_track = subparsers.add_parser('track', help='Track objects in files')
    parser_track.add_argument('--files', type=str, help='Video files to process or folder path with images')
    parser_track.add_argument("--conf-threshold", type=float, default=0.8, help="Object confidence threshold")
    parser_track.add_argument("--track-points", type=str, default="bbox", help="Track points: 'centroid' or 'bbox'")
    parser_track.add_argument("--distance-threshold", type=float, default=1.0, help="Distance threshold")
    parser_track.add_argument("--distance-function", type=str, default="scalar", help="Distance function: scalar or iou")
    parser_track.add_argument("--backbone", type=str, default="resnet50v2", help="Backbone for object detector")
    parser_track.add_argument("--draw", type=bool, default=False, help="Generate video drawing")
    parser_track.add_argument("--evalFile", type=bool, default=False, help="Generate file for evaluation")
    parser_track.add_argument("--isVideo", type=bool, default=False, help="The path is a video")
    parser_track.add_argument('--outputDir', type=str, default="outputFiles", help='Directory for outputs in eval mode')
    parser_track.add_argument('--device', type=str, default="cuda:0", help='CUDA Device')
    parser_track.add_argument('--detector', type=str, default="FasterRCNN_pretrained", help='Object Detector Model to be used')

def setup_player_classification_parser(subparsers):
    """Configura el subparser para la tarea 'player_classification'."""
    parser_player_class = subparsers.add_parser('player_classification', help='Identify the team to which each player belongs')
    parser_player_class.add_argument('--file', type=str, help='Input image to be clustered')

def setup_pass_detect(subparsers):
    """Configura el subparser para la tarea 'task2'."""
    parser_pass_detect = subparsers.add_parser('pass_detection', help='Main Task')
    parser_pass_detect.add_argument('--files', type=str, help='Video files to process or folder path with images')
    # parser_pass_detect.add_argument('--task2_arg2', type=str, help='Argument 2 for task2')

def main():
    """Main"""
    
    parser = argparse.ArgumentParser(description='Choose task to run')
    subparsers = parser.add_subparsers(dest='task', help='Choose task to run')

    # Configura los subparsers
    setup_track_parser(subparsers)
    setup_player_classification_parser(subparsers)
    setup_pass_detect(subparsers)
    subparsers.add_parser('default_task', help='Default task')

    args = parser.parse_args()

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
        # classifier.classify_players(args.file)
        classifier.classify(image_path=args.file, visualize=True)
    elif args.task == 'pass_detection':
        print('Running Pass Detection')
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
