# Autor: Benjamin Gutierrez Padilla


import argparse
from tasks.track import track


if __name__ == "__main__":
    print("Hello World")
    
    parser = argparse.ArgumentParser(description='Choose task to run')

    # Crea los subparsers
    subparsers = parser.add_subparsers(dest='task', help='Choose task to run', default='default_task')
    
    ############ Crea el parser para la tarea "Track" ###########################
    parser_track = subparsers.add_parser('track', help='Track objects in files')
    parser_track.add_argument('--files', type=str, help='Video files to process')
    parser_track.add_argument(
        "--conf-threshold",
        type=float,
        default="0.8",
        help="Object confidence threshold",
    )
    parser_track.add_argument(
        "--track-points",
        type=str,
        default="bbox",
        help="Track points: 'centroid' or 'bbox'",
    )
    
    
    ## TODO: Encontrar como implementar device
    # parser_track.add_argument(
    #     "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
    # )
    

    
    
    # Cambiar a otras tareas
    ############ Crea el parser para la tarea "Task23" ###########################
    parser_task2 = subparsers.add_parser('task2', help='Help for task2')
    parser_task2.add_argument('--task2_arg1', type=str, help='Argument 1 for task2')
    parser_task2.add_argument('--task2_arg2', type=str, help='Argument 2 for task2')

    # Crea el parser para la tarea por defecto
    parser_default = subparsers.add_parser('default_task', help='Default task')

    args = parser.parse_args()

    if args.task == 'track':
        print(args)
        print(args.files)
        
        track(
            args.files, 
            args.conf_threshold,  
            args.track_points
        )
        
    elif args.task == 'task2':
        print(args.task2_arg1)
        print(args.task2_arg2)
    elif args.task == 'default_task':
        print('Running default task')