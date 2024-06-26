import numpy as np

import norfair


def draw(
    paths_drawer,
    track_points,
    frame,
    detections,
    tracked_objects,
    coord_transformations,
    fix_paths,
):
    if track_points == "centroid":
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
    elif track_points == "bbox":
        norfair.draw_boxes(frame, detections)
        # norfair.draw_tracked_boxes(frame, tracked_objects)
        norfair.draw_boxes(frame=frame, drawables=tracked_objects, draw_labels = True, draw_ids = True)
        pass

    if fix_paths:
        frame = paths_drawer.draw(frame, tracked_objects, coord_transformations)
        pass
    elif paths_drawer is not None:
        frame = paths_drawer.draw(frame, tracked_objects)
        pass

    return frame


def center(points):
    return [np.mean(np.array(points), axis=0)]
