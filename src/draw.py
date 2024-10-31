"""
    Module extracted from Norfair to draw boxes on tracked objects and trajectories
"""

# Import libraries
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
    """
    Draws bounding boxes or centroids on detected and tracked objects in the frame, 
    and optionally overlays trajectories if specified.

    Parameters:
    - paths_drawer: Object for drawing paths/trajectories.
    - track_points: Determines if tracking is based on 'centroid' or 'bbox'.
    - frame: Current video frame to draw on.
    - detections: List of detected objects.
    - tracked_objects: List of tracked objects.
    - coord_transformations: Coordinate transformations for paths.
    - fix_paths: Boolean, if True applies fixed paths on objects.
    """
    if track_points == "centroid":
        # Draws centroids for detections and tracked objects
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
    elif track_points == "bbox":
        # Draws bounding boxes for detections and tracked objects with IDs and labels
        norfair.draw_boxes(frame, detections)
        norfair.draw_boxes(frame=frame, drawables=tracked_objects, draw_labels=True, draw_ids=True)

    # Draws trajectories if paths are fixed or if paths_drawer is available
    if fix_paths:
        frame = paths_drawer.draw(frame, tracked_objects, coord_transformations)
    elif paths_drawer is not None:
        frame = paths_drawer.draw(frame, tracked_objects)

    return frame

def center(points):
    """
    Calculates the geometric center of given points.

    Parameters:
    - points: List of points to find the center of.

    Returns:
    - Center point as a list with mean coordinates.
    """
    return [np.mean(np.array(points), axis=0)]
