import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    bboxes1: (N, 4) in [x1, y1, x2, y2]
    bboxes2: (M, 4) in [x1, y1, x2, y2]
    Returns: (N, M) matrix of IoU values.
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    bboxes1 = np.expand_dims(bboxes1, 1)  # (N, 1, 4)
    bboxes2 = np.expand_dims(bboxes2, 0)  # (1, M, 4)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)

    intersection = w * h
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    union = area1 + area2 - intersection
    union = np.maximum(union, 1e-6)

    return (intersection / union).astype(np.float32)


def iou_distance(tracks: list, detections: list) -> np.ndarray:
    """
    Computes IoU distance (1 - IoU) matrix between tracks and detections.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_boxes = np.array([t.tlbr for t in tracks], dtype=np.float32)
    det_boxes = np.array([d.tlbr for d in detections], dtype=np.float32)

    ious = iou_batch(track_boxes, det_boxes)
    cost_matrix = 1.0 - ious
    return cost_matrix


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solves linear sum assignment problem using Hungarian algorithm.
    Filters out matches with cost > thresh.

    Returns:
        matches: List of (track_idx, det_idx)
        unmatched_tracks: List of track_idx
        unmatched_detections: List of det_idx
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_detections = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_tracks:
                unmatched_tracks.remove(r)
            if c in unmatched_detections:
                unmatched_detections.remove(c)

    return matches, unmatched_tracks, unmatched_detections
