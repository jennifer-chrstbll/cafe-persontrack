
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Computes IoU between two sets of bboxes.
    bboxes1: (N,4) [x1,y1,x2,y2]
    bboxes2: (M,4) [x1,y1,x2,y2]
    Returns: (N,M) IoU matrix
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    b1 = np.expand_dims(bboxes1, 1)  # (N,1,4)
    b2 = np.expand_dims(bboxes2, 0)  # (1,M,4)

    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 2], b2[..., 2])
    yy2 = np.minimum(b1[..., 3], b2[..., 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    a1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    a2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = np.maximum(a1 + a2 - inter, 1e-6)
    return (inter / union).astype(np.float32)


def iou_distance(tracks: list, detections: list) -> np.ndarray:
    """Cost matrix based purely on 1 - IoU."""
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    tb = np.array([t.tlbr for t in tracks], dtype=np.float32)
    db = np.array([d.tlbr for d in detections], dtype=np.float32)
    return (1.0 - iou_batch(tb, db)).astype(np.float32)


def reid_distance(tracks: list, detections: list,
                  track_feats: Optional[np.ndarray],
                  det_feats: Optional[np.ndarray]) -> np.ndarray:
    """
    Cosine distance cost matrix between track appearance memory and
    detection appearance features.
    Returns matrix of shape (N_tracks, N_dets), values in [0, 2].
    """
    N, M = len(tracks), len(detections)
    if N == 0 or M == 0:
        return np.ones((N, M), dtype=np.float32)  # max cost = unknown

    if track_feats is None or det_feats is None:
        return np.ones((N, M), dtype=np.float32)

    # Normalize rows to unit length for cosine similarity
    tf = track_feats / np.maximum(np.linalg.norm(track_feats, axis=1, keepdims=True), 1e-6)
    df = det_feats  / np.maximum(np.linalg.norm(det_feats,   axis=1, keepdims=True), 1e-6)

    sim = tf @ df.T          # (N, M) cosine similarity
    dist = 1.0 - np.clip(sim, -1.0, 1.0)  # [0, 2], lower = more similar
    return dist.astype(np.float32)


def fused_distance(tracks: list, detections: list,
                   track_feats: Optional[np.ndarray],
                   det_feats: Optional[np.ndarray],
                   reid_weight: float = 0.40) -> np.ndarray:
    """
    Fused cost matrix:  cost = (1-w)*iou_cost + w*reid_cost

    When people cross paths, pure IoU is ambiguous because both detections
    have similar overlap with both tracks. Adding ReID (appearance) cost
    breaks the tie by choosing the visually correct assignment.

    reid_weight=0.40 means 60% IoU + 40% appearance.
    """
    iou_c  = iou_distance(tracks, detections)

    if track_feats is not None and det_feats is not None:
        reid_c = reid_distance(tracks, detections, track_feats, det_feats)
        # Only let ReID contribute where we actually have features (non-one entries)
        has_feat_mask = (reid_c < 1.0).astype(np.float32)
        cost = ((1.0 - reid_weight) * iou_c
                + reid_weight * reid_c * has_feat_mask
                + reid_weight * iou_c * (1.0 - has_feat_mask))
    else:
        cost = iou_c

    return cost.astype(np.float32)


def linear_assignment(cost_matrix: np.ndarray,
                      thresh: float) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Hungarian algorithm. Returns (matches, unmatched_tracks, unmatched_dets).
    Rejects matches whose cost exceeds thresh.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches, u_tr, u_det = [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in u_tr:  u_tr.remove(r)
            if c in u_det: u_det.remove(c)
    return matches, u_tr, u_det
