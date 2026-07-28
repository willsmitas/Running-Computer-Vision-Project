"""Single source of truth for the reduced 23-point landmark scheme.

pose.py WRITES CSV columns in this order and metrics.py READS them by
these names. The index order is a shared invariant (CLAUDE.md): changing
it in one consumer silently corrupts the other, so every module must
import the scheme from here and nowhere else.
"""

# Raw MediaPipe face landmarks (indices 0-10: nose, eyes, ears, mouth
# corners) that get collapsed into one averaged "head" point. The nose
# alone is noisy in profile view and jitters frame to frame; a head
# centroid is a much stabler anchor for the skeleton.
NUM_FACE_LANDMARKS = 11

# Reduced set: index 0 = averaged head, indices 1-22 = body landmarks in
# MediaPipe order starting from left_shoulder.
LANDMARK_NAMES = [
    "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Skeleton "bones" as (start_idx, end_idx) pairs in the REDUCED scheme.
POSE_CONNECTIONS = [
    (0, 1), (0, 2),                     # head -> left/right shoulder
    (1, 2), (1, 3), (3, 5), (5, 7), (5, 9), (5, 11), (7, 9),
    (2, 4), (4, 6), (6, 8), (6, 10), (6, 12), (8, 10),
    (1, 13), (2, 14), (13, 14), (13, 15), (15, 17), (17, 19), (17, 21),
    (19, 21), (14, 16), (16, 18), (18, 20), (18, 22), (20, 22),
]
