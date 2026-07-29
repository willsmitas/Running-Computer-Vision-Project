"""Thin shim kept for the original workflow. All logic now lives in
runform/pose.py — edit there, not here. The 23-point landmark scheme both
stages share is defined once in runform/landmarks.py.

Usage:
    python pose_skeleton_starter.py input_video.mp4
"""

from runform.pose import main

if __name__ == "__main__":
    main()
