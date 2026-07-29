"""Thin shim kept for the original workflow (formerly saved as
data_analytics_starter.py). All logic now lives in runform/metrics.py —
edit there, not here.

Usage:
    python running_metrics.py clip_landmarks.csv --fps 30 --width 1920 --height 1080
"""

from runform.metrics import main

if __name__ == "__main__":
    main()
