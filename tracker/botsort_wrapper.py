"""
tracker/botsort_wrapper.py
--------------------------
Adapts BoxMOT's BotSort (via BotSortTracker) to the STrack interface expected
by pipeline.py, demo_single_cam.py, and multicam_manager.py.

Exposes:
  - BotSortTracker: Drop-in replacement for ByteTracker
  - STrack: Alias / wrapper for BotSortTrack with all required attributes
"""

from tracker.botsort_tracker import BotSortTracker, BotSortTrack

# Expose BotSortTrack as STrack for backward compatibility with pipeline and multicam_manager
STrack = BotSortTrack

__all__ = ["BotSortTracker", "STrack"]
