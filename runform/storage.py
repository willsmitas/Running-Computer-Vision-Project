"""JSON-on-disk persistence for longitudinal use.

Layout under the data root (default ./data):

    <user_id>/profile.json                user profile + baseline setup + speeds
    <user_id>/session_001/session.json    session record
    <user_id>/session_001/assessment.json interpretation payload (+ narrative)
    <user_id>/session_001/plan.json       training plan
    <user_id>/session_001/<artifacts>     videos / CSVs / metrics

Plain files, no database: sessions are few, records are small, and
being able to eyeball every artifact in a file explorer is worth more
right now than query power.
"""

import json
import os
import re

DEFAULT_ROOT = "data"

_SESSION_DIR_RE = re.compile(r"session_(\d{3})$")


class Store:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = root

    # -- helpers ----------------------------------------------------------

    def _user_dir(self, user_id):
        return os.path.join(self.root, user_id)

    @staticmethod
    def save_json(path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2)
        return path

    @staticmethod
    def load_json(path):
        with open(path) as fh:
            return json.load(fh)

    # -- profile / baseline setup ----------------------------------------

    def profile_path(self, user_id):
        return os.path.join(self._user_dir(user_id), "profile.json")

    def save_profile(self, user_id, profile_dict):
        return self.save_json(self.profile_path(user_id), profile_dict)

    def load_profile(self, user_id):
        path = self.profile_path(user_id)
        return self.load_json(path) if os.path.exists(path) else None

    def baseline_setup(self, user_id):
        """The stored session-1 camera setup, for replaying to the user at
        every subsequent session (setup drift masquerades as form drift)."""
        profile = self.load_profile(user_id) or {}
        return profile.get("baseline_setup")

    # -- sessions ---------------------------------------------------------

    def list_session_dirs(self, user_id):
        udir = self._user_dir(user_id)
        if not os.path.isdir(udir):
            return []
        found = []
        for name in os.listdir(udir):
            m = _SESSION_DIR_RE.match(name)
            if m and os.path.isdir(os.path.join(udir, name)):
                found.append((int(m.group(1)), os.path.join(udir, name)))
        return [path for _, path in sorted(found)]

    def new_session_dir(self, user_id):
        """Create the next session_NNN directory; returns (number, path)."""
        existing = self.list_session_dirs(user_id)
        number = len(existing) + 1
        path = os.path.join(self._user_dir(user_id), f"session_{number:03d}")
        os.makedirs(path, exist_ok=True)
        return number, path

    def latest_session_dir(self, user_id):
        dirs = self.list_session_dirs(user_id)
        return dirs[-1] if dirs else None
