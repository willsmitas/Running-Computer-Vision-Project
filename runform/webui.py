"""Phase 6: local web UI — stdlib http.server, zero new dependencies.

Serves a single-page app (runform/static/index.html) plus a JSON API
over the same Store layout the CLI uses. The UI is a veneer: every
route delegates to the existing phase modules (pipeline, session,
narrative, plan, comparison), so the CLI and UI can never disagree.

    python -m runform ui            # http://127.0.0.1:8177

Single-user by design (local app, one runner). Videos upload as raw
request bodies (no multipart) into data/<user>/session_NNN/<label>.<ext>;
pipeline artifacts land next to them and are served back under /media/
with Range support so <video> seeking works.

Pose estimation runs in a background thread with progress reported via
/api/jobs — BUILD_PLAN: "pose estimation is slow; show it".
"""

import json
import mimetypes
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from .errors import RunFormError
from .references import SPEED_BANDS
from .storage import Store

DEFAULT_USER = "default"
DEFAULT_PORT = 8177

ALLOWED_VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi")
# Cap uploads well above any sane treadmill clip; refuses runaway bodies.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

_SESSION_NUM_RE = re.compile(r"^\d{1,3}$")

# In-memory job registry for background analysis runs. Jobs are
# ephemeral by design: a restart mid-analysis just means re-clicking
# Analyze; artifacts on disk are the durable record.
_jobs = {}
_jobs_lock = threading.Lock()
# Serializes writes to profile.json / session.json.
_state_lock = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_update(job_id, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_analysis(job_id, video_path, session_dir, label, smooth, mode):
    """Background worker: pose + metrics on one clip, then record the
    artifact facts into session.json."""
    from .pipeline import analyze_clip  # lazy: rtmlib/onnxruntime/cv2

    def cb(done, total):
        fields = {"frames_done": done, "frames_total": total}
        if total:
            fields["progress"] = min(done / total, 1.0)
        _job_update(job_id, **fields)

    try:
        _job_update(job_id, stage="pose estimation")
        result = analyze_clip(video_path, out_dir=session_dir,
                              mode=mode, smooth=smooth, progress_cb=cb)
        with _state_lock:
            sess = _load_session_json(session_dir)
            clip = sess["clips"].setdefault(label, {})
            clip.update({
                "analyzed_at": _now_iso(),
                "skeleton_video": os.path.basename(result["skeleton_video_path"]),
                "landmarks_csv": os.path.basename(result["landmarks_csv_path"]),
                "metrics_json": os.path.basename(result["metrics_json_path"]),
                "quality_json": os.path.basename(result["quality_json_path"]),
                "detection_rate": result["detection_rate"],
                "quality_flags": result["quality_flags"],
                "fps": result["fps"], "width": result["width"],
                "height": result["height"], "duration_s": result["duration_s"],
            })
            _save_session_json(session_dir, sess)
        _job_update(job_id, status="done", progress=1.0,
                    quality_flags=result["quality_flags"],
                    detection_rate=result["detection_rate"])
    except Exception as e:  # surfaced verbatim in the UI — fail loudly
        _job_update(job_id, status="error", error=str(e))


def _load_session_json(session_dir):
    return Store.load_json(os.path.join(session_dir, "session.json"))


def _save_session_json(session_dir, sess):
    Store.save_json(os.path.join(session_dir, "session.json"), sess)


def _invalidate_derived(session_dir):
    """A changed clip makes the session's interpretation stale. Honest
    exclusion beats stale output: drop assessment/plan; both are
    regenerable in one click."""
    for name in ("assessment.json", "plan.json"):
        path = os.path.join(session_dir, name)
        if os.path.exists(path):
            os.remove(path)


class Api:
    """Route logic, kept separate from the HTTP plumbing for testability."""

    def __init__(self, store, user_id=DEFAULT_USER):
        self.store = store
        self.user_id = user_id

    # -- helpers ----------------------------------------------------------

    def _session_dir(self, number):
        path = os.path.join(self.store.root, self.user_id,
                            f"session_{int(number):03d}")
        if not os.path.isdir(path):
            raise RunFormError(f"No such session: {number}")
        return path

    def _clip_state(self, session_dir, label, entry):
        rel = os.path.basename(session_dir)
        out = dict(entry)
        if entry.get("video"):
            out["video_url"] = f"/media/{rel}/{entry['video']}"
        if entry.get("skeleton_video"):
            out["skeleton_url"] = f"/media/{rel}/{entry['skeleton_video']}"
        mpath = os.path.join(session_dir, entry.get("metrics_json") or "")
        if entry.get("metrics_json") and os.path.exists(mpath):
            out["metrics"] = Store.load_json(mpath)
        return out

    # -- reads ------------------------------------------------------------

    def state(self):
        sessions = []
        for sdir in self.store.list_session_dirs(self.user_id):
            spath = os.path.join(sdir, "session.json")
            if not os.path.exists(spath):
                continue
            sess = Store.load_json(spath)
            entry = {
                "number": sess["session_number"],
                "recorded_at": sess.get("recorded_at"),
                "notes": sess.get("notes", ""),
                "setup_confirmed": sess.get("setup_confirmed", False),
                "clips": {
                    label: self._clip_state(sdir, label, c)
                    for label, c in sess.get("clips", {}).items()
                },
            }
            for name in ("assessment", "plan"):
                path = os.path.join(sdir, f"{name}.json")
                entry[name] = Store.load_json(path) if os.path.exists(path) else None
            sessions.append(entry)
        with _jobs_lock:
            jobs = {k: dict(v) for k, v in _jobs.items()}
        return {
            "today": datetime.now().strftime("%Y-%m-%d"),
            "speed_bands": list(SPEED_BANDS),
            "profile": self.store.load_profile(self.user_id),
            "sessions": sessions,
            "jobs": jobs,
        }

    def compare(self, before_n, after_n):
        from .comparison import compare_sessions

        payloads = []
        for n in (before_n, after_n):
            path = os.path.join(self._session_dir(n), "assessment.json")
            if not os.path.exists(path):
                raise RunFormError(
                    f"Session {n} has no assessment yet — run interpretation "
                    f"on it first."
                )
            payloads.append(Store.load_json(path))
        result = compare_sessions(*payloads)
        result["before_session"] = int(before_n)
        result["after_session"] = int(after_n)
        return result

    # -- writes -----------------------------------------------------------

    def save_profile(self, body):
        with _state_lock:
            profile = self.store.load_profile(self.user_id) or {
                "id": self.user_id, "created_at": _now_iso(),
            }
            for key in ("height_cm", "experience_level", "baseline_setup",
                        "speeds", "speed_unit"):
                if key in body:
                    profile[key] = body[key]
            self.store.save_profile(self.user_id, profile)
        return profile

    def create_session(self, body):
        with _state_lock:
            number, path = self.store.new_session_dir(self.user_id)
            sess = {
                "id": f"session_{number:03d}",
                "user_id": self.user_id,
                "recorded_at": _now_iso(),
                "session_number": number,
                "setup_confirmed": bool(body.get("setup_confirmed", False)),
                "notes": body.get("notes", ""),
                "clips": {},
            }
            _save_session_json(path, sess)
        return {"number": number}

    def update_session(self, number, body):
        sdir = self._session_dir(number)
        with _state_lock:
            sess = _load_session_json(sdir)
            for key in ("notes", "setup_confirmed"):
                if key in body:
                    sess[key] = body[key]
            for label, fields in (body.get("clips") or {}).items():
                if label not in SPEED_BANDS:
                    raise RunFormError(f"Unknown speed label: {label}")
                clip = sess["clips"].setdefault(label, {})
                for key in ("speed_value", "speed_unit"):
                    if key in fields:
                        clip[key] = fields[key]
            _save_session_json(sdir, sess)
        return sess

    def upload(self, number, label, filename, stream, length):
        if label not in SPEED_BANDS:
            raise RunFormError(f"Unknown speed label: {label}")
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            raise RunFormError(
                f"Unsupported video type '{ext}'. "
                f"Expected one of: {', '.join(ALLOWED_VIDEO_EXTS)}"
            )
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise RunFormError(f"Bad upload size: {length} bytes")
        sdir = self._session_dir(number)
        dest = os.path.join(sdir, f"{label}{ext}")

        remaining = length
        with open(dest, "wb") as fh:
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    raise RunFormError("Upload ended early — try again.")
                fh.write(chunk)
                remaining -= len(chunk)

        with _state_lock:
            sess = _load_session_json(sdir)
            old = sess["clips"].get(label, {})
            # A new video invalidates this clip's old artifacts and the
            # session-level interpretation built on them.
            for key in ("skeleton_video", "landmarks_csv", "metrics_json",
                        "quality_json"):
                if old.get(key):
                    stale = os.path.join(sdir, old[key])
                    if os.path.exists(stale):
                        os.remove(stale)
            if old.get("video") and old["video"] != os.path.basename(dest):
                stale = os.path.join(sdir, old["video"])
                if os.path.exists(stale):
                    os.remove(stale)
            sess["clips"][label] = {
                "video": os.path.basename(dest),
                "original_filename": filename,
                "speed_value": old.get("speed_value"),
                "speed_unit": old.get("speed_unit"),
                "uploaded_at": _now_iso(),
            }
            _save_session_json(sdir, sess)
            _invalidate_derived(sdir)
        return {"ok": True, "video": os.path.basename(dest)}

    def start_analysis(self, number, body):
        label = body.get("label")
        sdir = self._session_dir(number)
        sess = _load_session_json(sdir)
        clip = sess.get("clips", {}).get(label)
        if not clip or not clip.get("video"):
            raise RunFormError(f"No video uploaded for '{label}' yet.")
        with _jobs_lock:
            for j in _jobs.values():
                if (j["status"] == "running" and j["session"] == int(number)
                        and j["label"] == label):
                    return {"job_id": j["id"]}  # already running; don't stack
        job_id = uuid.uuid4().hex[:12]
        with _jobs_lock:
            _jobs[job_id] = {
                "id": job_id, "session": int(number), "label": label,
                "status": "running", "stage": "loading pose model",
                "progress": None, "frames_done": 0, "frames_total": None,
                "started_at": _now_iso(),
            }
        with _state_lock:
            _invalidate_derived(sdir)
        video_path = os.path.join(sdir, clip["video"])
        t = threading.Thread(
            target=_run_analysis,
            args=(job_id, video_path, sdir, label,
                  int(body.get("smooth", 9)), body.get("mode", "balanced")),
            daemon=True,
        )
        t.start()
        return {"job_id": job_id}

    def interpret(self, number):
        from .session import interpret_session

        sdir = self._session_dir(number)
        sess = _load_session_json(sdir)
        clips = []
        for label in SPEED_BANDS:
            entry = sess.get("clips", {}).get(label)
            if not entry or not entry.get("metrics_json"):
                continue
            if entry.get("speed_value") in (None, ""):
                raise RunFormError(
                    f"Clip '{label}' has no treadmill speed entered. The "
                    f"speed profile needs it — fill it in and retry."
                )
            clips.append({
                "speed_label": label,
                "speed_value": float(entry["speed_value"]),
                "metrics": Store.load_json(os.path.join(sdir, entry["metrics_json"])),
                "detection_rate": entry.get("detection_rate"),
                "quality_flags": entry.get("quality_flags", []),
            })
        if not clips:
            raise RunFormError("No analyzed clips in this session yet.")
        assessment = interpret_session(clips, notes=sess.get("notes", ""))
        with _state_lock:
            Store.save_json(os.path.join(sdir, "assessment.json"), assessment)
        return assessment

    def narrate(self, number, body):
        from .narrative import narrate

        sdir = self._session_dir(number)
        path = os.path.join(sdir, "assessment.json")
        if not os.path.exists(path):
            raise RunFormError("Run interpretation first — the narrative is "
                               "written on top of the deterministic assessment.")
        assessment = Store.load_json(path)
        try:
            result = narrate(
                assessment,
                model=body.get("model", "llama3.1:8b"),
                host=body.get("host", "http://localhost:11434"),
            )
        except OSError as e:
            raise RunFormError(
                f"Could not reach Ollama ({e}). Is it running? Start it, "
                f"then retry."
            )
        assessment["narrative"] = result
        with _state_lock:
            Store.save_json(path, assessment)
        return result

    def build_plan(self, number):
        from .plan import build_plan

        sdir = self._session_dir(number)
        path = os.path.join(sdir, "assessment.json")
        if not os.path.exists(path):
            raise RunFormError("Run interpretation first — the plan is built "
                               "from the assessment's root causes.")
        plan = build_plan(Store.load_json(path))
        with _state_lock:
            Store.save_json(os.path.join(sdir, "plan.json"), plan)
        return plan


class Handler(BaseHTTPRequestHandler):
    # Class attributes injected by run()
    api = None
    static_dir = None
    media_root = None
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):  # quiet: progress lives in the UI
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def _serve_file(self, path, cacheable=False):
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        # .mov files are typically H.264 in a QuickTime container;
        # video/mp4 lets browsers try them instead of refusing outright.
        if path.lower().endswith(".mov"):
            ctype = "video/mp4"
        size = os.path.getsize(path)
        start, end = 0, size - 1
        status = 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if m and (m.group(1) or m.group(2)):
                start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
                if m.group(1) and m.group(2):
                    end = min(int(m.group(2)), size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control",
                         "max-age=3600" if cacheable else "no-store")
        # One response per connection for file bodies: Chrome's media
        # stack pauses/aborts range reads mid-body, and a paused reader on
        # a keep-alive connection wedges the handler thread in write()
        # while follow-up range requests queue behind it forever.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionError, BrokenPipeError, OSError):
            pass  # client cancelled (normal for video seeks) — not an error

    def _dispatch(self, fn):
        try:
            self._send_json(fn())
        except RunFormError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": f"{type(e).__name__}: {e}"}, status=500)

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)

        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(self.static_dir, "index.html"))

        if path == "/api/state":
            return self._dispatch(self.api.state)

        if path == "/api/compare":
            q = parse_qs(url.query)
            before = (q.get("before") or [""])[0]
            after = (q.get("after") or [""])[0]
            if not (_SESSION_NUM_RE.match(before) and _SESSION_NUM_RE.match(after)):
                return self._send_json({"error": "before/after session numbers required"}, 400)
            return self._dispatch(lambda: self.api.compare(before, after))

        if path.startswith("/media/"):
            rel = os.path.normpath(path[len("/media/"):]).replace("\\", "/")
            full = os.path.normpath(os.path.join(self.media_root, rel))
            # Confine to the user's data directory — no traversal.
            if not full.startswith(os.path.normpath(self.media_root) + os.sep):
                return self._send_json({"error": "forbidden"}, 403)
            if not os.path.isfile(full):
                return self._send_json({"error": "not found"}, 404)
            return self._serve_file(full, cacheable=True)

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        path = unquote(url.path)

        if path == "/api/profile":
            body = self._read_json_body()
            return self._dispatch(lambda: self.api.save_profile(body))

        if path == "/api/sessions":
            body = self._read_json_body()
            return self._dispatch(lambda: self.api.create_session(body))

        m = re.match(r"^/api/sessions/(\d{1,3})/(\w+)$", path)
        if m:
            number, action = m.group(1), m.group(2)
            if action == "upload":
                q = parse_qs(url.query)
                label = (q.get("label") or [""])[0]
                filename = (q.get("filename") or ["clip.mp4"])[0]
                length = int(self.headers.get("Content-Length") or 0)
                return self._dispatch(
                    lambda: self.api.upload(number, label, filename,
                                            self.rfile, length))
            body = self._read_json_body()
            actions = {
                "update": lambda: self.api.update_session(number, body),
                "analyze": lambda: self.api.start_analysis(number, body),
                "interpret": lambda: self.api.interpret(number),
                "narrate": lambda: self.api.narrate(number, body),
                "plan": lambda: self.api.build_plan(number),
            }
            if action in actions:
                return self._dispatch(actions[action])

        self._send_json({"error": "not found"}, 404)


def run(host="127.0.0.1", port=DEFAULT_PORT, root="data",
        user_id=DEFAULT_USER, open_browser=True):
    store = Store(root)
    os.makedirs(os.path.join(root, user_id), exist_ok=True)
    Handler.api = Api(store, user_id)
    Handler.static_dir = os.path.join(os.path.dirname(__file__), "static")
    Handler.media_root = os.path.join(root, user_id)
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"runform UI at {url}  (data root: {os.path.abspath(root)})")
    print("Ctrl+C to stop.")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
