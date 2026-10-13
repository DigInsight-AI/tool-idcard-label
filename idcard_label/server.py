import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__
from .model_refresh import LocalModelRefresher, ModelRefreshError
from .store import ConflictError, ReviewStore


ITEM_PATTERN = re.compile(r"^/api/items/(-?\d+)$")
IMAGE_PATTERN = re.compile(r"^/api/images/(-?\d+)$")
REFRESH_PATTERN = re.compile(r"^/api/items/(-?\d+)/refresh$")
MODELS_PATH = "/api/models"
MODEL_DIRECTORY_PATH = "/api/models/directory"
ARCHIVE_EXPORT_PATH = "/api/archives/export"
BATCH_ACTIVATE_PATTERN = re.compile(r"^/api/review-batches/([^/]+)/activate$")
DEFAULT_INPUT_ROOT = "data/idcard"
DEFAULT_OUTPUT_ROOT = "data/workspace"
DEFAULT_MODEL_DIRECTORY = "data/model"


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        store=None,
        model_refresher=None,
        preferences_path=None,
        workspace_class_id=None,
        refresh_confidence=0.25,
        refresh_keypoint_confidence=0.5,
        refresh_device=None,
    ):
        super().__init__(address, ReviewHandler)
        self.store = store
        self.static_root = Path(__file__).parent / "static"
        self.workspace_class_id = workspace_class_id
        self.preferences_path = Path(preferences_path or _default_preferences_path()).expanduser()
        self.directory_preferences = _load_preferences(self.preferences_path)
        self.refresh_confidence = refresh_confidence
        self.refresh_keypoint_confidence = refresh_keypoint_confidence
        self.refresh_device = refresh_device
        self.model_refresher = model_refresher or self._preferred_model_refresher()
        self.directory_error = None
        if not self.store:
            self._restore_preferred_workspace()

    def replace_store(self, store):
        previous = self.store
        self.store = store
        if previous:
            previous.close()

    def close_store(self):
        if self.store:
            self.store.close()
            self.store = None

    def config(self):
        if self.store:
            config = self.store.config()
            config["configured"] = True
        else:
            config = {
                "configured": False,
                "classes": [],
                "input_root": self.directory_preferences.get("input_root", DEFAULT_INPUT_ROOT),
                "output_root": self.directory_preferences.get("output_root", DEFAULT_OUTPUT_ROOT),
                "directory_error": self.directory_error,
            }
        if self.model_refresher:
            config["model_refresh"] = self.model_refresher.describe()
        config["model_directory"] = self.directory_preferences.get(
            "model_directory", DEFAULT_MODEL_DIRECTORY
        )
        config["export_root"] = self.directory_preferences.get("export_root")
        config["export_format"] = self.directory_preferences.get("export_format", "yolo_pose")
        config["app_version"] = __version__
        return config

    def set_model_directory(self, model_directory):
        directory_text = str(model_directory or "").strip()
        if not directory_text:
            raise ValueError("model directory is required")
        directory = Path(directory_text).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("model directory does not exist")
        model = self.directory_preferences.get("model") or "best.pt"
        refresher = LocalModelRefresher(
            model_dir=directory,
            model=model,
            confidence=self.refresh_confidence,
            keypoint_confidence=self.refresh_keypoint_confidence,
            device=self.refresh_device,
        )
        description = refresher.describe()
        if not description["selected_exists"]:
            available = [entry["name"] for entry in description["models"] if entry["modified_ns"] is not None]
            if available:
                description = refresher.select_model(available[0])
        self.model_refresher = refresher
        self._save_preferences({
            "model_directory": description["directory"],
            "model": description["selected"],
        })
        return description

    def select_model(self, model):
        if not self.model_refresher:
            raise ModelRefreshError("configure a local model directory first")
        description = self.model_refresher.select_model(model)
        self._save_preferences({"model": description["selected"]})
        return description

    def export_archive_batch(self, archive_batch, destination_root, export_format="yolo_pose"):
        result = self.store.export_archive_batch(archive_batch, destination_root, export_format)
        export_root = str(Path(destination_root).expanduser().resolve())
        self._save_preferences({
            "export_root": export_root,
            "export_format": result["export_format"],
        })
        result["export_root"] = export_root
        return result

    def open_workspace(self, input_root, output_root, confirm_create=False):
        input_root, output_root = _validate_workspace_directories(
            input_root, output_root
        )
        database_path = output_root / "review.sqlite3"
        if not database_path.is_file() and not confirm_create:
            raise WorkspaceConfirmationRequired(output_root)
        candidate = ReviewStore(output_root, self.workspace_class_id)
        try:
            candidate.initialize(input_root)
            preferences = {
                "input_root": str(input_root),
                "output_root": str(output_root),
            }
            self._save_preferences(preferences)
        except Exception:
            candidate.close()
            raise
        self.replace_store(candidate)
        self.directory_error = None
        return self.config()

    def _restore_preferred_workspace(self):
        input_root = self.directory_preferences.get("input_root")
        output_root = self.directory_preferences.get("output_root")
        if not input_root and not output_root:
            return
        try:
            input_root, output_root = _validate_workspace_directories(
                input_root, output_root
            )
            if not (output_root / "review.sqlite3").is_file():
                raise ValueError("selected output directory has no review workspace")
            candidate = ReviewStore(output_root, self.workspace_class_id)
            try:
                candidate.initialize(input_root)
            except Exception:
                candidate.close()
                raise
            self.store = candidate
        except (OSError, ValueError) as error:
            self.directory_error = str(error)

    def _preferred_model_refresher(self):
        model_directory = self.directory_preferences.get(
            "model_directory", DEFAULT_MODEL_DIRECTORY
        )
        try:
            return LocalModelRefresher(
                model_dir=model_directory,
                model=self.directory_preferences.get("model") or "best.pt",
                confidence=self.refresh_confidence,
                keypoint_confidence=self.refresh_keypoint_confidence,
                device=self.refresh_device,
            )
        except (OSError, ValueError):
            return None

    def _save_preferences(self, updates):
        preferences = dict(self.directory_preferences)
        preferences.update(updates)
        _save_preferences(self.preferences_path, preferences)
        self.directory_preferences = preferences


class WorkspaceConfirmationRequired(ValueError):
    def __init__(self, output_root):
        super().__init__("confirm creating a new review workspace at %s" % output_root)
        self.output_root = output_root


def _default_preferences_path():
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "idcard-label" / "directories.json"


def _load_preferences(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: str(payload[key]).strip()
        for key in (
            "input_root", "output_root", "model_directory", "model", "export_root", "export_format"
        )
        if str(payload.get(key) or "").strip()
    }


def _save_preferences(path, preferences):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(preferences, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_workspace_directories(input_root, output_root):
    input_text = str(input_root or "").strip()
    output_text = str(output_root or "").strip()
    if not input_text:
        raise ValueError("input directory is required")
    if not output_text:
        raise ValueError("output directory is required")
    input_path = Path(input_text).expanduser().resolve()
    output_path = Path(output_text).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError("input directory does not exist")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output path must be a directory")
    return input_path, output_path


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "IDCardReview/%s" % __version__

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            return self._json(self.server.config())
        if parsed.path == MODELS_PATH:
            if not self.server.model_refresher:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "model refresh is not configured")
            return self._json(self.server.model_refresher.describe())
        if not self.server.store and parsed.path.startswith("/api/"):
            return self._workspace_not_open()
        if parsed.path == "/api/items":
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            return self._json(self.server.store.list_items(query))
        match = ITEM_PATTERN.match(parsed.path)
        if match:
            item = self.server.store.get_item(int(match.group(1)))
            if not item:
                return self._error(HTTPStatus.NOT_FOUND, "item not found")
            return self._json(item)
        match = IMAGE_PATTERN.match(parsed.path)
        if match:
            return self._image(int(match.group(1)))
        return self._static(parsed.path)

    def do_PUT(self):
        parsed = urlparse(self.path)
        if parsed.path == MODELS_PATH:
            if not self.server.model_refresher:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "model refresh is not configured")
            try:
                payload = self._read_json()
                return self._json(self.server.select_model(payload.get("model")))
            except (ModelRefreshError, ValueError, TypeError, json.JSONDecodeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        if not self.server.store:
            return self._workspace_not_open()
        match = ITEM_PATTERN.match(parsed.path)
        if not match:
            return self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
        try:
            payload = self._read_json()
            item = self.server.store.update_item(int(match.group(1)), payload)
            if not item:
                return self._error(HTTPStatus.NOT_FOUND, "item not found")
            return self._json(item)
        except ConflictError as error:
            return self._error(HTTPStatus.CONFLICT, str(error))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return self._error(HTTPStatus.BAD_REQUEST, str(error))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == MODEL_DIRECTORY_PATH:
            try:
                payload = self._read_json()
                return self._json(self.server.set_model_directory(payload.get("model_directory")))
            except (ModelRefreshError, ValueError, TypeError, json.JSONDecodeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        if parsed.path == "/api/workspace/open":
            try:
                payload = self._read_json()
                return self._json(self.server.open_workspace(
                    payload.get("input_root"),
                    payload.get("output_root"),
                    bool(payload.get("confirm_create")),
                ))
            except WorkspaceConfirmationRequired as error:
                return self._json(
                    {"error": str(error), "requires_confirmation": True},
                    HTTPStatus.CONFLICT,
                )
            except (ValueError, OSError, json.JSONDecodeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        if not self.server.store:
            return self._workspace_not_open()
        if parsed.path == "/api/review-batches":
            try:
                payload = self._read_json()
                return self._json(self.server.store.create_review_batch(payload.get("name")), HTTPStatus.CREATED)
            except (ValueError, json.JSONDecodeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        if parsed.path == "/api/archives/merge":
            try:
                payload = self._read_json()
                return self._json(
                    self.server.store.merge_archives(
                        payload.get("archive_batches"),
                        payload.get("folder_name"),
                        payload.get("destination_archive_batch"),
                    ),
                    HTTPStatus.CREATED,
                )
            except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        if parsed.path == ARCHIVE_EXPORT_PATH:
            try:
                payload = self._read_json()
                return self._json(
                    self.server.export_archive_batch(
                        payload.get("archive_batch"),
                        payload.get("destination_root"),
                        payload.get("export_format", payload.get("format", "yolo_pose")),
                    ),
                    HTTPStatus.CREATED,
                )
            except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        batch_match = BATCH_ACTIVATE_PATTERN.match(parsed.path)
        if batch_match:
            try:
                return self._json(self.server.store.activate_review_batch(batch_match.group(1)))
            except ValueError as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        match = REFRESH_PATTERN.match(parsed.path)
        if match:
            if not self.server.model_refresher:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "model refresh is not configured")
            item = self.server.store.get_item(int(match.group(1)))
            if not item:
                return self._error(HTTPStatus.NOT_FOUND, "item not found")
            try:
                result = self.server.model_refresher.refresh(item, self.server.store.config()["classes"])
                return self._json(result, HTTPStatus.OK)
            except ModelRefreshError as error:
                return self._error(HTTPStatus.BAD_GATEWAY, str(error))
        if parsed.path == "/api/archive":
            try:
                payload = self._read_json()
                if payload.get("all_candidates"):
                    result = self.server.store.archive_all_candidates(
                        payload.get("review_batch"),
                        confirm_pose_semantics=bool(payload.get("confirm_pose_semantics")),
                        merge_into_archive=payload.get("merge_into_archive"),
                    )
                else:
                    result = self.server.store.archive_saved(payload.get("review_batch"))
                return self._json(
                    result,
                    HTTPStatus.CREATED,
                )
            except (ValueError, OSError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
        return self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def log_message(self, format_string, *args):
        print("%s - %s" % (self.log_date_time_string(), format_string % args))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2 * 1024 * 1024:
            raise ValueError("request body is too large")
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        return self._json({"error": message}, status)

    def _workspace_not_open(self):
        return self._error(HTTPStatus.PRECONDITION_REQUIRED, "choose valid directories first")

    def _image(self, item_id):
        if not self.server.store:
            return self._workspace_not_open()
        path = self.server.store.image_path(item_id)
        if not path or not os.path.isfile(path):
            return self._error(HTTPStatus.NOT_FOUND, "image not found")
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _static(self, request_path):
        relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        candidate = (self.server.static_root / relative).resolve()
        static_root = self.server.static_root.resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError:
            return self._error(HTTPStatus.FORBIDDEN, "invalid path")
        if not candidate.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "file not found")
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
