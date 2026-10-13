import math
import os
import tempfile
import threading
from pathlib import Path

from .geometry import KEYPOINT_SEMANTICS
from .pipeline import PipelineError, YOLOPoseDetector, load_image


class ModelRefreshError(Exception):
    pass


class LocalModelRefresher:
    MODEL_SUFFIXES = (".pth", ".pt")

    def __init__(self, model_dir=".", model="best.pt", confidence=0.25, device=None,
                 keypoint_confidence=0.5, detector_factory=YOLOPoseDetector):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.confidence = float(confidence)
        self.device = device
        self.keypoint_confidence = float(keypoint_confidence)
        self.detector_factory = detector_factory
        self._lock = threading.RLock()
        self._selected_model = self._resolve_model_name(model)
        self._detector = None
        self._detector_signature = None

    def describe(self):
        with self._lock:
            selected_model = self._selected_model
        selected_path = self._model_path(selected_model)
        models = []
        if self.model_dir.is_dir():
            for path in self.model_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in self.MODEL_SUFFIXES:
                    continue
                try:
                    relative = path.resolve().relative_to(self.model_dir)
                except ValueError:
                    continue
                stat = path.stat()
                models.append({
                    "name": relative.as_posix(),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                })
        models.sort(key=lambda value: (-value["modified_ns"], value["name"]))
        if not any(value["name"] == selected_model for value in models):
            models.insert(0, {
                "name": selected_model,
                "size": None,
                "modified_ns": None,
            })
        return {
            "directory": str(self.model_dir),
            "selected": selected_model,
            "selected_path": str(selected_path),
            "selected_exists": selected_path.is_file(),
            "models": models,
        }

    def select_model(self, model):
        model_name = self._resolve_model_name(model)
        model_path = self._model_path(model_name)
        if not model_path.is_file():
            raise ModelRefreshError("local model does not exist: %s" % model_path)
        with self._lock:
            if model_name != self._selected_model:
                self._selected_model = model_name
                self._detector = None
                self._detector_signature = None
        return self.describe()

    def refresh(self, item, classes):
        with self._lock:
            model_path = self._model_path(self._selected_model)
            if not model_path.is_file():
                raise ModelRefreshError("local model does not exist: %s" % model_path)
            try:
                stat = model_path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if self._detector is None or signature != self._detector_signature:
                    self._detector = self._load_detector(model_path)
                    self._detector_signature = signature
                predictions = self._detector.predict(load_image(item["image_path"]))
            except (OSError, PipelineError, RuntimeError, ValueError) as error:
                raise ModelRefreshError("local model failed: %s" % error)
        annotations = self.annotations_from_predictions(
            predictions, item, classes, self.keypoint_confidence
        )
        if not annotations:
            raise ModelRefreshError("local model did not recognize the image")
        return {
            "annotations": annotations,
            "model": self._selected_model,
            "detections": len(annotations),
        }

    @staticmethod
    def annotations_from_predictions(predictions, item, classes, keypoint_confidence=0.5):
        class_names = {int(value["id"]): value["name"] for value in classes}
        width = float(item["width"])
        height = float(item["height"])
        annotations = []
        for source_index, prediction in enumerate(predictions):
            points = prediction.get("keypoints")
            if not isinstance(points, list) or len(points) != 4:
                raise ModelRefreshError("local pose model must return exactly four keypoints")
            polygon = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ModelRefreshError("local pose model returned an invalid keypoint")
                try:
                    x, y = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    raise ModelRefreshError("local pose model returned an invalid keypoint")
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ModelRefreshError("local pose model returned a non-finite keypoint")
                polygon.append([max(0.0, min(width, x)), max(0.0, min(height, y))])
            try:
                class_id = int(prediction.get("class_id"))
            except (TypeError, ValueError):
                raise ModelRefreshError("local model returned an invalid class id")
            if class_id not in class_names:
                raise ModelRefreshError("local model returned unknown class id %d" % class_id)
            scores = prediction.get("keypoint_scores") or [1.0] * 4
            if len(scores) != 4:
                raise ModelRefreshError("local pose model returned invalid keypoint scores")
            visibility = [
                2 if float(score) >= float(keypoint_confidence) else 1
                for score in scores
            ]
            annotations.append({
                "class_id": class_id,
                "class_name": class_names[class_id],
                "score": float(prediction["score"]) if prediction.get("score") is not None else None,
                "polygon": polygon,
                "keypoint_visibility": visibility,
                "keypoint_semantics": KEYPOINT_SEMANTICS,
                "source_index": source_index,
            })
        return annotations

    def _normalize_model_name(self, model):
        value = str(model or "").strip()
        if not value:
            raise ModelRefreshError("local model name is required")
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                candidate = candidate.expanduser().resolve().relative_to(self.model_dir)
            except ValueError:
                raise ModelRefreshError("local model must be inside %s" % self.model_dir)
        normalized = candidate.as_posix()
        resolved = (self.model_dir / normalized).resolve()
        try:
            resolved.relative_to(self.model_dir)
        except ValueError:
            raise ModelRefreshError("local model must be inside %s" % self.model_dir)
        if resolved.suffix.lower() not in self.MODEL_SUFFIXES:
            raise ModelRefreshError("local model must use a .pth or .pt extension")
        return resolved.relative_to(self.model_dir).as_posix()

    def _resolve_model_name(self, model):
        model_name = self._normalize_model_name(model)
        if self._model_path(model_name).is_file():
            return model_name
        path = Path(model_name)
        alternate_suffix = ".pt" if path.suffix.lower() == ".pth" else ".pth"
        alternate_name = path.with_suffix(alternate_suffix).as_posix()
        if self._model_path(alternate_name).is_file():
            return alternate_name
        return model_name

    def _model_path(self, model_name):
        return (self.model_dir / model_name).resolve()

    def _load_detector(self, model_path):
        if model_path.suffix.lower() != ".pth":
            return self.detector_factory(str(model_path), self.confidence, self.device)
        with tempfile.TemporaryDirectory(prefix="idcard-model-") as directory:
            alias = Path(directory) / (model_path.stem + ".pt")
            os.symlink(model_path, alias)
            return self.detector_factory(str(alias), self.confidence, self.device)
