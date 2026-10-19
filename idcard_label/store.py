import json
import os
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from .formats import (
    IMAGE_SUFFIXES,
    compute_image_md5,
    export_record,
    image_dimensions,
    yolo_pose_line,
)
from .geometry import (
    KEYPOINT_NAMES,
    KEYPOINT_SEMANTICS,
    bbox_from_keypoints,
    signed_area,
    validate_keypoints,
)


VALID_STATUSES = ("pending", "accepted", "rejected", "ignored", "needs_review")
EXPORT_FORMATS = {
    "yolo_pose": {"label": "YOLO Pose", "suffix": ""},
    "coco_keypoints": {"label": "COCO Keypoints", "suffix": "-coco-keypoints"},
}
EXPORT_FORMAT_ALIASES = {
    "yolo": "yolo_pose",
    "yolo_pose": "yolo_pose",
    "coco": "coco_keypoints",
    "coco_keypoints": "coco_keypoints",
}


class ConflictError(Exception):
    pass


def _configured_default_class_id(value, classes):
    class_ids = {int(entry["id"]) for entry in classes}
    if not class_ids:
        return None
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = min(class_ids)
    return candidate if candidate in class_ids else min(class_ids)


def _side_classes():
    return [{"id": 0, "name": "class_0"}, {"id": 1, "name": "class_1"}]


def _training_split(records, validation_fraction=0.1):
    image_md5s = sorted(records)
    if len(image_md5s) < 2:
        return {image_md5: "train" for image_md5 in image_md5s}
    validation_count = int(round(len(image_md5s) * float(validation_fraction)))
    validation_count = max(1, min(len(image_md5s) - 1, validation_count))
    validation = set(image_md5s[-validation_count:])
    return {
        image_md5: ("val" if image_md5 in validation else "train")
        for image_md5 in image_md5s
    }


def _archive_record_image_source(archive_path, record):
    archived_image = archive_path / str(record.get("image") or "")
    source_image = Path(str(record.get("source_image") or "")).expanduser()
    if archived_image.is_file():
        return archived_image.resolve()
    if source_image.is_file():
        return source_image.resolve()
    identifier = record.get("image_md5") or record.get("image_id") or record.get("source_image")
    raise OSError("missing source image for archive record %s" % identifier)


def _archive_record_md5(archive_path, record, image_source=None):
    image_md5 = str(record.get("image_md5") or "").strip()
    if image_md5:
        return image_md5
    return compute_image_md5(image_source or _archive_record_image_source(archive_path, record))


class ReviewStore:
    def __init__(self, workspace, workspace_class_id=None):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace_class_id = (
            int(workspace_class_id) if workspace_class_id is not None else None
        )
        if self.workspace_class_id not in (None, 0, 1):
            raise ValueError("workspace_class_id must be 0 or 1")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.database_path = self.workspace / "review.sqlite3"
        self.connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._archive_update_lock = threading.RLock()
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        self.sync_archive_index()
        self._remove_pose_confirmation_gate()
        self._ensure_active_review_batch()
        if self.meta("input_root"):
            self.sync_candidates()

    def close(self):
        # Pending/follow-up/legacy ignored rows are only the lazily loaded
        # input queue. Rejections remain as a durable content-hash exclusion.
        with self.connection:
            self.connection.execute(
                "DELETE FROM items WHERE archived=0 AND supersedes_archive_batch IS NULL "
                "AND status IN ('pending','needs_review','ignored')"
            )
        self.connection.close()

    def _create_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                image_path TEXT NOT NULL UNIQUE,
                image_md5 TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                input_index INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0,
                archived_at TEXT,
                archive_batch TEXT,
                pose_ready INTEGER NOT NULL DEFAULT 0,
                pose_error TEXT NOT NULL DEFAULT '',
                supersedes_archive_batch TEXT,
                review_batch TEXT,
                note TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY,
                item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                score REAL,
                polygon_json TEXT NOT NULL,
                source_index INTEGER,
                keypoint_visibility_json TEXT NOT NULL,
                keypoint_semantics TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS archive_items (
                id INTEGER PRIMARY KEY,
                archive_batch TEXT NOT NULL,
                archive_key TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_md5 TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                status TEXT NOT NULL,
                reviewed INTEGER NOT NULL DEFAULT 1,
                archived_at TEXT,
                review_batch TEXT,
                note TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                pose_ready INTEGER NOT NULL DEFAULT 0,
                pose_error TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1,
                UNIQUE(archive_batch, archive_key)
            );
            CREATE TABLE IF NOT EXISTS archive_annotations (
                id INTEGER PRIMARY KEY,
                archive_item_id INTEGER NOT NULL REFERENCES archive_items(id) ON DELETE CASCADE,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                score REAL,
                polygon_json TEXT NOT NULL,
                source_index INTEGER,
                keypoint_visibility_json TEXT NOT NULL,
                keypoint_semantics TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS archive_manifest_state (
                archive_batch TEXT PRIMARY KEY,
                manifest_path TEXT NOT NULL,
                modified_ns INTEGER NOT NULL,
                size INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_items_status ON items(status, id);
            CREATE INDEX IF NOT EXISTS idx_annotations_item ON annotations(item_id);
            CREATE INDEX IF NOT EXISTS idx_annotations_class ON annotations(class_id, item_id);
            CREATE TABLE IF NOT EXISTS review_batches (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                archived_at TEXT,
                archive_batch TEXT
            );
            """
        )
        self._migrate_schema()
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_items_reviewed ON items(reviewed, id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_items_archived ON items(archived, id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_items_review_batch ON items(review_batch, archived, id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_items_image_md5 ON items(image_md5)")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_archive_origin "
            "ON items(archive_origin_batch, archive_origin_key)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_items_batch ON archive_items(archive_batch, id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_items_md5 ON archive_items(image_md5)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_annotations_item "
            "ON archive_annotations(archive_item_id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_annotations_class "
            "ON archive_annotations(class_id, archive_item_id)"
        )
        self.connection.commit()

    def _migrate_schema(self):
        """Bring review databases created by earlier releases up to date."""
        columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(items)")
        }
        reviewed_column_missing = "reviewed" not in columns
        self._legacy_source_line_column = "source_line" in columns
        item_columns = (
            ("image_md5", "TEXT NOT NULL DEFAULT ''"),
            ("input_index", "INTEGER NOT NULL DEFAULT 0"),
            ("reviewed", "INTEGER NOT NULL DEFAULT 0"),
            ("archived", "INTEGER NOT NULL DEFAULT 0"),
            ("archived_at", "TEXT"),
            ("archive_batch", "TEXT"),
            ("pose_ready", "INTEGER NOT NULL DEFAULT 0"),
            ("pose_error", "TEXT NOT NULL DEFAULT ''"),
            ("supersedes_archive_batch", "TEXT"),
            ("archive_origin_batch", "TEXT"),
            ("archive_origin_key", "TEXT"),
            ("review_batch", "TEXT"),
        )
        for name, definition in item_columns:
            if name not in columns:
                self.connection.execute("ALTER TABLE items ADD COLUMN %s %s" % (name, definition))

        if "source_line" in columns:
            self.connection.execute(
                "UPDATE items SET input_index=source_line WHERE input_index=0"
            )
        self.connection.execute("UPDATE items SET input_index=id WHERE input_index=0")
        if reviewed_column_missing:
            self.connection.execute(
                "UPDATE items SET reviewed=1 WHERE reviewed=0 AND status IN ('accepted','rejected')"
            )

        # Old databases have no content digest. Prefer the real hash when the
        # source still exists; retain a unique marker for missing source files.
        missing_hashes = self.connection.execute(
            "SELECT id,image_path FROM items WHERE image_md5=''"
        ).fetchall()
        for row in missing_hashes:
            try:
                image_md5 = compute_image_md5(row["image_path"])
            except OSError:
                image_md5 = "legacy-%d" % row["id"]
            self.connection.execute(
                "UPDATE items SET image_md5=? WHERE id=?", (image_md5, row["id"])
            )

        annotation_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(annotations)")
        }
        if "keypoint_visibility_json" not in annotation_columns:
            self.connection.execute(
                "ALTER TABLE annotations ADD COLUMN keypoint_visibility_json "
                "TEXT NOT NULL DEFAULT '[2,2,2,2]'"
            )
        if "keypoint_semantics" not in annotation_columns:
            self.connection.execute(
                "ALTER TABLE annotations ADD COLUMN keypoint_semantics "
                "TEXT NOT NULL DEFAULT 'legacy_polygon_v0'"
            )

        archive_item_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(archive_items)")
        }
        if "version" not in archive_item_columns:
            self.connection.execute(
                "ALTER TABLE archive_items ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )

    def _remove_pose_confirmation_gate(self):
        if self.meta("pose_confirmation_removed") == "1":
            return
        with self.connection:
            self.connection.execute(
                "UPDATE annotations SET keypoint_semantics=?",
                (KEYPOINT_SEMANTICS,),
            )
            self.connection.execute(
                "UPDATE archive_annotations SET keypoint_semantics=?",
                (KEYPOINT_SEMANTICS,),
            )
            for table, annotation_table, foreign_key in (
                ("items", "annotations", "item_id"),
                ("archive_items", "archive_annotations", "archive_item_id"),
            ):
                rows = self.connection.execute(
                    "SELECT id,width,height,status FROM %s" % table
                ).fetchall()
                for row in rows:
                    annotations = [
                        self._annotation(value)
                        for value in self.connection.execute(
                            "SELECT * FROM %s WHERE %s=? ORDER BY id"
                            % (annotation_table, foreign_key),
                            (row["id"],),
                        )
                    ]
                    pose_ready, pose_error = self._pose_readiness(annotations, row)
                    self.connection.execute(
                        "UPDATE %s SET pose_ready=?,pose_error=? WHERE id=?" % table,
                        (
                            1 if pose_ready else 0,
                            "" if pose_ready or row["status"] != "accepted" else pose_error,
                            row["id"],
                        ),
                    )
            self.set_meta("pose_confirmation_removed", "1")

    def initialize(self, input_root, limit=None):
        input_root = Path(input_root).expanduser().resolve()
        if not input_root.is_dir():
            raise ValueError("input directory does not exist")
        now = _now()
        with self.connection:
            # Pending/follow-up/legacy ignored rows are an input-session queue.
            # Keep rejected hashes so rejected content never re-enters later.
            self.connection.execute(
                "DELETE FROM items WHERE archived=0 AND supersedes_archive_batch IS NULL "
                "AND status IN ('pending','needs_review','ignored')"
            )
            # Do not scan input images here. Large directories are consumed in
            # small batches when the pending queue needs another page.
            classes = _side_classes() if self.workspace_class_id is not None else [{"id": 0, "name": "class_0"}]
            self.set_meta("input_root", str(input_root))
            self.set_meta("classes", json.dumps(classes, separators=(",", ":")))
            default_class_id = (
                self.workspace_class_id
                if self.workspace_class_id is not None
                else _configured_default_class_id(None, classes)
            )
            self.set_meta("default_class_id", str(default_class_id))
            self.set_meta("created_at", now)
            self.set_meta("image_scan_offset", "0")
            self.set_meta("image_scan_complete", "0")
            self.set_meta("input_limit", str(limit or ""))
            self.set_meta("import_errors", "[]")
        # Establish the output layout when a workspace is first opened so a
        # newly selected output directory is immediately a usable workspace.
        self._candidates_root()
        self.archive_root().mkdir(parents=True, exist_ok=True)
        self.sync_candidates()
        return {"imported": 0, "errors": [], "existing": False}

    def _lazy_import_pending(self, required_count):
        """Fill the pending queue without walking the whole input directory."""
        if self.meta("image_scan_complete") == "1":
            return
        input_root = self.meta("input_root")
        if not input_root:
            return
        existing = self.connection.execute(
            "SELECT COUNT(*) AS count FROM items WHERE archived=0 AND status IN ('pending','needs_review')"
        ).fetchone()["count"]
        if existing >= required_count:
            return
        start_offset = int(self.meta("image_scan_offset") or 0)
        limit_text = self.meta("input_limit") or ""
        source_limit = int(limit_text) if limit_text.isdigit() else None
        errors = json.loads(self.meta("import_errors") or "[]")
        now = _now()
        last_offset = start_offset
        reached_end = True
        image_iter = enumerate(self._input_images(Path(input_root)), 1)
        with self.connection:
            for offset, image_path in image_iter:
                if offset <= start_offset:
                    continue
                last_offset = offset
                if source_limit and offset > source_limit:
                    break
                try:
                    image_md5 = compute_image_md5(image_path)
                    width, height = image_dimensions(image_path)
                except (OSError, ValueError) as error:
                    if len(errors) < 100:
                        errors.append({"image": str(image_path), "error": str(error)})
                    continue
                processed = self.connection.execute(
                    "SELECT 1 WHERE EXISTS ("
                    "SELECT 1 FROM items WHERE image_md5=? AND status IN ('accepted','rejected')"
                    ") OR EXISTS (SELECT 1 FROM archive_items WHERE image_md5=?)",
                    (image_md5, image_md5),
                ).fetchone()
                if processed:
                    continue
                if self._legacy_source_line_column:
                    cursor = self.connection.execute(
                        "INSERT OR IGNORE INTO items("
                        "image_path,image_md5,width,height,input_index,source_line,updated_at"
                        ") VALUES(?,?,?,?,?,?,?)",
                        (str(image_path), image_md5, width, height, offset, offset, now),
                    )
                else:
                    cursor = self.connection.execute(
                        "INSERT OR IGNORE INTO items("
                        "image_path,image_md5,width,height,input_index,updated_at"
                        ") VALUES(?,?,?,?,?,?)",
                        (str(image_path), image_md5, width, height, offset, now),
                    )
                if not cursor.rowcount:
                    continue
                existing += 1
                if existing >= required_count:
                    try:
                        next(image_iter)
                    except StopIteration:
                        reached_end = True
                    else:
                        reached_end = False
                    break
            self.set_meta("image_scan_offset", str(last_offset))
            self.set_meta("image_scan_complete", "1" if reached_end else "0")
            self.set_meta("import_errors", json.dumps(errors, separators=(",", ":")))

    def _input_images(self, input_root):
        for root, directories, files in os.walk(input_root):
            current = Path(root)
            directories[:] = sorted(
                name for name in directories
                if (current / name).resolve() != self.workspace
            )
            for name in sorted(files):
                path = current / name
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    yield path.resolve()

    def meta(self, key):
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key, value):
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def config(self):
        batches = self.list_review_batches()
        input_root = self.meta("input_root")
        classes = json.loads(self.meta("classes") or "[]")
        return {
            "classes": classes,
            "default_class_id": _configured_default_class_id(
                self.meta("default_class_id"), classes
            ),
            "workspace": str(self.workspace),
            "input_root": input_root,
            "output_root": str(self.workspace),
            "candidates_root": str(self.workspace / "candidates"),
            "archive_root": str(self.archive_root()),
            "statuses": list(VALID_STATUSES),
            "keypoint_names": list(KEYPOINT_NAMES),
            "keypoint_semantics": KEYPOINT_SEMANTICS,
            "active_review_batch": self.meta("active_review_batch"),
            "review_batches": batches,
            "archive_batches": self.list_archive_batches(),
            "counts": self.status_counts(),
        }

    def archive_root(self):
        return self.workspace / "archives"

    def sync_archive_index(self):
        """Mirror immutable archive manifests into SQLite for review navigation."""
        manifests = {
            "%s/%s" % (path.parent.parent.name, path.parent.name): path
            for path in self.archive_root().glob("*/*/annotations.jsonl")
        }
        if manifests:
            self._backup_legacy_archive_rows()
        indexed_batches = {
            row["archive_batch"]
            for row in self.connection.execute("SELECT archive_batch FROM archive_manifest_state")
        }
        with self.connection:
            for removed_batch in indexed_batches - set(manifests):
                self.connection.execute(
                    "DELETE FROM archive_items WHERE archive_batch=?", (removed_batch,)
                )
                self.connection.execute(
                    "DELETE FROM archive_manifest_state WHERE archive_batch=?", (removed_batch,)
                )

            for archive_batch, manifest in sorted(manifests.items()):
                stat = manifest.stat()
                state = self.connection.execute(
                    "SELECT modified_ns,size FROM archive_manifest_state WHERE archive_batch=?",
                    (archive_batch,),
                ).fetchone()
                indexed = self.connection.execute(
                    "SELECT 1 FROM archive_items WHERE archive_batch=? LIMIT 1",
                    (archive_batch,),
                ).fetchone()
                if (
                    state
                    and indexed
                    and state["modified_ns"] == stat.st_mtime_ns
                    and state["size"] == stat.st_size
                ):
                    continue
                self._replace_archive_index(archive_batch, manifest)
                self.connection.execute(
                    "INSERT INTO archive_manifest_state(archive_batch,manifest_path,modified_ns,size) "
                    "VALUES(?,?,?,?) ON CONFLICT(archive_batch) DO UPDATE SET "
                    "manifest_path=excluded.manifest_path,modified_ns=excluded.modified_ns,size=excluded.size",
                    (archive_batch, str(manifest), stat.st_mtime_ns, stat.st_size),
                )

            self._link_archive_corrections()
            if manifests:
                placeholders = ",".join("?" for _ in manifests)
                self.connection.execute(
                    "DELETE FROM items WHERE archived=1 AND archive_batch IN (%s)" % placeholders,
                    list(manifests),
                )

    def _backup_legacy_archive_rows(self):
        archive_indexed = self.connection.execute(
            "SELECT 1 FROM archive_manifest_state LIMIT 1"
        ).fetchone()
        archived_rows = self.connection.execute(
            "SELECT 1 FROM items WHERE archived=1 LIMIT 1"
        ).fetchone()
        backup_path = self.database_path.with_name(
            self.database_path.name + ".pre-archive-index.bak"
        )
        if archive_indexed or not archived_rows or backup_path.exists():
            return
        backup = sqlite3.connect(str(backup_path))
        try:
            self.connection.backup(backup)
        finally:
            backup.close()

    def _link_archive_corrections(self):
        corrections = self.connection.execute(
            "SELECT id,image_path,image_md5,supersedes_archive_batch FROM items "
            "WHERE archived=0 AND supersedes_archive_batch IS NOT NULL AND ("
            "archive_origin_batch IS NULL OR archive_origin_key IS NULL OR NOT EXISTS ("
            "SELECT 1 FROM archive_items ai WHERE ai.archive_batch=items.archive_origin_batch "
            "AND ai.archive_key=items.archive_origin_key))"
        ).fetchall()
        for correction in corrections:
            archive_item = self.connection.execute(
                "SELECT archive_batch,archive_key FROM archive_items "
                "WHERE archive_batch=? AND (image_path=? OR image_md5=?) "
                "ORDER BY CASE WHEN image_path=? THEN 0 ELSE 1 END,id LIMIT 1",
                (
                    correction["supersedes_archive_batch"],
                    correction["image_path"],
                    correction["image_md5"],
                    correction["image_path"],
                ),
            ).fetchone()
            if archive_item:
                self.connection.execute(
                    "UPDATE items SET archive_origin_batch=?,archive_origin_key=? WHERE id=?",
                    (archive_item["archive_batch"], archive_item["archive_key"], correction["id"]),
                )

    def _replace_archive_index(self, archive_batch, manifest):
        archive_path = manifest.parent
        report = {}
        report_path = archive_path / "archive_report.json"
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = {}
        self.connection.execute("DELETE FROM archive_items WHERE archive_batch=?", (archive_batch,))
        with open(manifest, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "invalid JSON in %s:%d: %s" % (manifest, line_number, error)
                    )
                archive_key = str(
                    record.get("image_md5") or record.get("image_id") or line_number
                )
                try:
                    image_path = _archive_record_image_source(archive_path, record)
                    image_md5 = str(record.get("image_md5") or "").strip()
                    if not image_md5:
                        existing = self.connection.execute(
                            "SELECT image_md5 FROM items WHERE image_path=? AND image_md5<>'' "
                            "ORDER BY archived DESC,id DESC LIMIT 1",
                            (str(image_path),),
                        ).fetchone()
                        image_md5 = existing["image_md5"] if existing else compute_image_md5(image_path)
                except OSError:
                    image_path = Path(str(record.get("source_image") or "")).expanduser()
                    image_md5 = str(record.get("image_md5") or record.get("image_id") or archive_key)
                review = record.get("review") or {}
                status = review.get("status") or "accepted"
                detections = record.get("detections") or []
                dimensions = {
                    "width": int(record.get("width") or 0),
                    "height": int(record.get("height") or 0),
                }
                pose_ready, pose_error = self._pose_readiness(detections, dimensions)
                cursor = self.connection.execute(
                    "INSERT INTO archive_items(archive_batch,archive_key,image_path,image_md5,width,height,"
                    "status,reviewed,archived_at,review_batch,note,updated_at,pose_ready,pose_error,version) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        archive_batch,
                        archive_key,
                        str(image_path),
                        image_md5,
                        dimensions["width"],
                        dimensions["height"],
                        status,
                        1 if review.get("reviewed", True) else 0,
                        report.get("archived_at"),
                        review.get("review_batch") or record.get("review_batch"),
                        str(review.get("note") or ""),
                        review.get("updated_at") or report.get("archived_at") or _now(),
                        1 if pose_ready else 0,
                        "" if pose_ready or status != "accepted" else pose_error,
                        max(1, int(review.get("version") or 1)),
                    ),
                )
                archive_item_id = cursor.lastrowid
                for index, detection in enumerate(detections):
                    self.connection.execute(
                        "INSERT INTO archive_annotations(archive_item_id,class_id,class_name,score,"
                        "polygon_json,source_index,keypoint_visibility_json,keypoint_semantics) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            archive_item_id,
                            int(detection.get("class_id") or 0),
                            str(detection.get("class_name") or "class_%d" % int(detection.get("class_id") or 0)),
                            detection.get("score"),
                            json.dumps(detection.get("polygon") or [], separators=(",", ":")),
                            detection.get("source_index", index),
                            json.dumps(
                                detection.get("keypoint_visibility") or [2, 2, 2, 2],
                                separators=(",", ":"),
                            ),
                            KEYPOINT_SEMANTICS,
                        ),
                    )

    def list_archive_batches(self):
        batches = []
        for manifest in sorted(self.archive_root().glob("*/*/annotations.jsonl"), reverse=True):
            archive_batch = "%s/%s" % (manifest.parent.parent.name, manifest.parent.name)
            report_path = manifest.parent / "archive_report.json"
            report = {}
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    report = {}
            decisions = {
                "accepted": int(report.get("accepted_images") or report.get("archived_images") or 0),
                "rejected": int(report.get("excluded_images") or 0),
                "ignored": 0,
            }
            if not report:
                try:
                    with open(manifest, "r", encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            status = (record.get("review") or {}).get("status") or "accepted"
                            if status in decisions:
                                decisions[status] += 1
                except (OSError, ValueError):
                    decisions = {"accepted": 0, "rejected": 0, "ignored": 0}
            batches.append({
                "archive_batch": archive_batch,
                "path": str(manifest.parent),
                "archived_at": report.get("archived_at"),
                "review_batch": report.get("review_batch"),
                "review_batch_name": report.get("review_batch_name"),
                "item_count": sum(decisions.values()),
                "accepted_count": decisions["accepted"],
                "excluded_count": decisions["rejected"] + decisions["ignored"],
                "correction_revisions": int(report.get("correction_revisions") or 0),
            })
        return batches

    def export_archive_batch(self, archive_batch, destination_root, export_format="yolo_pose"):
        archive_batch = str(archive_batch or "").strip().strip("/")
        requested_format = str(export_format or "yolo_pose").strip().lower()
        export_format = EXPORT_FORMAT_ALIASES.get(requested_format)
        if not export_format:
            raise ValueError("unsupported export format: %s" % requested_format)
        destination_text = str(destination_root or "").strip()
        if not destination_text:
            raise ValueError("export directory is required")
        available = {entry["archive_batch"]: entry for entry in self.list_archive_batches()}
        source = available.get(archive_batch)
        if not source:
            raise ValueError("archive batch not found")
        destination_root = Path(destination_text).expanduser().resolve()
        if destination_root.exists() and not destination_root.is_dir():
            raise ValueError("export path must be a directory")
        destination_root.mkdir(parents=True, exist_ok=True)
        target = destination_root / (
            Path(source["path"]).name + EXPORT_FORMATS[export_format]["suffix"]
        )
        replacing = target.exists() or target.is_symlink()
        if replacing:
            if target.is_symlink() or not target.is_dir():
                raise ValueError("export destination is not a dataset directory: %s" % target)
            report_path = target / "dataset_report.json"
            try:
                existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("export destination is not a managed dataset: %s" % error)
            if existing_report.get("source_archive_batch") != archive_batch:
                raise ValueError("export destination belongs to another archive: %s" % target)
            if existing_report.get("annotation_format", "yolo_pose") != export_format:
                raise ValueError("export destination uses another annotation format: %s" % target)
        source_path = Path(source["path"])
        records = {}
        with open(source_path / "annotations.jsonl", "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "invalid JSON in %s:%d: %s"
                        % (source_path / "annotations.jsonl", line_number, error)
                    )
                image_source = _archive_record_image_source(source_path, record)
                image_md5 = _archive_record_md5(source_path, record, image_source)
                record = dict(record)
                record["image_md5"] = image_md5
                record.pop("image_id", None)
                decision = (record.get("review") or {}).get("status") or "accepted"
                if decision == "accepted":
                    records[image_md5] = (record, image_source)
                elif decision in ("rejected", "ignored", "needs_review", "pending"):
                    records.pop(image_md5, None)
                else:
                    raise ValueError("archive record has unsupported review status %s" % decision)
        if not records:
            raise ValueError("archive contains no accepted training images")

        classes = [
            line.rstrip("\r\n")
            for line in (source_path / "classes.txt").read_text(encoding="utf-8").splitlines()
        ]
        if not classes:
            raise ValueError("archive class map is empty")
        assignments = _training_split(records)
        temporary_path = Path(tempfile.mkdtemp(prefix=target.name + "-", dir=str(destination_root)))
        installed = False
        backup_root = None
        backup_path = None
        split_counts = {"train": 0, "val": 0}
        try:
            if export_format == "yolo_pose":
                _write_yolo_pose_dataset(
                    temporary_path, target, records, classes, assignments, split_counts
                )
                data_path = target / "data.yaml"
            else:
                _write_coco_keypoints_dataset(
                    temporary_path, records, classes, assignments, split_counts
                )
                data_path = target / "annotations"
            report = {
                "source_archive_batch": archive_batch,
                "exported_at": _now(),
                "annotation_format": export_format,
                "keypoint_names": list(KEYPOINT_NAMES),
                "keypoint_semantics": KEYPOINT_SEMANTICS,
                "included_images": len(records),
                "train_images": split_counts["train"],
                "validation_images": split_counts["val"],
                "validation_uses_training_images": split_counts["val"] == 0,
            }
            _atomic_write_text(
                temporary_path / "dataset_report.json",
                json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            )
            if replacing:
                backup_root = Path(tempfile.mkdtemp(prefix=".archive-export-backup-", dir=str(destination_root)))
                backup_path = backup_root / target.name
                os.replace(str(target), str(backup_path))
            os.replace(str(temporary_path), str(target))
            installed = True
            if backup_root:
                shutil.rmtree(str(backup_root), ignore_errors=True)
                backup_root = None
                backup_path = None
        except Exception:
            if backup_path and backup_path.exists() and not target.exists():
                os.replace(str(backup_path), str(target))
            raise
        finally:
            if not installed:
                shutil.rmtree(str(temporary_path), ignore_errors=True)
            if backup_root:
                shutil.rmtree(str(backup_root), ignore_errors=True)
        return {
            "archive_batch": archive_batch,
            "path": str(target),
            "data": str(data_path),
            "replaced": replacing,
            "export_format": export_format,
            "format_label": EXPORT_FORMATS[export_format]["label"],
            "images": len(records),
            "train_images": split_counts["train"],
            "validation_images": split_counts["val"],
        }

    def merge_archives(self, archive_batches, folder_name=None, destination_archive_batch=None):
        requested = []
        for value in archive_batches or []:
            archive_batch = str(value or "").strip().strip("/")
            if archive_batch and archive_batch not in requested:
                requested.append(archive_batch)
        archive_entries = self.list_archive_batches()
        available = {entry["archive_batch"]: entry for entry in archive_entries}
        missing = [value for value in requested if value not in available]
        if missing:
            raise ValueError("unknown archive batches: %s" % ", ".join(missing))

        destination_archive_batch = str(destination_archive_batch or "").strip().strip("/")
        if destination_archive_batch:
            if destination_archive_batch not in available:
                raise ValueError("destination archive not found")
            if destination_archive_batch not in requested:
                requested.append(destination_archive_batch)
            destination = available[destination_archive_batch]
            archive_batch = destination_archive_batch
            final_path = Path(destination["path"])
            folder_name = destination.get("review_batch_name") or final_path.name
        else:
            folder_name = str(folder_name or "").strip()
            if not folder_name or folder_name in (".", "..") or "/" in folder_name or "\\" in folder_name:
                raise ValueError("folder name must be a simple non-empty name")
            local_now = datetime.now()
            date_name = local_now.strftime("%Y-%m-%d")
            final_path = self.archive_root() / date_name / folder_name
            archive_batch = "%s/%s" % (date_name, folder_name)
            if final_path.exists():
                raise ValueError("archive folder already exists: %s" % final_path)

        if len(requested) < 2:
            raise ValueError("select at least two archive batches to merge")

        sources = [(archive_batch, Path(available[archive_batch]["path"])) for archive_batch in requested]
        sources.sort(key=lambda entry: self._archive_sort_key(entry[0], entry[1]))
        classes_content = None
        latest_records = {}
        correction_revisions = 0

        for source_batch, source_path in sources:
            current_classes = (source_path / "classes.txt").read_text(encoding="utf-8")
            if classes_content is None:
                classes_content = current_classes
            elif current_classes != classes_content:
                raise ValueError("class map changed across archived batches")
            report_path = source_path / "archive_report.json"
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    correction_revisions += int(report.get("correction_revisions") or 0)
                except (OSError, ValueError):
                    pass
            with open(source_path / "annotations.jsonl", "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            "invalid JSON in %s:%d: %s" % (source_path / "annotations.jsonl", line_number, error)
                        )
                    image_md5 = _archive_record_md5(source_path, record)
                    record = dict(record)
                    record["image_md5"] = image_md5
                    record.pop("image_id", None)
                    latest_records[image_md5] = (record, source_path, source_batch)

        date_root = final_path.parent
        date_root.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(tempfile.mkdtemp(prefix=final_path.name + "-", dir=str(date_root)))
        backup_root = None
        moved_sources = []
        archived_at = _now()
        installed_final = False

        try:
            images_path = temporary_path / "images"
            labels_path = temporary_path / "labels"
            records_path = temporary_path / "records"
            images_path.mkdir()
            labels_path.mkdir()
            records_path.mkdir()
            accepted = 0
            excluded = 0

            with open(temporary_path / "annotations.jsonl", "w", encoding="utf-8") as manifest:
                for image_md5, (record, source_path, source_batch) in sorted(latest_records.items()):
                    record = dict(record)
                    image_source = _archive_record_image_source(source_path, record)
                    record["source_image"] = str(image_source)
                    suffix = image_source.suffix.lower() or ".jpg"
                    image_name = image_md5 + suffix
                    (images_path / image_name).symlink_to(image_source)

                    decision = (record.get("review") or {}).get("status") or "accepted"
                    if decision == "accepted":
                        accepted += 1
                        source_label = source_path / "labels" / (image_md5 + ".txt")
                        if source_label.is_file():
                            label_content = source_label.read_text(encoding="utf-8")
                        else:
                            label_content = "\n".join(
                                yolo_pose_line(
                                    {
                                        "class_id": detection["class_id"],
                                        "polygon": detection["polygon"],
                                        "keypoint_visibility": detection["keypoint_visibility"],
                                        "keypoint_semantics": detection["keypoint_semantics"],
                                    },
                                    int(record["width"]),
                                    int(record["height"]),
                                )
                                for detection in record.get("detections", [])
                            )
                            if label_content:
                                label_content += "\n"
                        _atomic_write_text(labels_path / (image_md5 + ".txt"), label_content)
                    elif decision in ("rejected", "ignored", "needs_review", "pending"):
                        excluded += 1
                    else:
                        raise ValueError("archive record has unsupported review status %s" % decision)

                    record["image"] = "images/" + image_name
                    record["archive_batch"] = archive_batch
                    record["merged_from_archive_batch"] = source_batch
                    serialized = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
                    manifest.write(serialized + "\n")
                    _atomic_write_text(
                        records_path / (image_md5 + ".json"),
                        json.dumps(record, ensure_ascii=True, indent=2) + "\n",
                    )

            _atomic_write_text(temporary_path / "classes.txt", classes_content or "")
            report = {
                "archive_batch": archive_batch,
                "review_batch_name": folder_name,
                "archived_at": archived_at,
                "input_root": self.meta("input_root"),
                "merged_from": [entry[0] for entry in sources],
                "archived_images": len(latest_records),
                "accepted_images": accepted,
                "excluded_images": excluded,
                "missing_images": [],
                "annotation_format": "yolo_pose",
                "keypoint_names": list(KEYPOINT_NAMES),
                "keypoint_semantics": KEYPOINT_SEMANTICS,
                "correction_revisions": correction_revisions,
            }
            _atomic_write_text(
                temporary_path / "archive_report.json",
                json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            )
            backup_root = Path(tempfile.mkdtemp(prefix=".archive-merge-", dir=str(self.workspace)))
            for source_batch, source_path in sources:
                backup_path = backup_root / source_batch
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(source_path), str(backup_path))
                moved_sources.append((source_path, backup_path))
            os.replace(str(temporary_path), str(final_path))
            installed_final = True

            placeholders = ",".join("?" for _ in requested)
            with self.connection:
                self.connection.execute(
                    "UPDATE items SET supersedes_archive_batch=? WHERE archived=0 "
                    "AND supersedes_archive_batch IN (%s)" % placeholders,
                    [archive_batch] + requested,
                )
                self.connection.execute(
                    "UPDATE items SET archive_origin_batch=? WHERE archived=0 "
                    "AND archive_origin_batch IN (%s)" % placeholders,
                    [archive_batch] + requested,
                )
                self.connection.execute(
                    "UPDATE items SET archive_batch=?,archived_at=?,supersedes_archive_batch=NULL,updated_at=? "
                    "WHERE archived=1 AND archive_batch IN (%s)" % placeholders,
                    [archive_batch, archived_at, archived_at] + requested,
                )
                self.connection.execute(
                    "UPDATE review_batches SET archive_batch=?,archived_at=? WHERE archive_batch IN (%s)" % placeholders,
                    [archive_batch, archived_at] + requested,
                )
        except Exception:
            if installed_final and final_path.exists():
                shutil.rmtree(str(final_path), ignore_errors=True)
            for source_path, backup_path in reversed(moved_sources):
                if backup_path.exists() and not source_path.exists():
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(str(backup_path), str(source_path))
            shutil.rmtree(str(temporary_path), ignore_errors=True)
            if backup_root:
                shutil.rmtree(str(backup_root), ignore_errors=True)
            raise

        if backup_root:
            shutil.rmtree(str(backup_root), ignore_errors=True)
        for _, source_path in sources:
            try:
                source_path.parent.rmdir()
            except OSError:
                pass
        self.sync_archive_index()
        return {
            "path": str(final_path),
            "archive_batch": archive_batch,
            "source_batches": [entry[0] for entry in sources],
            "items": len(latest_records),
            "accepted": accepted,
            "excluded": excluded,
        }

    def archive_all_candidates(self, review_batch=None, merge_into_archive=None):
        merge_into_archive = str(merge_into_archive or "").strip().strip("/")
        if merge_into_archive:
            available = {entry["archive_batch"]: entry for entry in self.list_archive_batches()}
            destination = available.get(merge_into_archive)
            if not destination:
                raise ValueError("archive merge destination not found")
            current_classes = "\n".join(value["name"] for value in self.config()["classes"]) + "\n"
            destination_classes = (Path(destination["path"]) / "classes.txt").read_text(encoding="utf-8")
            if destination_classes != current_classes:
                raise ValueError("class map changed since the selected archive")

        archived = self.archive_saved(
            review_batch,
            all_candidates=True,
        )
        if not merge_into_archive:
            return archived

        merged = self.merge_archives(
            [merge_into_archive, archived["archive_batch"]],
            destination_archive_batch=merge_into_archive,
        )
        merged.update({
            "archived": archived["archived"],
            "review_batch": archived["review_batch"],
            "next_review_batch": archived["next_review_batch"],
            "merged_into": merge_into_archive,
            "missing": archived["missing"],
        })
        return merged

    @staticmethod
    def _archive_sort_key(archive_batch, source_path):
        report_path = source_path / "archive_report.json"
        if report_path.is_file():
            try:
                archived_at = json.loads(report_path.read_text(encoding="utf-8")).get("archived_at")
                if archived_at:
                    return archived_at, archive_batch
            except (OSError, ValueError):
                pass
        return archive_batch, archive_batch

    def list_review_batches(self):
        rows = self.connection.execute(
            """
            SELECT b.*,
                   COUNT(i.id) + COALESCE(ai.item_count, 0) AS item_count,
                   SUM(CASE WHEN i.archived=0 AND i.reviewed=1 AND
                       ((i.status='accepted' AND i.pose_ready=1) OR
                        (i.supersedes_archive_batch IS NOT NULL AND i.status IN ('rejected','ignored')))
                       THEN 1 ELSE 0 END) AS ready_count,
                   SUM(CASE WHEN i.archived=0 AND i.status='accepted' AND i.reviewed=1 AND i.pose_ready=0 THEN 1 ELSE 0 END) AS blocked_count
            FROM review_batches b
            LEFT JOIN items i ON i.review_batch=b.id
            LEFT JOIN (
                SELECT review_batch, COUNT(*) AS item_count
                FROM archive_items WHERE review_batch IS NOT NULL
                GROUP BY review_batch
            ) ai ON ai.review_batch=b.id
            GROUP BY b.id ORDER BY b.created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def create_review_batch(self, name=None):
        now = datetime.now()
        batch_id = "review-" + now.strftime("%Y%m%d-%H%M%S-%f")
        display_name = str(name or ("Review " + now.strftime("%Y-%m-%d %H:%M"))).strip()[:120]
        if not display_name:
            raise ValueError("review batch name is required")
        with self.connection:
            self.connection.execute(
                "INSERT INTO review_batches(id,name,status,created_at) VALUES(?,?,?,?)",
                (batch_id, display_name, "open", _now()),
            )
            self.set_meta("active_review_batch", batch_id)
        return self.review_batch(batch_id)

    def activate_review_batch(self, batch_id):
        batch = self.review_batch(batch_id)
        if not batch:
            raise ValueError("review batch not found")
        if batch["status"] != "open":
            raise ValueError("archived review batch cannot be activated")
        with self.connection:
            self.set_meta("active_review_batch", batch_id)
        return batch

    def review_batch(self, batch_id):
        for batch in self.list_review_batches():
            if batch["id"] == batch_id:
                return batch
        return None

    def _ensure_active_review_batch(self):
        active = self.meta("active_review_batch")
        row = self.connection.execute(
            "SELECT id FROM review_batches WHERE id=? AND status='open'", (active,)
        ).fetchone() if active else None
        if not row:
            self.create_review_batch()

    def status_counts(self):
        counts = {status: 0 for status in VALID_STATUSES}
        for row in self.connection.execute("SELECT status, COUNT(*) AS count FROM items GROUP BY status"):
            counts[row["status"]] = row["count"]
        for row in self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM archive_items GROUP BY status"
        ):
            counts[row["status"]] = counts.get(row["status"], 0) + row["count"]
        counts["total"] = sum(counts.values())
        counts["reviewed"] = self.connection.execute(
            "SELECT (SELECT COUNT(*) FROM items WHERE reviewed=1) + "
            "(SELECT COUNT(*) FROM archive_items WHERE reviewed=1) AS count"
        ).fetchone()["count"]
        counts["archived"] = self.connection.execute(
            "SELECT COUNT(*) AS count FROM archive_items"
        ).fetchone()["count"]
        counts["candidates"] = self.connection.execute(
            "SELECT COUNT(*) AS count FROM items WHERE status='accepted' AND archived=0"
        ).fetchone()["count"]
        counts["ready_to_archive"] = self.connection.execute(
            "SELECT COUNT(*) AS count FROM items WHERE status='accepted' AND reviewed=1 "
            "AND archived=0 AND pose_ready=1"
        ).fetchone()["count"]
        counts["unreviewed"] = counts["total"] - counts["reviewed"]
        return counts

    def list_items(self, filters):
        if (filters.get("status") or "pending") == "archive":
            return self._list_archive_items(filters)
        if (filters.get("status") or "pending") == "pending":
            page = max(1, int(filters.get("page") or 1))
            page_size = max(1, min(200, int(filters.get("page_size") or 50)))
            self._lazy_import_pending(page * page_size)
        clauses = []
        values = []
        status = filters.get("status")
        if status == "pending":
            clauses.extend(("i.status IN ('pending','needs_review')", "i.archived = 0"))
        elif status == "candidate":
            clauses.extend(("i.status = 'accepted'", "i.archived = 0"))
        elif status == "archive":
            clauses.append("i.archived = 1")
        elif status in ("rejected", "ignored"):
            clauses.extend(("i.status = ?", "i.archived = 0"))
            values.append(status)
        elif status and status != "all":
            clauses.append("i.status = ?")
            values.append(status)
        reviewed = filters.get("reviewed")
        if reviewed in ("0", "1"):
            clauses.append("i.reviewed = ?")
            values.append(int(reviewed))
        archived = filters.get("archived")
        if archived in ("0", "1"):
            clauses.append("i.archived = ?")
            values.append(int(archived))
        archive_batch = (filters.get("archive_batch") or "").strip()
        if archive_batch and archive_batch != "all":
            clauses.append("i.archive_batch = ?")
            values.append(archive_batch)
        class_id = filters.get("class_id")
        if class_id not in (None, "", "all"):
            clauses.append("EXISTS (SELECT 1 FROM annotations ca WHERE ca.item_id=i.id AND ca.class_id=?)")
            values.append(int(class_id))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        page = max(1, int(filters.get("page") or 1))
        page_size = max(1, min(200, int(filters.get("page_size") or 50)))
        total = self.connection.execute("SELECT COUNT(*) AS count FROM items i" + where, values).fetchone()["count"]
        rows = self.connection.execute(
            """
            SELECT i.*, COUNT(a.id) AS annotation_count, MIN(a.score) AS min_score,
                   GROUP_CONCAT(DISTINCT a.class_id) AS class_ids
            FROM items i LEFT JOIN annotations a ON a.item_id=i.id
            %s GROUP BY i.id ORDER BY i.id LIMIT ? OFFSET ?
            """ % where,
            values + [page_size, (page - 1) * page_size],
        ).fetchall()
        scan_complete = True
        if status == "pending":
            scan_complete = self.meta("image_scan_complete") == "1"
            # Keep pagination open while more input images remain. The stored
            # total is only the imported pending queue, so a full page would
            # otherwise look like the last page and never request the next.
            if not scan_complete and total >= page * page_size:
                total = max(total, page * page_size + 1)
        return {
            "items": [self._summary(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "scan_complete": scan_complete,
            "counts": self.status_counts(),
        }

    def _list_archive_items(self, filters):
        clauses = []
        values = []
        archive_batch = (filters.get("archive_batch") or "").strip()
        if archive_batch and archive_batch != "all":
            clauses.append("ai.archive_batch = ?")
            values.append(archive_batch)
        class_id = filters.get("class_id")
        if class_id not in (None, "", "all"):
            clauses.append(
                "EXISTS (SELECT 1 FROM archive_annotations aa "
                "WHERE aa.archive_item_id=ai.id AND aa.class_id=?)"
            )
            values.append(int(class_id))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        page = max(1, int(filters.get("page") or 1))
        page_size = max(1, min(200, int(filters.get("page_size") or 50)))
        total = self.connection.execute(
            "SELECT COUNT(*) AS count FROM archive_items ai" + where, values
        ).fetchone()["count"]
        rows = self.connection.execute(
            """
            SELECT ai.*, COUNT(aa.id) AS annotation_count, MIN(aa.score) AS min_score,
                   GROUP_CONCAT(DISTINCT aa.class_id) AS class_ids,
                   (
                       SELECT correction.id FROM items correction
                       WHERE correction.archived=0
                         AND correction.archive_origin_batch=ai.archive_batch
                         AND correction.archive_origin_key=ai.archive_key
                       ORDER BY correction.id DESC LIMIT 1
                   ) AS correction_id
            FROM archive_items ai
            LEFT JOIN archive_annotations aa ON aa.archive_item_id=ai.id
            %s GROUP BY ai.id ORDER BY ai.id LIMIT ? OFFSET ?
            """ % where,
            values + [page_size, (page - 1) * page_size],
        ).fetchall()
        return {
            "items": [self._archive_summary(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "scan_complete": True,
            "counts": self.status_counts(),
        }

    def get_item(self, item_id):
        if int(item_id) < 0:
            archive_item_id = -int(item_id)
            row = self.connection.execute(
                "SELECT * FROM archive_items WHERE id=?", (archive_item_id,)
            ).fetchone()
            if not row:
                return None
            annotations = self.connection.execute(
                "SELECT * FROM archive_annotations WHERE archive_item_id=? ORDER BY id",
                (archive_item_id,),
            ).fetchall()
            result = self._archive_summary(row)
            result["annotations"] = [self._archive_annotation(value) for value in annotations]
            return result
        row = self.connection.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if not row:
            return None
        annotations = self.connection.execute(
            "SELECT * FROM annotations WHERE item_id=? ORDER BY id", (item_id,)
        ).fetchall()
        result = dict(row)
        result["annotations"] = [self._annotation(annotation) for annotation in annotations]
        result["file_name"] = os.path.basename(result["image_path"])
        return result

    def update_item(self, item_id, payload):
        item_id = int(item_id)
        if item_id < 0:
            return self._update_archive_item(item_id, payload)
        status = payload.get("status")
        if status not in VALID_STATUSES:
            raise ValueError("invalid review status")
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("annotations must be a list")
        item = self.get_item(item_id)
        if not item:
            return None
        expected_version = int(payload.get("version") or 0)
        if expected_version != item["version"]:
            raise ConflictError("item changed in another session")

        normalized = [self._normalize_annotation(value, item) for value in annotations]
        if status == "accepted" and not normalized:
            raise ValueError("at least one target is required to add an item to candidates")
        reviewed = 1 if payload.get("reviewed") else 0
        pose_ready, pose_error = self._pose_readiness(normalized, item)
        was_archived = bool(item.get("archived"))
        if was_archived and not payload.get("review_approved"):
            reviewed = 0
        archived = 0 if was_archived else int(item.get("archived") or 0)
        now = _now()
        archived_at = None if was_archived else item.get("archived_at")
        archive_batch = None if was_archived else item.get("archive_batch")
        supersedes_archive_batch = (
            item.get("archive_batch") if was_archived else item.get("supersedes_archive_batch")
        )
        is_correction = was_archived or bool(item.get("supersedes_archive_batch"))
        review_batch = (
            self.meta("active_review_batch")
            if reviewed and (status == "accepted" or is_correction)
            else (item.get("review_batch") if not was_archived else None)
        )
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE items SET status=?,reviewed=?,pose_ready=?,pose_error=?,archived=?,archived_at=?,"
                "archive_batch=?,supersedes_archive_batch=?,review_batch=?,note=?,version=version+1,updated_at=? "
                "WHERE id=? AND version=?",
                (
                    status,
                    reviewed,
                    1 if pose_ready else 0,
                    pose_error,
                    archived,
                    archived_at,
                    archive_batch,
                    supersedes_archive_batch,
                    review_batch,
                    str(payload.get("note") or "")[:2000],
                    now,
                    item_id,
                    expected_version,
                ),
            )
            if not cursor.rowcount:
                raise ConflictError("item changed in another session")
            self.connection.execute("DELETE FROM annotations WHERE item_id=?", (item_id,))
            for index, annotation in enumerate(normalized):
                self.connection.execute(
                    "INSERT INTO annotations(item_id,class_id,class_name,score,polygon_json,source_index,keypoint_visibility_json,keypoint_semantics) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        item_id,
                        annotation["class_id"],
                        annotation["class_name"],
                        annotation["score"],
                        json.dumps(annotation["polygon"], separators=(",", ":")),
                        annotation.get("source_index", index),
                        json.dumps(annotation["keypoint_visibility"], separators=(",", ":")),
                        annotation["keypoint_semantics"],
                    ),
                )
        updated = self.get_item(item_id)
        self._sync_candidate_item(updated)
        if not was_archived and status == "rejected":
            # The queue can already contain duplicate files with the same
            # content. Remove those transient copies immediately as well.
            with self.connection:
                self.connection.execute(
                    "DELETE FROM items WHERE id<>? AND archived=0 AND image_md5=? "
                    "AND supersedes_archive_batch IS NULL "
                    "AND status IN ('pending','needs_review','ignored')",
                    (item_id, updated["image_md5"]),
                )
        # Pending and follow-up rows are the mutable index for the current
        # input directory. Keep them addressable for the rest of this input
        # session; initialize() replaces them when the input root changes and
        # close() discards them when the session ends.
        return updated

    def _update_archive_item(self, item_id, payload):
        with self._archive_update_lock:
            return self._update_archive_item_locked(item_id, payload)

    def _update_archive_item_locked(self, item_id, payload):
        snapshot = self.get_item(item_id)
        if not snapshot:
            return None
        if int(payload.get("version") or 0) != snapshot["version"]:
            raise ConflictError("archive item changed in another session")
        status = payload.get("status")
        if status not in VALID_STATUSES:
            raise ValueError("invalid review status")
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("annotations must be a list")
        normalized = [self._normalize_annotation(value, snapshot) for value in annotations]
        if status == "accepted" and not normalized:
            raise ValueError("at least one target is required for an accepted archive item")
        pose_ready, pose_error = self._pose_readiness(normalized, snapshot)
        if status == "accepted" and not pose_ready:
            raise ValueError("accepted archive item is not pose-ready: %s" % pose_error)

        archive_batch = snapshot["archive_batch"]
        archive_key = snapshot["archive_key"]
        archive_path = self.archive_root() / archive_batch
        manifest_path = archive_path / "annotations.jsonl"
        if not manifest_path.is_file():
            raise ValueError("archive manifest not found")

        records = []
        matched = False
        updated_at = _now()
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "invalid JSON in %s:%d: %s" % (manifest_path, line_number, error)
                    )
                record_key = str(record.get("image_md5") or record.get("image_id") or line_number)
                if record_key == archive_key:
                    review = dict(record.get("review") or {})
                    review.update({
                        "status": status,
                        "reviewed": bool(payload.get("reviewed")),
                        "note": str(payload.get("note") or "")[:2000],
                        "updated_at": updated_at,
                        "version": snapshot["version"] + 1,
                    })
                    record["review"] = review
                    record["detections"] = [
                        {
                            "polygon": annotation["polygon"],
                            "score": annotation["score"],
                            "class_id": annotation["class_id"],
                            "class_name": annotation["class_name"],
                            "keypoint_names": list(KEYPOINT_NAMES),
                            "keypoint_visibility": annotation["keypoint_visibility"],
                            "keypoint_semantics": annotation["keypoint_semantics"],
                        }
                        for annotation in normalized
                    ]
                    matched = True
                records.append(record)
        if not matched:
            raise ValueError("archive record not found")

        serialized = "".join(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
            for record in records
        )
        updated_record = next(
            record for index, record in enumerate(records, 1)
            if str(record.get("image_md5") or record.get("image_id") or index) == archive_key
        )
        image_md5 = _archive_record_md5(archive_path, updated_record)
        record_path = archive_path / "records" / (image_md5 + ".json")
        label_path = archive_path / "labels" / (image_md5 + ".txt")
        report_path = archive_path / "archive_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        accepted = sum(
            1 for record in records
            if (record.get("review") or {}).get("status", "accepted") == "accepted"
        )
        report.update({
            "archive_batch": archive_batch,
            "archived_images": len(records),
            "accepted_images": accepted,
            "excluded_images": len(records) - accepted,
            "correction_revisions": int(report.get("correction_revisions") or 0) + 1,
            "updated_at": _now(),
        })
        updates = {
            manifest_path: serialized,
            record_path: json.dumps(updated_record, ensure_ascii=True, indent=2) + "\n",
            report_path: json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        }
        if status == "accepted":
            label_lines = [
                yolo_pose_line(annotation, snapshot["width"], snapshot["height"])
                for annotation in normalized
            ]
            updates[label_path] = "\n".join(label_lines) + "\n"
        elif label_path.exists() or label_path.is_symlink():
            updates[label_path] = None

        backup_root = Path(tempfile.mkdtemp(prefix=".archive-edit-backup-", dir=str(archive_path.parent)))
        backups = {}
        try:
            for index, path in enumerate(updates):
                backup = backup_root / str(index)
                exists = path.exists() or path.is_symlink()
                backups[path] = (backup, exists)
                if path.is_symlink():
                    backup.symlink_to(os.readlink(path))
                elif exists:
                    os.link(str(path), str(backup))

            for path, content in updates.items():
                if content is None:
                    path.unlink()
                else:
                    _atomic_write_text(path, content)
            manifest_stat = manifest_path.stat()
            with self.connection:
                cursor = self.connection.execute(
                    "UPDATE archive_items SET status=?,reviewed=?,note=?,updated_at=?,"
                    "pose_ready=?,pose_error=?,version=? WHERE id=?",
                    (
                        status,
                        1 if payload.get("reviewed") else 0,
                        str(payload.get("note") or "")[:2000],
                        updated_at,
                        1 if pose_ready else 0,
                        "" if pose_ready or status != "accepted" else pose_error,
                        snapshot["version"] + 1,
                        -int(item_id),
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("archive item not found")
                self.connection.execute(
                    "DELETE FROM archive_annotations WHERE archive_item_id=?",
                    (-int(item_id),),
                )
                for index, annotation in enumerate(normalized):
                    self.connection.execute(
                        "INSERT INTO archive_annotations(archive_item_id,class_id,class_name,score,"
                        "polygon_json,source_index,keypoint_visibility_json,keypoint_semantics) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            -int(item_id),
                            annotation["class_id"],
                            annotation["class_name"],
                            annotation["score"],
                            json.dumps(annotation["polygon"], separators=(",", ":")),
                            annotation.get("source_index", index),
                            json.dumps(
                                annotation["keypoint_visibility"], separators=(",", ":")
                            ),
                            annotation["keypoint_semantics"],
                        ),
                    )
                self.connection.execute(
                    "INSERT INTO archive_manifest_state(archive_batch,manifest_path,modified_ns,size) "
                    "VALUES(?,?,?,?) ON CONFLICT(archive_batch) DO UPDATE SET "
                    "manifest_path=excluded.manifest_path,modified_ns=excluded.modified_ns,"
                    "size=excluded.size",
                    (
                        archive_batch,
                        str(manifest_path),
                        manifest_stat.st_mtime_ns,
                        manifest_stat.st_size,
                    ),
                )
        except Exception:
            for path, (backup, existed) in backups.items():
                if path.exists() or path.is_symlink():
                    path.unlink()
                if existed:
                    os.replace(str(backup), str(path))
            raise
        finally:
            shutil.rmtree(str(backup_root), ignore_errors=True)

        with self.connection:
            stale_corrections = list(self.connection.execute(
                "SELECT id,image_md5 FROM items WHERE archived=0 AND archive_origin_batch=? "
                "AND archive_origin_key=?",
                (archive_batch, archive_key),
            ))
            self.connection.execute(
                "DELETE FROM items WHERE archived=0 AND archive_origin_batch=? AND archive_origin_key=?",
                (archive_batch, archive_key),
            )
        for correction in stale_corrections:
            self._remove_candidate_files(correction["image_md5"])

        row = self.connection.execute(
            "SELECT id FROM archive_items WHERE archive_batch=? AND archive_key=?",
            (archive_batch, archive_key),
        ).fetchone()
        return self.get_item(-row["id"]) if row else None

    def _prepare_archive_correction(self, item_id, payload):
        snapshot = self.get_item(item_id)
        if not snapshot:
            raise ValueError("archive item not found")
        expected_version = int(payload.get("version") or 0)
        if expected_version != snapshot["version"]:
            raise ConflictError("archive item changed in another session")
        row = self.connection.execute(
            "SELECT * FROM items WHERE archived=0 AND archive_origin_batch=? "
            "AND archive_origin_key=? ORDER BY id DESC LIMIT 1",
            (snapshot["archive_batch"], snapshot["archive_key"]),
        ).fetchone()
        if not row:
            row = self.connection.execute(
                "SELECT * FROM items WHERE archived=0 AND supersedes_archive_batch=? "
                "AND (image_path=? OR image_md5=?) ORDER BY id DESC LIMIT 1",
                (snapshot["archive_batch"], snapshot["image_path"], snapshot["image_md5"]),
            ).fetchone()
        if not row:
            row = self.connection.execute(
                "SELECT * FROM items WHERE archived=0 AND image_path=? ORDER BY id DESC LIMIT 1",
                (snapshot["image_path"],),
            ).fetchone()
        now = _now()
        with self.connection:
            if row:
                self.connection.execute(
                    "UPDATE items SET supersedes_archive_batch=?,archive_origin_batch=?,"
                    "archive_origin_key=? WHERE id=?",
                    (
                        snapshot["archive_batch"],
                        snapshot["archive_batch"],
                        snapshot["archive_key"],
                        row["id"],
                    ),
                )
                correction_id = row["id"]
            else:
                cursor = self.connection.execute(
                    "INSERT INTO items(image_path,image_md5,width,height,input_index,status,reviewed,"
                    "archived,pose_ready,pose_error,supersedes_archive_batch,archive_origin_batch,"
                    "archive_origin_key,review_batch,note,version,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot["image_path"],
                        snapshot["image_md5"],
                        snapshot["width"],
                        snapshot["height"],
                        0,
                        snapshot["status"],
                        0,
                        0,
                        snapshot["pose_ready"],
                        snapshot["pose_error"],
                        snapshot["archive_batch"],
                        snapshot["archive_batch"],
                        snapshot["archive_key"],
                        None,
                        snapshot["note"],
                        1,
                        now,
                    ),
                )
                correction_id = cursor.lastrowid
        correction = self.get_item(correction_id)
        revised_payload = dict(payload)
        revised_payload["version"] = correction["version"]
        return correction_id, revised_payload

    def sync_candidates(self):
        if not self.meta("classes"):
            return
        root = self._candidates_root()
        accepted_ids = {
            row["id"] for row in self.connection.execute(
                "SELECT id FROM items WHERE status='accepted' AND archived=0 AND pose_ready=1 "
                "AND EXISTS (SELECT 1 FROM annotations WHERE annotations.item_id=items.id)"
            )
        }
        for item_id in accepted_ids:
            self._sync_candidate_item(self.get_item(item_id))
        for record_path in (root / "records").glob("*.json"):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("review_item_id") not in accepted_ids:
                self._remove_candidate_files(record_path.stem)

    def image_path(self, item_id):
        if int(item_id) < 0:
            row = self.connection.execute(
                "SELECT image_path FROM archive_items WHERE id=?", (-int(item_id),)
            ).fetchone()
            return row["image_path"] if row else None
        row = self.connection.execute("SELECT image_path FROM items WHERE id=?", (item_id,)).fetchone()
        return row["image_path"] if row else None

    def _candidates_root(self):
        root = self.workspace / "candidates"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "records").mkdir(parents=True, exist_ok=True)
        classes = self.config()["classes"]
        _atomic_write_text(root / "classes.txt", "\n".join(value["name"] for value in classes) + "\n")
        return root

    def _sync_candidate_item(self, item):
        if not item:
            return
        image_md5 = compute_image_md5(item["image_path"])
        if (
            item["status"] != "accepted"
            or item.get("archived")
            or not item.get("pose_ready")
        ):
            self._remove_candidate_files(image_md5)
            return
        source_path = Path(item["image_path"])
        if not source_path.is_file():
            raise OSError("accepted source image is missing: %s" % source_path)
        root = self._candidates_root()
        suffix = source_path.suffix.lower() or ".jpg"
        image_name = image_md5 + suffix
        image_path = root / "images" / image_name
        if image_path.is_symlink() or image_path.exists():
            if not image_path.is_symlink() or image_path.resolve() != source_path.resolve():
                image_path.unlink()
        if not image_path.exists():
            image_path.symlink_to(source_path)
        label_lines = [
            yolo_pose_line(annotation, item["width"], item["height"])
            for annotation in item["annotations"]
        ]
        _atomic_write_text(
            root / "labels" / (image_md5 + ".txt"),
            "\n".join(label_lines) + ("\n" if label_lines else ""),
        )
        record = export_record(item, item["annotations"], image_md5, "images/" + image_name)
        record["review_item_id"] = item["id"]
        _atomic_write_text(
            root / "records" / (image_md5 + ".json"),
            json.dumps(record, ensure_ascii=True, indent=2) + "\n",
        )

    def _remove_candidate_files(self, image_md5):
        root = self.workspace / "candidates"
        for path in (root / "labels" / (image_md5 + ".txt"), root / "records" / (image_md5 + ".json")):
            if path.exists() or path.is_symlink():
                path.unlink()
        images = root / "images"
        if images.is_dir():
            for path in images.glob(image_md5 + ".*"):
                if path.exists() or path.is_symlink():
                    path.unlink()

    def archive_saved(self, review_batch=None, all_candidates=False):
        review_batch = review_batch or self.meta("active_review_batch")
        batch = self.review_batch(review_batch)
        if not batch:
            raise ValueError("review batch not found")
        if batch["status"] != "open":
            raise ValueError("review batch is already archived")
        archive_annotations = {}
        if all_candidates:
            rows = list(self.connection.execute(
                "SELECT * FROM items WHERE archived=0 AND "
                "(status='accepted' OR (review_batch=? AND reviewed=1 AND "
                "supersedes_archive_batch IS NOT NULL AND status IN ('rejected','ignored'))) ORDER BY id",
                (review_batch,),
            ))
            missing_sources = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
            if missing_sources:
                raise OSError("candidate source images are missing: %s" % ", ".join(missing_sources))
            blocked = []
            for row in rows:
                if row["status"] != "accepted":
                    continue
                annotations = [self._annotation(value) for value in self.connection.execute(
                    "SELECT * FROM annotations WHERE item_id=? ORDER BY id", (row["id"],)
                )]
                pose_ready, pose_error = self._pose_readiness(annotations, row)
                if not pose_ready:
                    blocked.append((row["id"], pose_error))
                archive_annotations[row["id"]] = annotations
            if blocked:
                raise ValueError(
                    "%d candidates are not pose-ready; item %d: %s"
                    % (len(blocked), blocked[0][0], blocked[0][1])
                )
        else:
            if batch["blocked_count"]:
                raise ValueError(
                    "review batch has %d accepted items with invalid pose labels" % batch["blocked_count"]
                )
            rows = list(self.connection.execute(
                "SELECT * FROM items WHERE reviewed=1 AND archived=0 AND review_batch=? AND "
                "((status='accepted' AND pose_ready=1) OR "
                "(supersedes_archive_batch IS NOT NULL AND status IN ('rejected','ignored'))) ORDER BY id",
                (review_batch,),
            ))
        if not rows:
            if all_candidates:
                raise ValueError("there are no candidates to archive")
            raise ValueError("there is no saved training data in this review batch")
        if not all_candidates:
            for row in rows:
                if row["status"] != "accepted":
                    continue
                annotations = [self._annotation(value) for value in self.connection.execute(
                    "SELECT * FROM annotations WHERE item_id=? ORDER BY id", (row["id"],)
                )]
                pose_ready, pose_error = self._pose_readiness(annotations, row)
                if not pose_ready:
                    raise ValueError(
                        "candidate is not pose-ready; item %d: %s" % (row["id"], pose_error)
                    )

        local_now = datetime.now()
        date_name = local_now.strftime("%Y-%m-%d")
        batch_name = "batch-" + local_now.strftime("%H%M%S-%f")
        date_root = self.archive_root() / date_name
        date_root.mkdir(parents=True, exist_ok=True)
        final_path = date_root / batch_name
        temporary_path = Path(tempfile.mkdtemp(prefix=batch_name + "-", dir=str(date_root)))
        labels_path = temporary_path / "labels"
        images_path = temporary_path / "images"
        records_path = temporary_path / "records"
        labels_path.mkdir()
        images_path.mkdir()
        records_path.mkdir()
        archived_at = _now()
        archive_batch = "%s/%s" % (date_name, batch_name)
        missing = []
        archived_ids = []
        archive_entries = {}
        database_updated = False

        try:
            for row in rows:
                item = dict(row)
                if all_candidates:
                    item["reviewed"] = 1
                    item["review_batch"] = review_batch
                source_path = Path(item["image_path"])
                if not source_path.is_file():
                    missing.append(item["image_path"])
                    continue
                annotations = archive_annotations.get(item["id"])
                if annotations is None:
                    annotations = [self._annotation(value) for value in self.connection.execute(
                        "SELECT * FROM annotations WHERE item_id=? ORDER BY id", (item["id"],)
                    )]
                image_md5 = compute_image_md5(item["image_path"])
                archived_ids.append(item["id"])
                entry = {
                    "item": item,
                    "source_path": source_path,
                    "annotations": annotations,
                }
                previous = archive_entries.get(image_md5)
                if previous is None or (
                    str(item.get("updated_at") or ""), item["id"]
                ) >= (
                    str(previous["item"].get("updated_at") or ""), previous["item"]["id"]
                ):
                    archive_entries[image_md5] = entry

            with open(temporary_path / "annotations.jsonl", "w", encoding="utf-8") as manifest:
                for image_md5, entry in sorted(archive_entries.items()):
                    item = entry["item"]
                    source_path = entry["source_path"]
                    annotations = entry["annotations"]
                    suffix = source_path.suffix.lower() or ".jpg"
                    image_name = image_md5 + suffix
                    (images_path / image_name).symlink_to(source_path)
                    if item["status"] == "accepted":
                        lines = [yolo_pose_line(annotation, item["width"], item["height"]) for annotation in annotations]
                        _atomic_write_text(
                            labels_path / (image_md5 + ".txt"),
                            "\n".join(lines) + ("\n" if lines else ""),
                        )
                    record = export_record(item, annotations, image_md5, "images/" + image_name)
                    record["review_item_id"] = item["id"]
                    record["archive_batch"] = archive_batch
                    record["review_batch"] = review_batch
                    if item.get("supersedes_archive_batch"):
                        record["supersedes_archive_batch"] = item["supersedes_archive_batch"]
                    _atomic_write_text(
                        records_path / (image_md5 + ".json"),
                        json.dumps(record, ensure_ascii=True, indent=2) + "\n",
                    )
                    manifest.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")

            if not archived_ids:
                raise OSError("all saved source images are missing")
            classes = self.config()["classes"]
            _atomic_write_text(
                temporary_path / "classes.txt",
                "\n".join(value["name"] for value in classes) + "\n",
            )
            report = {
                "archive_batch": archive_batch,
                "review_batch": review_batch,
                "review_batch_name": batch["name"],
                "archived_at": archived_at,
                "input_root": self.meta("input_root"),
                "archived_images": len(archive_entries),
                "missing_images": missing,
                "annotation_format": "yolo_pose",
                "keypoint_names": list(KEYPOINT_NAMES),
                "keypoint_semantics": KEYPOINT_SEMANTICS,
                "correction_revisions": sum(
                    1 for entry in archive_entries.values()
                    if entry["item"]["supersedes_archive_batch"]
                ),
                "accepted_images": sum(
                    1 for entry in archive_entries.values()
                    if entry["item"]["status"] == "accepted"
                ),
                "excluded_images": sum(
                    1 for entry in archive_entries.values()
                    if entry["item"]["status"] in ("rejected", "ignored")
                ),
            }
            _atomic_write_text(
                temporary_path / "archive_report.json",
                json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            )
            os.replace(str(temporary_path), str(final_path))
            placeholders = ",".join("?" for _ in archived_ids)
            with self.connection:
                if all_candidates:
                    accepted_ids = [
                        row["id"] for row in rows
                        if row["status"] == "accepted" and row["id"] in archived_ids
                    ]
                    self.connection.execute(
                        "UPDATE items SET reviewed=1,review_batch=?,archived=1,archived_at=?,archive_batch=?,"
                        "version=version+1,updated_at=? WHERE archived=0 AND id IN (%s)" % placeholders,
                        [review_batch, archived_at, archive_batch, archived_at] + archived_ids,
                    )
                    accepted_placeholders = ",".join("?" for _ in accepted_ids)
                    self.connection.execute(
                        "UPDATE items SET pose_ready=1,pose_error='' WHERE id IN (%s)"
                        % accepted_placeholders,
                        accepted_ids,
                    )
                else:
                    self.connection.execute(
                        "UPDATE items SET archived=1,archived_at=?,archive_batch=?,version=version+1,updated_at=? "
                        "WHERE archived=0 AND id IN (%s)" % placeholders,
                        [archived_at, archive_batch, archived_at] + archived_ids,
                    )
                self.connection.execute(
                    "UPDATE review_batches SET status='archived',archived_at=?,archive_batch=? WHERE id=?",
                    (archived_at, archive_batch, review_batch),
                )
            database_updated = True
            self.sync_archive_index()
            self.sync_candidates()
        except Exception:
            shutil.rmtree(str(temporary_path), ignore_errors=True)
            if not database_updated and final_path.exists():
                shutil.rmtree(str(final_path), ignore_errors=True)
            raise
        if self.meta("active_review_batch") == review_batch:
            next_batch = self.create_review_batch()
        else:
            next_batch = self.review_batch(self.meta("active_review_batch"))
        return {
            "path": str(final_path),
            "archive_batch": archive_batch,
            "review_batch": review_batch,
            "next_review_batch": next_batch,
            "archive_date": date_name,
            "archived": len(archived_ids),
            "missing": missing,
        }

    def _summary(self, row):
        value = dict(row)
        value["file_name"] = os.path.basename(value["image_path"])
        value["class_ids"] = [int(part) for part in (value.get("class_ids") or "").split(",") if part]
        return value

    def _archive_summary(self, row):
        value = dict(row)
        value["id"] = -int(value["id"])
        value["file_name"] = os.path.basename(value["image_path"])
        value["class_ids"] = [
            int(part) for part in (value.get("class_ids") or "").split(",") if part
        ]
        value["archived"] = 1
        value["supersedes_archive_batch"] = None
        value["input_index"] = 0
        value["has_correction"] = bool(value.get("correction_id"))
        return value

    def _annotation(self, row):
        value = dict(row)
        value["polygon"] = json.loads(value.pop("polygon_json"))
        value["keypoint_visibility"] = json.loads(value.pop("keypoint_visibility_json"))
        return value

    def _archive_annotation(self, row):
        value = dict(row)
        value["polygon"] = json.loads(value.pop("polygon_json"))
        value["keypoint_visibility"] = json.loads(value.pop("keypoint_visibility_json"))
        value.pop("archive_item_id", None)
        return value

    def _normalize_annotation(self, value, item):
        polygon = value.get("polygon")
        if not isinstance(polygon, list) or len(polygon) != 4:
            raise ValueError("each annotation needs four polygon points")
        points = []
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("invalid polygon point")
            x = max(0.0, min(float(item["width"]), float(point[0])))
            y = max(0.0, min(float(item["height"]), float(point[1])))
            points.append([round(x, 3), round(y, 3)])
        class_id = int(value["class_id"])
        classes = {entry["id"]: entry["name"] for entry in self.config()["classes"]}
        if class_id not in classes:
            raise ValueError("unknown class id %d" % class_id)
        score = value.get("score")
        visibility = value.get("keypoint_visibility", [2, 2, 2, 2])
        if not isinstance(visibility, list) or len(visibility) != 4:
            raise ValueError("keypoint_visibility must contain four values")
        visibility = [int(entry) for entry in visibility]
        if any(entry not in (0, 1, 2) for entry in visibility):
            raise ValueError("keypoint visibility must be 0, 1, or 2")
        return {
            "class_id": class_id,
            "class_name": classes[class_id],
            "score": float(score) if score is not None else None,
            "polygon": points,
            "keypoint_visibility": visibility,
            "keypoint_semantics": KEYPOINT_SEMANTICS,
            "source_index": value.get("source_index"),
        }

    def _pose_readiness(self, annotations, item):
        if not annotations:
            return False, "at least one target is required"
        for index, annotation in enumerate(annotations, 1):
            if any(value == 0 for value in annotation.get("keypoint_visibility", [2, 2, 2, 2])):
                return False, "target %d: all four keypoints must be labeled" % index
            try:
                validate_keypoints(annotation["polygon"], item["width"], item["height"])
            except ValueError as error:
                return False, "target %d: %s" % (index, error)
        return True, ""


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent), prefix=path.name + ".", delete=False
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _write_yolo_pose_dataset(root, target, records, classes, assignments, split_counts):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for image_md5, (record, image_source) in sorted(records.items()):
        split = assignments[image_md5]
        suffix = image_source.suffix.lower() or ".jpg"
        image_name = image_md5 + suffix
        shutil.copy2(str(image_source), str(root / "images" / split / image_name))
        label_lines = [
            yolo_pose_line(
                {
                    "class_id": detection["class_id"],
                    "polygon": detection["polygon"],
                    "keypoint_visibility": detection.get("keypoint_visibility"),
                    "keypoint_semantics": detection.get("keypoint_semantics"),
                },
                int(record["width"]),
                int(record["height"]),
            )
            for detection in record.get("detections", [])
        ]
        _atomic_write_text(
            root / "labels" / split / (image_md5 + ".txt"),
            "\n".join(label_lines) + ("\n" if label_lines else ""),
        )
        split_counts[split] += 1

    validation_path = "images/val" if split_counts["val"] else "images/train"
    yaml_lines = [
        "path: %s" % json.dumps(str(target), ensure_ascii=True),
        "train: images/train",
        "val: %s" % validation_path,
        "kpt_shape: [4, 3]",
        "flip_idx: [1, 0, 3, 2]",
        "names:",
    ]
    yaml_lines.extend(
        "  %d: %s" % (index, json.dumps(name, ensure_ascii=True))
        for index, name in enumerate(classes)
    )
    _atomic_write_text(root / "data.yaml", "\n".join(yaml_lines) + "\n")


def _write_coco_keypoints_dataset(root, records, classes, assignments, split_counts):
    categories = [
        {
            "id": index + 1,
            "name": name,
            "supercategory": "id_card",
            "keypoints": list(KEYPOINT_NAMES),
            "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]],
        }
        for index, name in enumerate(classes)
    ]
    datasets = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "val")
    }
    for split in datasets:
        (root / "images" / split).mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    annotation_id = 1
    for image_id, (image_md5, (record, image_source)) in enumerate(sorted(records.items()), 1):
        split = assignments[image_md5]
        suffix = image_source.suffix.lower() or ".jpg"
        image_name = image_md5 + suffix
        shutil.copy2(str(image_source), str(root / "images" / split / image_name))
        width = int(record["width"])
        height = int(record["height"])
        datasets[split]["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
        })
        for detection in record.get("detections", []):
            points = detection["polygon"]
            validate_keypoints(points, width, height)
            visibility = detection.get("keypoint_visibility") or [2, 2, 2, 2]
            if len(visibility) != 4 or any(int(value) not in (0, 1, 2) for value in visibility):
                raise ValueError("keypoint visibility must contain four values from 0 to 2")
            class_id = int(detection["class_id"])
            if class_id < 0 or class_id >= len(classes):
                raise ValueError("class id is outside the archive class map: %s" % class_id)
            left, top, right, bottom = bbox_from_keypoints(points)
            keypoints = []
            for point, visible in zip(points, visibility):
                keypoints.extend([float(point[0]), float(point[1]), int(visible)])
            datasets[split]["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [left, top, right - left, bottom - top],
                "area": abs(signed_area(points)),
                "iscrowd": 0,
                "segmentation": [[coordinate for point in points for coordinate in point]],
                "keypoints": keypoints,
                "num_keypoints": sum(int(value) > 0 for value in visibility),
            })
            annotation_id += 1
        split_counts[split] += 1

    for split, dataset in datasets.items():
        _atomic_write_text(
            root / "annotations" / ("keypoints_%s.json" % split),
            json.dumps(dataset, ensure_ascii=True, indent=2) + "\n",
        )
