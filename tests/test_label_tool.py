import json
import sqlite3
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from idcard_label.cli import build_parser
from idcard_label.formats import image_dimensions
from idcard_label.geometry import KEYPOINT_SEMANTICS
from idcard_label.server import (
    ReviewServer,
    WorkspaceConfirmationRequired,
    _validate_workspace_directories,
)
from idcard_label.store import ReviewStore


class LabelToolTest(unittest.TestCase):
    @staticmethod
    def write_image(path, content=b""):
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 100, 80) + content
        )

    @staticmethod
    def annotations():
        return [{
            "class_id": 0,
            "class_name": "class_0",
            "polygon": [[10, 10], [90, 10], [90, 70], [10, 70]],
            "keypoint_visibility": [2, 2, 2, 2],
            "keypoint_semantics": KEYPOINT_SEMANTICS,
        }]

    def test_import_and_save_materialize_a_pose_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "card.jpg"
            self.write_image(image)
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                self.assertIsNone(store.get_item(1))
                item = store.list_items({"status": "pending", "page": 1, "page_size": 1})["items"][0]
                item = store.get_item(item["id"])
                updated = store.update_item(item["id"], {
                    "version": item["version"],
                    "status": "accepted",
                    "reviewed": True,
                    "annotations": self.annotations(),
                })
                self.assertTrue(updated["pose_ready"])
                labels = list((root / "workspace" / "candidates" / "labels").glob("*.txt"))
                self.assertEqual(len(labels), 1)
                self.assertEqual(len(labels[0].read_text().split()), 17)
            finally:
                store.close()

    def test_input_directory_searches_image_subdirectories_recursively(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            nested_root = input_root / "country" / "batch"
            nested_root.mkdir(parents=True)
            self.write_image(input_root / "root.jpg")
            self.write_image(nested_root / "nested.png")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(input_root)
                items = store.list_items({"status": "pending", "page": 1, "page_size": 10})
                self.assertEqual(items["total"], 2)
                self.assertEqual(
                    {item["file_name"] for item in items["items"]},
                    {"root.jpg", "nested.png"},
                )
            finally:
                store.close()

    def test_follow_up_item_remains_addressable_in_current_input_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            self.write_image(input_root / "card.jpg")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(input_root)
                item = store.get_item(
                    store.list_items({"status": "pending"})["items"][0]["id"]
                )
                updated = store.update_item(item["id"], {
                    "version": item["version"],
                    "status": "needs_review",
                    "reviewed": False,
                    "annotations": self.annotations(),
                })

                self.assertEqual(store.get_item(updated["id"])["status"], "needs_review")
                pending = store.list_items({"status": "pending"})
                self.assertEqual([entry["id"] for entry in pending["items"]], [updated["id"]])
            finally:
                store.close()

    def test_switching_input_directories_replaces_transient_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            self.write_image(first_root / "first.jpg", b"first")
            self.write_image(second_root / "second.jpg", b"second")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(first_root)
                first = store.get_item(
                    store.list_items({"status": "pending"})["items"][0]["id"]
                )
                store.update_item(first["id"], {
                    "version": first["version"],
                    "status": "needs_review",
                    "reviewed": False,
                    "annotations": self.annotations(),
                })

                store.initialize(second_root)

                self.assertIsNone(store.get_item(first["id"]))
                pending = store.list_items({"status": "pending"})
                self.assertEqual(
                    [entry["file_name"] for entry in pending["items"]],
                    ["second.jpg"],
                )
            finally:
                store.close()

    def test_opening_legacy_database_adds_lazy_input_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            workspace = root / "workspace"
            input_root.mkdir()
            workspace.mkdir()
            self.write_image(input_root / "card.jpg")

            connection = sqlite3.connect(workspace / "review.sqlite3")
            try:
                connection.executescript(
                    """
                    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE items (
                        id INTEGER PRIMARY KEY,
                        image_path TEXT NOT NULL UNIQUE,
                        width INTEGER NOT NULL,
                        height INTEGER NOT NULL,
                        source_line INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        note TEXT NOT NULL DEFAULT '',
                        version INTEGER NOT NULL DEFAULT 1,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE annotations (
                        id INTEGER PRIMARY KEY,
                        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                        class_id INTEGER NOT NULL,
                        class_name TEXT NOT NULL,
                        score REAL,
                        polygon_json TEXT NOT NULL,
                        source_index INTEGER
                    );
                    """
                )
                connection.commit()
            finally:
                connection.close()

            store = ReviewStore(workspace)
            try:
                self.assertIn(
                    "input_index",
                    {row["name"] for row in store.connection.execute("PRAGMA table_info(items)")},
                )
                store.initialize(input_root)
                items = store.list_items({"status": "pending", "page": 1, "page_size": 1})
                self.assertEqual(items["total"], 1)
                self.assertEqual(items["items"][0]["file_name"], "card.jpg")
            finally:
                store.close()

    def test_reopening_current_database_preserves_unreviewed_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "card.jpg")
            workspace = root / "workspace"
            store = ReviewStore(workspace)
            store.initialize(root)
            item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
            saved = store.update_item(item["id"], {
                "version": item["version"],
                "status": "accepted",
                "reviewed": False,
                "annotations": self.annotations(),
            })
            store.close()

            reopened = ReviewStore(workspace)
            try:
                self.assertFalse(reopened.get_item(saved["id"])["reviewed"])
            finally:
                reopened.close()

    def test_empty_annotations_cannot_be_saved_as_a_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "card.jpg")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                with self.assertRaisesRegex(ValueError, "at least one target"):
                    store.update_item(item["id"], {
                        "version": item["version"],
                        "status": "accepted",
                        "reviewed": False,
                        "annotations": [],
                    })
                self.assertEqual(store.get_item(item["id"])["status"], "pending")
                self.assertEqual(list((root / "workspace" / "candidates" / "records").glob("*.json")), [])
            finally:
                store.close()

    def test_empty_legacy_candidate_cannot_be_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "card.jpg")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                with store.connection:
                    store.connection.execute(
                        "UPDATE items SET status='accepted', reviewed=1, pose_ready=1, review_batch=? WHERE id=?",
                        (store.meta("active_review_batch"), item["id"]),
                    )
                store.sync_candidates()
                self.assertEqual(list((root / "workspace" / "candidates" / "records").glob("*.json")), [])
                with self.assertRaisesRegex(ValueError, "at least one target"):
                    store.archive_saved()
                with self.assertRaisesRegex(ValueError, "at least one target"):
                    store.archive_all_candidates(confirm_pose_semantics=True)
            finally:
                store.close()

    def test_image_dimensions_reads_jpeg_without_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "card.jpg"
            image.write_bytes(
                b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x50\x00\x64" + b"\x00" * 32
            )
            self.assertEqual(image_dimensions(image), (100, 80))

    def test_switching_input_directories_excludes_candidate_image_by_content_md5(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first_image = first_root / "card.jpg"
            second_image = second_root / "copy.jpg"
            self.write_image(first_image, b"same-image-bytes")
            self.write_image(second_image, b"same-image-bytes")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(first_root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                store.update_item(item["id"], {
                    "version": item["version"],
                    "status": "accepted",
                    "reviewed": False,
                    "annotations": self.annotations(),
                })
                self.assertEqual(len(list((root / "workspace" / "candidates" / "records").glob("*.json"))), 1)
                store.initialize(second_root)
                self.assertEqual(store.list_items({"status": "pending"})["total"], 0)
            finally:
                store.close()

    def test_rejected_image_never_returns_to_pending_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            workspace = root / "workspace"
            first_root.mkdir()
            second_root.mkdir()
            self.write_image(first_root / "card.jpg", b"rejected-image-bytes")
            self.write_image(first_root / "copy.jpg", b"rejected-image-bytes")
            self.write_image(second_root / "renamed.jpg", b"rejected-image-bytes")

            store = ReviewStore(workspace)
            store.initialize(first_root)
            pending = store.list_items({"status": "pending"})
            self.assertEqual(pending["total"], 2)
            item = store.get_item(pending["items"][0]["id"])
            rejected = store.update_item(item["id"], {
                "version": item["version"],
                "status": "rejected",
                "reviewed": True,
                "annotations": [],
            })
            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(store.list_items({"status": "pending"})["total"], 0)
            store.close()

            reopened = ReviewStore(workspace)
            try:
                reopened.initialize(first_root)
                self.assertEqual(reopened.list_items({"status": "pending"})["total"], 0)
                reopened.initialize(second_root)
                self.assertEqual(reopened.list_items({"status": "pending"})["total"], 0)
                self.assertEqual(reopened.list_items({"status": "rejected"})["total"], 1)
            finally:
                reopened.close()

    def test_ignore_action_is_not_exposed_in_the_review_ui(self):
        static_root = Path(__file__).parents[1] / "idcard_label" / "static"
        html = (static_root / "index.html").read_text(encoding="utf-8")
        javascript = (static_root / "app.js").read_text(encoding="utf-8")
        self.assertNotIn('id="ignoreButton"', html)
        self.assertNotIn('<option value="ignored">', html)
        self.assertNotIn('decisionActionAvailable("ignoreButton")', javascript)

    def test_export_archive_batch_creates_trainable_pose_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "card.jpg"
            self.write_image(image)
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                store.update_item(item["id"], {
                    "version": item["version"], "status": "accepted", "reviewed": False,
                    "annotations": self.annotations(),
                })
                archived = store.archive_all_candidates(confirm_pose_semantics=True)
                exported = store.export_archive_batch(archived["archive_batch"], root / "export")
                dataset = Path(exported["path"])
                data_yaml = (dataset / "data.yaml").read_text(encoding="utf-8")
                self.assertIn("train: images/train", data_yaml)
                self.assertIn("val: images/train", data_yaml)
                self.assertIn("kpt_shape: [4, 3]", data_yaml)
                self.assertIn("flip_idx: [1, 0, 3, 2]", data_yaml)
                images = list((dataset / "images" / "train").glob("*"))
                labels = list((dataset / "labels" / "train").glob("*.txt"))
                self.assertEqual(len(images), 1)
                self.assertEqual(len(labels), 1)
                self.assertEqual(len(labels[0].read_text().split()), 17)
                report = json.loads((dataset / "dataset_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["included_images"], 1)
                self.assertTrue(report["validation_uses_training_images"])
                self.assertFalse(exported["replaced"])
                (dataset / "stale-file.txt").write_text("old export", encoding="utf-8")
                repeated = store.export_archive_batch(archived["archive_batch"], root / "export")
                self.assertTrue(repeated["replaced"])
                self.assertFalse((dataset / "stale-file.txt").exists())
                self.assertTrue((dataset / "data.yaml").is_file())
                self.assertEqual(list((root / "workspace" / "candidates" / "records").glob("*.json")), [])
            finally:
                store.close()

    def test_export_archive_batch_supports_legacy_image_id_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "card.jpg")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                store.update_item(item["id"], {
                    "version": item["version"], "status": "accepted", "reviewed": False,
                    "annotations": self.annotations(),
                })
                archived = store.archive_all_candidates(confirm_pose_semantics=True)
                archive_path = Path(archived["path"])
                manifest_path = archive_path / "annotations.jsonl"
                record = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_md5 = record.pop("image_md5")
                record["image_id"] = "legacy-image-id"
                manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

                exported = store.export_archive_batch(archived["archive_batch"], root / "export")
                dataset = Path(exported["path"])
                exported_images = list((dataset / "images" / "train").glob("*"))
                self.assertEqual(len(exported_images), 1)
                self.assertEqual(exported_images[0].stem, expected_md5)
            finally:
                store.close()

    def test_export_archive_batch_creates_coco_keypoints_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "card.jpg"
            self.write_image(image)
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                store.update_item(item["id"], {
                    "version": item["version"], "status": "accepted", "reviewed": False,
                    "annotations": self.annotations(),
                })
                archived = store.archive_all_candidates(confirm_pose_semantics=True)
                exported = store.export_archive_batch(
                    archived["archive_batch"], root / "export", "coco_keypoints"
                )
                dataset = Path(exported["path"])
                self.assertTrue(dataset.name.endswith("-coco-keypoints"))
                self.assertEqual(exported["export_format"], "coco_keypoints")
                coco = json.loads(
                    (dataset / "annotations" / "keypoints_train.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(coco["images"]), 1)
                self.assertEqual(len(coco["annotations"]), 1)
                self.assertEqual(coco["annotations"][0]["category_id"], 1)
                self.assertEqual(coco["annotations"][0]["num_keypoints"], 4)
                self.assertEqual(len(coco["annotations"][0]["keypoints"]), 12)
                self.assertEqual(coco["categories"][0]["keypoints"], ["K0", "K1", "K2", "K3"])
                self.assertTrue((dataset / "images" / "train" / coco["images"][0]["file_name"]).is_file())
                report = json.loads((dataset / "dataset_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["annotation_format"], "coco_keypoints")
            finally:
                store.close()

    def test_export_archive_batch_rejects_unknown_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ReviewStore(Path(temporary) / "workspace")
            try:
                with self.assertRaisesRegex(ValueError, "unsupported export format"):
                    store.export_archive_batch("missing", Path(temporary) / "export", "pascal_voc")
            finally:
                store.close()

    def test_export_archive_batch_creates_distinct_validation_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "first.jpg", b"first-image")
            self.write_image(root / "second.jpg", b"second-image")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                for summary in store.list_items({"status": "pending", "page_size": 10})["items"]:
                    item = store.get_item(summary["id"])
                    store.update_item(item["id"], {
                        "version": item["version"],
                        "status": "accepted",
                        "reviewed": False,
                        "annotations": self.annotations(),
                    })
                archived = store.archive_all_candidates(confirm_pose_semantics=True)
                exported = store.export_archive_batch(archived["archive_batch"], root / "export")
                dataset = Path(exported["path"])
                train_images = list((dataset / "images" / "train").glob("*"))
                validation_images = list((dataset / "images" / "val").glob("*"))
                self.assertEqual(len(train_images), 1)
                self.assertEqual(len(validation_images), 1)
                for image_path in train_images + validation_images:
                    split = image_path.parent.name
                    self.assertTrue((dataset / "labels" / split / (image_path.stem + ".txt")).is_file())
                self.assertIn(
                    "val: images/val",
                    (dataset / "data.yaml").read_text(encoding="utf-8"),
                )
            finally:
                store.close()

    def test_archive_deduplicates_candidates_with_identical_image_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "first.jpg", b"same-image-bytes")
            self.write_image(root / "second.jpg", b"same-image-bytes")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                pending = store.list_items({"status": "pending", "page_size": 10})["items"]
                self.assertEqual(len(pending), 2)
                saved = []
                for summary in pending:
                    item = store.get_item(summary["id"])
                    saved.append(store.update_item(item["id"], {
                        "version": item["version"],
                        "status": "accepted",
                        "reviewed": True,
                        "annotations": self.annotations(),
                    }))

                archived = store.archive_saved()
                archive_path = Path(archived["path"])
                records = [
                    json.loads(line)
                    for line in (archive_path / "annotations.jsonl").read_text().splitlines()
                ]
                self.assertEqual(archived["archived"], 2)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["review_item_id"], saved[-1]["id"])
                archived_rows = store.connection.execute(
                    "SELECT COUNT(*) FROM items WHERE archived=1"
                ).fetchone()[0]
                self.assertEqual(archived_rows, 0)
                archive_items = store.list_items({
                    "status": "archive", "archive_batch": archived["archive_batch"]
                })
                self.assertEqual(archive_items["total"], 1)
                self.assertLess(archive_items["items"][0]["id"], 0)
            finally:
                store.close()

    def test_archive_review_creates_a_correction_without_removing_the_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "card.jpg")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                store.update_item(item["id"], {
                    "version": item["version"],
                    "status": "accepted",
                    "reviewed": False,
                    "annotations": self.annotations(),
                })
                archived = store.archive_all_candidates(confirm_pose_semantics=True)
                archive_batch = archived["archive_batch"]
                archive_list = store.list_items({"status": "archive", "archive_batch": archive_batch})
                self.assertEqual(archive_list["total"], 1)
                snapshot = store.get_item(archive_list["items"][0]["id"])
                original_polygon = snapshot["annotations"][0]["polygon"]
                corrected_annotations = self.annotations()
                corrected_annotations[0]["polygon"][0] = [12, 12]

                correction = store.update_item(snapshot["id"], {
                    "version": snapshot["version"],
                    "status": "accepted",
                    "reviewed": False,
                    "annotations": corrected_annotations,
                })

                self.assertGreater(correction["id"], 0)
                self.assertFalse(correction["archived"])
                self.assertEqual(correction["supersedes_archive_batch"], archive_batch)
                archive_with_correction = store.list_items({
                    "status": "archive", "archive_batch": archive_batch
                })
                self.assertEqual(archive_with_correction["total"], 1)
                self.assertTrue(archive_with_correction["items"][0]["has_correction"])
                self.assertEqual(
                    archive_with_correction["items"][0]["correction_id"], correction["id"]
                )
                self.assertEqual(store.get_item(snapshot["id"])["annotations"][0]["polygon"], original_polygon)
                self.assertEqual(store.list_items({"status": "candidate"})["total"], 1)

                store.archive_all_candidates(
                    confirm_pose_semantics=True,
                    merge_into_archive=archive_batch,
                )
                final_list = store.list_items({"status": "archive", "archive_batch": archive_batch})
                self.assertEqual(final_list["total"], 1)
                final_item = store.get_item(final_list["items"][0]["id"])
                self.assertEqual(final_item["annotations"][0]["polygon"][0], [12, 12])
                self.assertEqual(store.list_items({"status": "candidate"})["total"], 0)
                report = json.loads(
                    (Path(archived["path"]) / "archive_report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(report["accepted_images"], 1)
                self.assertEqual(report["correction_revisions"], 1)
            finally:
                store.close()

    def test_archive_review_can_reject_and_merge_the_final_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "card.jpg")
            store = ReviewStore(root / "workspace")
            try:
                store.initialize(root)
                item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
                store.update_item(item["id"], {
                    "version": item["version"],
                    "status": "accepted",
                    "reviewed": False,
                    "annotations": self.annotations(),
                })
                archived = store.archive_all_candidates(confirm_pose_semantics=True)
                archive_batch = archived["archive_batch"]
                snapshot_id = store.list_items({
                    "status": "archive", "archive_batch": archive_batch
                })["items"][0]["id"]
                snapshot = store.get_item(snapshot_id)

                rejected = store.update_item(snapshot_id, {
                    "version": snapshot["version"],
                    "status": "rejected",
                    "reviewed": True,
                    "annotations": [],
                })
                self.assertEqual(rejected["status"], "rejected")
                self.assertEqual(rejected["supersedes_archive_batch"], archive_batch)

                store.archive_all_candidates(
                    confirm_pose_semantics=True,
                    merge_into_archive=archive_batch,
                )
                final_list = store.list_items({"status": "archive", "archive_batch": archive_batch})
                self.assertEqual(final_list["total"], 1)
                self.assertEqual(store.get_item(final_list["items"][0]["id"])["status"], "rejected")
                report = json.loads(
                    (Path(archived["path"]) / "archive_report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(report["accepted_images"], 0)
                self.assertEqual(report["excluded_images"], 1)
                self.assertEqual(report["correction_revisions"], 1)
            finally:
                store.close()

    def test_follow_up_archive_correction_survives_reopening_the_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_image(root / "card.jpg")
            workspace = root / "workspace"
            store = ReviewStore(workspace)
            store.initialize(root)
            item = store.get_item(store.list_items({"status": "pending"})["items"][0]["id"])
            store.update_item(item["id"], {
                "version": item["version"],
                "status": "accepted",
                "reviewed": False,
                "annotations": self.annotations(),
            })
            archived = store.archive_all_candidates(confirm_pose_semantics=True)
            snapshot = store.get_item(store.list_items({
                "status": "archive", "archive_batch": archived["archive_batch"]
            })["items"][0]["id"])
            correction = store.update_item(snapshot["id"], {
                "version": snapshot["version"],
                "status": "needs_review",
                "reviewed": False,
                "annotations": snapshot["annotations"],
            })
            correction_id = correction["id"]
            store.close()

            reopened = ReviewStore(workspace)
            try:
                restored = reopened.get_item(correction_id)
                self.assertIsNotNone(restored)
                self.assertEqual(restored["status"], "needs_review")
                self.assertEqual(restored["supersedes_archive_batch"], archived["archive_batch"])
                archive_item = reopened.list_items({
                    "status": "archive", "archive_batch": archived["archive_batch"]
                })["items"][0]
                self.assertEqual(archive_item["correction_id"], correction_id)
            finally:
                reopened.close()

    def test_label_cli_only_exposes_the_review_server(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["serve"]).command, "serve")
        commands = next(action for action in parser._actions if getattr(action, "choices", None))
        self.assertEqual(set(commands.choices), {"serve"})

    def test_workspace_directories_accept_an_empty_input_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, output_root = _validate_workspace_directories(root, root / "workspace")
            self.assertEqual(input_root, root.resolve())
            self.assertEqual(output_root, (root / "workspace").resolve())

    def test_unconfigured_server_defaults_to_relative_data_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            preferences = Path(temporary) / "preferences.json"
            server = ReviewServer(("127.0.0.1", 0), preferences_path=preferences)
            try:
                config = server.config()
                self.assertFalse(config["configured"])
                self.assertEqual(config["input_root"], "data/idcard")
                self.assertEqual(config["output_root"], "data/workspace")
                self.assertEqual(config["model_directory"], "data/model")
            finally:
                server.server_close()

    def test_missing_item_update_returns_a_structured_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            store = ReviewStore(root / "workspace")
            store.initialize(input_root)
            server = ReviewServer(("127.0.0.1", 0), store=store)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                request = Request(
                    "http://127.0.0.1:%d/api/items/999" % server.server_address[1],
                    data=json.dumps({
                        "version": 1,
                        "status": "accepted",
                        "reviewed": False,
                        "annotations": self.annotations(),
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                )
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request)
                self.assertEqual(raised.exception.code, 404)
                self.assertEqual(
                    json.loads(raised.exception.read().decode("utf-8")),
                    {"error": "item not found"},
                )
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
                store.close()

    def test_new_workspace_requires_confirmation_and_remembers_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            image = source_root / "card.jpg"
            self.write_image(image)
            workspace = root / "workspace"
            preferences = root / "preferences.json"
            server = ReviewServer(("127.0.0.1", 0), preferences_path=preferences)
            try:
                with self.assertRaises(WorkspaceConfirmationRequired):
                    server.open_workspace(source_root, workspace)
                self.assertFalse(workspace.exists())
                config = server.open_workspace(source_root, workspace, confirm_create=True)
                self.assertTrue(config["configured"])
                self.assertEqual(config["input_root"], str(source_root.resolve()))
                self.assertTrue((workspace / "review.sqlite3").is_file())
                self.assertTrue((workspace / "candidates").is_dir())
                self.assertTrue((workspace / "archives").is_dir())
            finally:
                server.server_close()
                server.close_store()

            restored = ReviewServer(("127.0.0.1", 0), preferences_path=preferences)
            try:
                self.assertTrue(restored.config()["configured"])
                self.assertEqual(restored.config()["output_root"], str(workspace.resolve()))
            finally:
                restored.server_close()
                restored.close_store()

    def test_model_directory_and_selected_model_are_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_directory = root / "models"
            nested_directory = model_directory / "nested"
            nested_directory.mkdir(parents=True)
            (model_directory / "best.pt").write_bytes(b"first")
            (nested_directory / "newest.pth").write_bytes(b"second")
            input_root = root / "input"
            input_root.mkdir()
            workspace = root / "workspace"
            preferences = root / "preferences.json"

            server = ReviewServer(("127.0.0.1", 0), preferences_path=preferences)
            try:
                configured = server.set_model_directory(model_directory)
                self.assertEqual(configured["directory"], str(model_directory.resolve()))
                self.assertEqual(configured["selected"], "best.pt")
                server.open_workspace(input_root, workspace, confirm_create=True)
                selected = server.select_model("nested/newest.pth")
                self.assertEqual(selected["selected"], "nested/newest.pth")
            finally:
                server.server_close()
                server.close_store()

            restored = ReviewServer(("127.0.0.1", 0), preferences_path=preferences)
            try:
                model_refresh = restored.config()["model_refresh"]
                self.assertEqual(model_refresh["directory"], str(model_directory.resolve()))
                self.assertEqual(model_refresh["selected"], "nested/newest.pth")
            finally:
                restored.server_close()
                restored.close_store()

    def test_export_directory_is_persisted_after_successful_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            self.write_image(input_root / "card.jpg")
            workspace = root / "workspace"
            export_root = root / "exports"
            preferences = root / "preferences.json"

            server = ReviewServer(("127.0.0.1", 0), preferences_path=preferences)
            try:
                server.open_workspace(input_root, workspace, confirm_create=True)
                item = server.store.get_item(
                    server.store.list_items({"status": "pending"})["items"][0]["id"]
                )
                server.store.update_item(item["id"], {
                    "version": item["version"],
                    "status": "accepted",
                    "reviewed": False,
                    "annotations": self.annotations(),
                })
                archived = server.store.archive_all_candidates(confirm_pose_semantics=True)
                exported = server.export_archive_batch(
                    archived["archive_batch"], export_root, "coco_keypoints"
                )
                self.assertEqual(exported["export_root"], str(export_root.resolve()))
                self.assertEqual(server.config()["export_root"], str(export_root.resolve()))
                self.assertEqual(server.config()["export_format"], "coco_keypoints")
            finally:
                server.server_close()
                server.close_store()

            restored = ReviewServer(("127.0.0.1", 0), preferences_path=preferences)
            try:
                self.assertEqual(restored.config()["export_root"], str(export_root.resolve()))
                self.assertEqual(restored.config()["export_format"], "coco_keypoints")
            finally:
                restored.server_close()
                restored.close_store()
