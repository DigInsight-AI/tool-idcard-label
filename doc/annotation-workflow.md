# ID Card Annotation Workflow

## Inputs and Outputs

Start with a directory of images. Input images remain read-only. An empty output
directory is a valid workspace target. A review workspace stores a local MD5
index, mutable `candidates/` files, and correctable `archives/` batches.
It does not copy rejected or pending input images into output data. Rejected
image hashes are retained so the same content does not return to the queue.
The current input directory can be switched at any time; content MD5 excludes
images already present in candidates or archives, as well as rejected images.

## Canonical Keypoints

Every accepted card has exactly four semantic Pose keypoints:

- `K0`: physical top-left in the upright card.
- `K1`: physical top-right.
- `K2`: physical bottom-right.
- `K3`: physical bottom-left.

The editor treats this order as canonical. Geometry and visibility are validated
when labels are saved or archived; no separate confirmation step is required.

## Review and Archive

Correct or add targets, mark the image reviewed, and approve it for the active
batch. Archive only complete batches. Corrections to archived items update the
archive files and local index atomically before the save completes.

The review database maintains a synchronized index of every archive manifest.
Re-reviewing an archived record immediately updates its manifest entry, record,
YOLO label, archive report, and index. Rejected and follow-up records stop
contributing to exports immediately, and all decisions survive workspace restarts.

The ML repository consumes the closed `archives/` directory to build cumulative
YOLO Pose datasets. A single archive group can also be exported from the review
console as a directly trainable YOLO Pose or COCO Keypoints dataset. The export
contains only the final accepted records and keeps train/validation image
splits. YOLO exports text labels plus `data.yaml`; COCO exports
`annotations/keypoints_train|val.json`. Re-exporting the same archive and
format to the same location atomically replaces the previous generated dataset.
