import hashlib
import os
import struct

from .geometry import (
    KEYPOINT_NAMES,
    bbox_from_keypoints,
    validate_keypoints,
)


IMAGE_SUFFIXES = frozenset((".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"))


def compute_image_md5(image_path):
    """Return the content identity used throughout a review workspace.

    Input directories are intentionally interchangeable, so a path-based ID
    would admit duplicates when the same image is reached through another
    directory.  The image bytes are the durable identity instead.
    """
    digest = hashlib.md5()
    with open(os.path.expanduser(image_path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(image_path):
    """Read common raster dimensions without loading image processing dependencies."""
    try:
        with open(os.path.expanduser(image_path), "rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                return struct.unpack(">II", header[16:24])
            if header[:3] == b"GIF":
                return struct.unpack("<HH", header[6:10])
            if header[:2] == b"BM":
                width, height = struct.unpack("<ii", header[18:26])
                return abs(width), abs(height)
            if header[:2] == b"\xff\xd8":
                handle.seek(2)
                return _jpeg_dimensions(handle)
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                return _webp_dimensions(header)
    except struct.error as error:
        raise ValueError("invalid image header: %s" % error)
    raise ValueError("unsupported image format")


def _jpeg_dimensions(handle):
    while True:
        marker_prefix = handle.read(1)
        while marker_prefix == b"\xff":
            marker_prefix = handle.read(1)
        if not marker_prefix:
            break
        marker = marker_prefix[0]
        if marker in (0xD8, 0xD9):
            continue
        length_data = handle.read(2)
        if len(length_data) != 2:
            break
        segment_length = struct.unpack(">H", length_data)[0]
        if segment_length < 2:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            dimensions = handle.read(5)
            if len(dimensions) != 5:
                break
            height, width = struct.unpack(">HH", dimensions[1:5])
            return width, height
        handle.seek(segment_length - 2, 1)
    raise ValueError("invalid JPEG image")


def _webp_dimensions(header):
    chunk = header[12:16]
    if chunk == b"VP8X":
        data = header[16:26]
        return (
            1 + int.from_bytes(data[4:7], "little"),
            1 + int.from_bytes(data[7:10], "little"),
        )
    if chunk == b"VP8 ":
        data = header[16:26]
        if data[3:6] == b"\x9d\x01\x2a":
            width, height = struct.unpack("<HH", data[6:10])
            return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        data = header[16:21]
        if data[:1] == b"\x2f":
            value = int.from_bytes(data[1:5], "little")
            return (value & 0x3FFF) + 1, ((value >> 14) & 0x3FFF) + 1
    raise ValueError("unsupported WEBP image")


def yolo_pose_line(annotation, width, height):
    points = annotation["polygon"]
    validate_keypoints(points, width, height)
    left, top, right, bottom = bbox_from_keypoints(points)
    values = [
        str(int(annotation["class_id"])),
        _number(((left + right) / 2.0) / width),
        _number(((top + bottom) / 2.0) / height),
        _number((right - left) / width),
        _number((bottom - top) / height),
    ]
    visibility = _visibility(annotation.get("keypoint_visibility"))
    for index, (x, y) in enumerate(points):
        values.extend([_number(x / width), _number(y / height), str(visibility[index])])
    return " ".join(values)


def export_record(item, annotations, image_md5, exported_image_path):
    return {
        "image_md5": image_md5,
        "image": exported_image_path,
        "source_image": item["image_path"],
        "width": item["width"],
        "height": item["height"],
        "review": {
            "status": item["status"],
            "reviewed": bool(item.get("reviewed")),
            "review_batch": item.get("review_batch"),
            "note": item["note"] or "",
            "updated_at": item["updated_at"],
        },
        "detections": [
            {
                "polygon": annotation["polygon"],
                "score": annotation["score"],
                "class_id": annotation["class_id"],
                "class_name": annotation["class_name"],
                "keypoint_names": list(KEYPOINT_NAMES),
                "keypoint_visibility": _visibility(annotation.get("keypoint_visibility")),
                "keypoint_semantics": annotation["keypoint_semantics"],
            }
            for annotation in annotations
        ],
    }


def _number(value):
    return ("%.8f" % max(0.0, min(1.0, value))).rstrip("0").rstrip(".")


def _visibility(value):
    if value is None:
        return [2, 2, 2, 2]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("keypoint_visibility must contain four values")
    result = [int(item) for item in value]
    if any(item not in (0, 1, 2) for item in result):
        raise ValueError("keypoint visibility must be 0, 1, or 2")
    return result
