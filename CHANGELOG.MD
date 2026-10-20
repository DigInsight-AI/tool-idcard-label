# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Fixed

- Reuse the input-directory scan between pending-queue refills so saving does
  not repeatedly enumerate, sort, and skip previously scanned image paths.
- Prefetch 100 additional pending images when the browser queue needs filling.
- Avoid full workspace configuration queries and unchanged class-map writes
  during individual saves; rejection uses the indexed image hash for cleanup.

## [1.0.0] - 2026-10-13

### Added

- Local browser-based workflow for ID-card detection and pose annotation.
- Model-assisted annotation refresh with manual correction tools.
- Review queues, immediately correctable archives, archive edit tracking, and MD5 deduplication.
- YOLO Pose and COCO Keypoints dataset export.
- Application version reporting in the UI, CLI, API, and HTTP Server header.
