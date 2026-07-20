from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from urllib.error import URLError
from urllib.request import Request, urlopen

from .detector import PlateCandidate
from .ocr import PlateText, looks_like_indian_plate


CSV_FIELDNAMES = [
    "timestamp",
    "source",
    "camera_location",
    "frame_number",
    "elapsed_ms",
    "plate_text",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "crop_path",
]


@dataclass(frozen=True)
class PlateDetection:
    candidate: PlateCandidate
    text: PlateText | None
    frame_number: int
    elapsed_ms: int


class PlateLogger:
    def __init__(
        self,
        csv_path: Path,
        crop_dir: Path | None = None,
        duplicate_seconds: float = 5.0,
        min_confidence: float = 0.35,
        db_path: Path | None = None,
        camera_location: str = "Unknown",
        server_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.crop_dir = crop_dir
        self.duplicate_seconds = duplicate_seconds
        self.min_confidence = min_confidence
        self.db_path = db_path
        self.camera_location = camera_location
        self.server_url = server_url.rstrip("/") if server_url else None
        self.api_key = api_key
        self.last_seen: dict[str, float] = {}

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.crop_dir is not None:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_header()
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_database()

    def log(self, detections: list[PlateDetection], source: str | int) -> None:
        rows = []
        for detection in detections:
            if detection.text is None:
                continue
            if not looks_like_indian_plate(detection.text.text):
                continue
            if detection.text.confidence < self.min_confidence:
                continue
            if self._is_duplicate(detection.text.text):
                continue

            crop_path = self._save_crop(detection)
            x, y, width, height = detection.candidate.bbox
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source": str(source),
                    "camera_location": self.camera_location,
                    "frame_number": detection.frame_number,
                    "elapsed_ms": detection.elapsed_ms,
                    "plate_text": detection.text.text,
                    "confidence": f"{detection.text.confidence:.4f}",
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_width": width,
                    "bbox_height": height,
                    "crop_path": str(crop_path) if crop_path is not None else "",
                }
            )

        if not rows:
            return

        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
            writer.writerows(rows)

        if self.db_path is not None:
            self._write_database_rows(rows)
        if self.server_url is not None:
            self._send_remote_rows(rows)

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames == CSV_FIELDNAMES:
                    return
                old_rows = list(reader)
                old_fieldnames = reader.fieldnames or []

            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                for row in old_rows:
                    migrated = {name: row.get(name, "") for name in old_fieldnames}
                    migrated.setdefault("camera_location", "")
                    writer.writerow(
                        {name: migrated.get(name, "") for name in CSV_FIELDNAMES}
                    )
            return

        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_FIELDNAMES)

    def _is_duplicate(self, plate_text: str) -> bool:
        now = monotonic()
        previous = self.last_seen.get(plate_text)
        if previous is not None and now - previous < self.duplicate_seconds:
            return True

        self.last_seen[plate_text] = now
        return False

    def _save_crop(self, detection: PlateDetection) -> Path | None:
        if self.crop_dir is None or detection.text is None:
            return None

        filename = f"{detection.frame_number:08d}_{detection.text.text}.jpg"
        crop_path = self.crop_dir / filename
        import cv2

        cv2.imwrite(str(crop_path), detection.candidate.crop)
        return crop_path

    def _ensure_database(self) -> None:
        assert self.db_path is not None
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS authorized_vehicles (
                    plate_text TEXT PRIMARY KEY,
                    owner_name TEXT,
                    vehicle_model TEXT,
                    note TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    camera_location TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    elapsed_ms INTEGER NOT NULL,
                    plate_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_width INTEGER NOT NULL,
                    bbox_height INTEGER NOT NULL,
                    crop_path TEXT,
                    is_authorized INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_plate ON detections(plate_text)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(timestamp)"
            )

    def _write_database_rows(self, rows: list[dict[str, object]]) -> None:
        assert self.db_path is not None
        with sqlite3.connect(self.db_path) as connection:
            authorized = {
                row[0]
                for row in connection.execute(
                    "SELECT plate_text FROM authorized_vehicles"
                ).fetchall()
            }
            connection.executemany(
                """
                INSERT INTO detections (
                    timestamp, source, camera_location, frame_number, elapsed_ms,
                    plate_text, confidence, bbox_x, bbox_y, bbox_width, bbox_height,
                    crop_path, is_authorized
                )
                VALUES (
                    :timestamp, :source, :camera_location, :frame_number, :elapsed_ms,
                    :plate_text, :confidence, :bbox_x, :bbox_y, :bbox_width, :bbox_height,
                    :crop_path, :is_authorized
                )
                """,
                [
                    {
                        **row,
                        "confidence": float(row["confidence"]),
                        "is_authorized": 1
                        if str(row["plate_text"]) in authorized
                        else 0,
                    }
                    for row in rows
                ],
            )

    def _send_remote_rows(self, rows: list[dict[str, object]]) -> None:
        assert self.server_url is not None
        for row in rows:
            payload = json.dumps(row).encode("utf-8")
            request = Request(
                f"{self.server_url}/api/detections",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key or "",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    response.read()
            except URLError as error:
                print(f"Could not send detection to server: {error}")
