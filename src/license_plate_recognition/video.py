from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2

from .detector import PlateCandidate, PlateDetector
from .ocr import PlateReader, PlateText
from .storage import PlateDetection, PlateLogger


def process_video(
    source: str | int,
    detector: PlateDetector,
    reader: PlateReader,
    output_path: Path | None = None,
    display: bool = False,
    frame_skip: int = 1,
    max_candidates: int = 2,
    async_ocr: bool = True,
    logger: PlateLogger | None = None,
    full_frame_ocr: bool = False,
    detection_ttl_frames: int = 10,
    min_display_confidence: float = 0.35,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = _build_writer(capture, output_path)
    latest_detections: list[PlateDetection] = []
    display_detections: list[PlateDetection] = []
    pending_ocr: Future[list[PlateDetection]] | None = None
    frame_number = 0

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if pending_ocr is not None and pending_ocr.done():
                    latest_detections = pending_ocr.result()
                    if logger is not None:
                        logger.log(latest_detections, source)
                    display_detections = _mark_display_frame(
                        latest_detections, frame_number
                    )
                    pending_ocr = None

                should_run_ocr = frame_number % frame_skip == 0
                if should_run_ocr and (pending_ocr is None or not async_ocr):
                    candidates = detector.detect(frame)[:max_candidates]
                    elapsed_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    if async_ocr:
                        pending_ocr = executor.submit(
                            _read_candidates,
                            reader,
                            candidates,
                            frame_number,
                            elapsed_ms,
                            frame if full_frame_ocr else None,
                        )
                    else:
                        latest_detections = _read_candidates(
                            reader,
                            candidates,
                            frame_number,
                            elapsed_ms,
                            frame if full_frame_ocr else None,
                        )
                        if logger is not None:
                            logger.log(latest_detections, source)
                        display_detections = _mark_display_frame(
                            latest_detections, frame_number
                        )

                visible_detections = _visible_detections(
                    display_detections,
                    frame_number,
                    detection_ttl_frames,
                    min_display_confidence,
                )
                annotated = draw_detections(frame, visible_detections)

                if writer is not None:
                    writer.write(annotated)
                if display:
                    cv2.imshow("License Plate Recognition", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def _mark_display_frame(
    detections: list[PlateDetection], display_frame_number: int
) -> list[PlateDetection]:
    return [
        replace(detection, frame_number=display_frame_number)
        for detection in detections
    ]


def _visible_detections(
    detections: list[PlateDetection],
    current_frame: int,
    ttl_frames: int,
    min_confidence: float,
) -> list[PlateDetection]:
    visible = []
    for detection in detections:
        if current_frame - detection.frame_number > ttl_frames:
            continue
        if detection.text is not None and detection.text.confidence < min_confidence:
            continue
        visible.append(detection)
    return visible


def _read_candidates(
    reader: PlateReader,
    candidates: list[PlateCandidate],
    frame_number: int,
    elapsed_ms: int,
    frame=None,
) -> list[PlateDetection]:
    detections = []
    for candidate in candidates:
        detections.append(
            PlateDetection(
                candidate=candidate,
                text=reader.read(candidate.crop),
                frame_number=frame_number,
                elapsed_ms=elapsed_ms,
            )
        )
    if frame is not None:
        detections.extend(_read_full_frame_regions(reader, frame, frame_number, elapsed_ms))
    return detections


def _read_full_frame_regions(
    reader: PlateReader,
    frame,
    frame_number: int,
    elapsed_ms: int,
) -> list[PlateDetection]:
    detections = []
    for region in reader.read_regions(frame):
        x, y, width, height = _pad_region(frame, region.bbox)
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            continue
        detections.append(
            PlateDetection(
                candidate=PlateCandidate((x, y, width, height), crop, 1.0),
                text=region.text,
                frame_number=frame_number,
                elapsed_ms=elapsed_ms,
            )
        )
    return detections


def _pad_region(frame, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    pad_x = int(width * 0.15)
    pad_y = int(height * 0.45)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + width + pad_x)
    y2 = min(frame.shape[0], y + height + pad_y)
    return x1, y1, x2 - x1, y2 - y1


def draw_detections(
    frame,
    detections: list[PlateDetection],
):
    annotated = frame.copy()
    for detection in detections:
        candidate = detection.candidate
        text = detection.text
        x, y, width, height = candidate.bbox
        label = text.text if text is not None else "PLATE"
        confidence = f" {text.confidence:.2f}" if text is not None else ""
        color = (0, 220, 0) if text is not None else (0, 180, 255)

        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        _draw_label(annotated, f"{label}{confidence}", x, y, color)

    return annotated


def _draw_label(frame, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y - text_height - baseline - 8)

    cv2.rectangle(
        frame,
        (x, top),
        (x + text_width + 10, top + text_height + baseline + 8),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 5, top + text_height + 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _build_writer(capture: cv2.VideoCapture, output_path: Path | None):
    if output_path is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
