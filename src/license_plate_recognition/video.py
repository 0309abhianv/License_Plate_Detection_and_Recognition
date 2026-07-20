from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = _build_writer(capture, output_path)
    latest_detections: list[PlateDetection] = []
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
                        )
                    else:
                        latest_detections = _read_candidates(
                            reader,
                            candidates,
                            frame_number,
                            elapsed_ms,
                        )
                        if logger is not None:
                            logger.log(latest_detections, source)

                annotated = draw_detections(frame, latest_detections)

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


def _read_candidates(
    reader: PlateReader,
    candidates: list[PlateCandidate],
    frame_number: int,
    elapsed_ms: int,
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
    return detections


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
