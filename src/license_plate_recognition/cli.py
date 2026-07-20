from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and recognize car license plates from a video in real time."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video/image path or webcam index. Use 0 for the default webcam.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the annotated output video.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path("outputs/plate_log.csv"),
        help="CSV file where recognized plate data will be stored.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable CSV logging.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("outputs/plates.db"),
        help="SQLite database where recognized plate data will be stored.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable SQLite database logging.",
    )
    parser.add_argument(
        "--camera-location",
        default="Main Gate",
        help="Location name stored with each detection, for example Main Gate.",
    )
    parser.add_argument(
        "--server-url",
        help="Online dashboard URL that receives detections, for example https://your-app.onrender.com.",
    )
    parser.add_argument(
        "--api-key",
        help="Optional API key required by the online dashboard.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped plate images beside the CSV log.",
    )
    parser.add_argument(
        "--duplicate-seconds",
        type=float,
        default=5.0,
        help="Do not log the same plate again within this many seconds.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum OCR confidence required before storing a plate.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display annotated frames while processing.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=8,
        help="Run OCR every N frames and reuse the latest result between OCR frames.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=2,
        help="Maximum plate candidates to send to OCR per processed frame.",
    )
    parser.add_argument(
        "--fast-ocr",
        action="store_true",
        help="Use fewer OCR image variants for faster webcam processing.",
    )
    parser.add_argument(
        "--sync-ocr",
        action="store_true",
        help="Run OCR on the main video loop instead of using the background worker.",
    )
    parser.add_argument(
        "--async-ocr",
        action="store_true",
        help="Force background OCR. Useful for webcam smoothness, but not recommended for moving video files.",
    )
    parser.add_argument(
        "--full-frame-ocr",
        action="store_true",
        help="High-accuracy mode: also run OCR on the full frame to find plates missed by contour detection.",
    )
    parser.add_argument(
        "--full-frame-width",
        type=int,
        default=960,
        help="Resize full-frame OCR to this width. Lower is faster; higher is better for tiny plates.",
    )
    parser.add_argument(
        "--detection-ttl-frames",
        type=int,
        default=8,
        help="Number of frames to keep a recognized plate on screen after OCR finds it.",
    )
    parser.add_argument(
        "--stop-after-detections",
        type=int,
        default=0,
        help="Stop processing after this many valid detections. Use 0 to scan the whole video.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many frames. Use 0 to disable progress messages.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1500,
        help="Minimum contour area for a plate candidate.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=40000,
        help="Maximum contour area for a plate candidate.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="EasyOCR language codes, for example: en hi",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force EasyOCR to run on CPU.",
    )
    return parser


def normalize_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def should_use_async_ocr(source: str, sync_ocr: bool, async_ocr: bool) -> bool:
    if sync_ocr:
        return False
    if async_ocr:
        return True
    return source.isdigit()


def main() -> int:
    args = build_parser().parse_args()

    from .detector import PlateDetector
    from .ocr import PlateReader
    from .storage import PlateLogger
    from .video import process_video

    detector = PlateDetector(min_area=args.min_area, max_area=args.max_area)
    reader = PlateReader(
        languages=args.languages,
        use_gpu=not args.cpu,
        fast_mode=args.fast_ocr,
        full_frame_width=max(480, args.full_frame_width),
    )
    logger = None
    if not args.no_log:
        logger = PlateLogger(
            csv_path=args.log_csv,
            crop_dir=args.log_csv.parent / "plate_crops" if args.save_crops else None,
            duplicate_seconds=max(0.0, args.duplicate_seconds),
            min_confidence=max(0.0, args.min_confidence),
            db_path=None if args.no_db else args.db,
            camera_location=args.camera_location,
            server_url=args.server_url,
            api_key=args.api_key,
        )

    process_video(
        source=normalize_source(args.source),
        detector=detector,
        reader=reader,
        output_path=args.output,
        display=args.display,
        frame_skip=max(1, args.frame_skip),
        max_candidates=max(1, args.max_candidates),
        async_ocr=should_use_async_ocr(args.source, args.sync_ocr, args.async_ocr),
        logger=logger,
        full_frame_ocr=args.full_frame_ocr,
        detection_ttl_frames=max(0, args.detection_ttl_frames),
        min_display_confidence=max(0.0, args.min_confidence),
        stop_after_detections=max(0, args.stop_after_detections),
        progress_every=max(0, args.progress_every),
    )
    return 0
