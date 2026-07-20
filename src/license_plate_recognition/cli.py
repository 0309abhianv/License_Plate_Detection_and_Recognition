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
        async_ocr=not args.sync_ocr,
        logger=logger,
    )
    return 0
