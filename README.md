# Detect and Recognize Car License Plate from a Video in Real Time

This project implements a real-time license plate detection and recognition pipeline inspired by the GeeksforGeeks article "Detect and Recognize Car License Plate from a video in real time".

The article uses OpenCV preprocessing, contour filtering, character segmentation, and a TensorFlow OCR model. This version keeps the OpenCV plate-localization approach, then uses EasyOCR for recognition so the project can run without the article's missing `.pb` model files.

## What It Does

- Reads frames from a video file or webcam.
- Detects license-plate-like rectangular regions using OpenCV.
- Crops and enhances likely plates.
- Recognizes plate text with EasyOCR.
- Draws bounding boxes and recognized text on the video stream.
- Stores recognized plate data in CSV and SQLite.
- Optionally saves cropped plate images and an annotated output video.
- Sends detections to an online dashboard from multiple checkpoints.

## Project Structure

```text
.
|-- main.py
|-- web_dashboard.py
|-- Procfile
|-- requirements.txt
|-- requirements-web.txt
|-- README.md
|-- DEPLOY_ONLINE.md
|-- static
|-- templates
`-- src
    `-- license_plate_recognition
        |-- __init__.py
        |-- cli.py
        |-- detector.py
        |-- ocr.py
        |-- storage.py
        `-- video.py
```

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

EasyOCR may download its OCR model the first time you run the project.

## Usage

Run on a video file:

```powershell
python main.py --source path\to\traffic_video.mp4 --display
```

Run on your webcam:

```powershell
python main.py --source 0 --display --frame-skip 12 --max-candidates 1 --fast-ocr --cpu
```

Save an annotated output video:

```powershell
python main.py --source path\to\traffic_video.mp4 --output outputs\annotated.mp4
```

Store recognized plate data in CSV:

```powershell
python main.py --source path\to\traffic_video.mp4 --log-csv outputs\plate_log.csv
```

Store recognized plate data in SQLite:

```powershell
python main.py --source 0 --display --db outputs\plates.db --camera-location "Main Gate"
```

Start the local dashboard:

```powershell
python web_dashboard.py
```

Then open:

```text
http://127.0.0.1:5000
```

Send detections to an online dashboard:

```powershell
python main.py --source 0 --display --frame-skip 5 --max-candidates 2 --fast-ocr --cpu --min-confidence 0.65 --server-url "https://your-online-dashboard-url" --api-key "your-secret-key" --camera-location "Main Gate"
```

Store data and cropped plate images:

```powershell
python main.py --source path\to\traffic_video.mp4 --log-csv outputs\plate_log.csv --save-crops
```

Press `q` in the video window to stop.

## Stored Data

The CSV file contains:

```text
timestamp, source, frame_number, elapsed_ms, plate_text, confidence,
bbox_x, bbox_y, bbox_width, bbox_height, crop_path
```

By default, duplicate plate text is not stored again for 5 seconds. You can change that with `--duplicate-seconds`.

## Useful Options

```text
--source             Video path, image path, or webcam index. Default: 0
--output             Optional annotated output video path.
--log-csv            CSV path for recognized plate data. Default: outputs/plate_log.csv
--no-log             Disable CSV logging.
--db                 SQLite database path. Default: outputs/plates.db
--no-db              Disable SQLite logging.
--camera-location    Location/checkpoint name stored with every detection.
--server-url         Online dashboard URL that receives detections.
--api-key            Optional API key for the online dashboard.
--save-crops         Save cropped plate images.
--duplicate-seconds  Do not store the same plate again within this many seconds.
--min-confidence     Minimum OCR confidence required before storing a plate.
--display            Show live annotated frames.
--frame-skip         Process OCR every N frames. Default: 8
--max-candidates     Maximum plate candidates to OCR per processed frame. Default: 2
--fast-ocr           Use fewer OCR image variants for faster webcam processing.
--sync-ocr           Disable background OCR worker.
--min-area           Minimum candidate plate area. Default: 1500
--max-area           Maximum candidate plate area. Default: 40000
--languages          EasyOCR language codes. Default: en
--cpu                Force EasyOCR to use CPU.
```

## Notes

This is a practical starter project, not a production ALPR system. Plate detection quality depends heavily on camera angle, plate size, blur, lighting, and regional plate formats. For production use, pair this pipeline with a trained plate detector such as YOLO and region-specific OCR validation rules.

Source analyzed: https://www.geeksforgeeks.org/python/detect-and-recognize-car-license-plate-from-a-video-in-real-time/

## Online Mode

For online use, deploy `web_dashboard.py` as the central server and run `main.py` on each checkpoint device. Each checkpoint sends its detections to the hosted dashboard using `--server-url`.

Read the full deployment guide in `DEPLOY_ONLINE.md`.
