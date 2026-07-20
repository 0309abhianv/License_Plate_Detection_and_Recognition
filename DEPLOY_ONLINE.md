# Online Deployment Guide

This project can run as a central online dashboard. The hosted website stores detections in SQLite and checkpoint devices send plate detections to it through an API.

## What Runs Online

The online server runs:

- `web_dashboard.py`
- SQLite database
- dashboard pages
- `/api/detections` endpoint

The checkpoint devices run:

- `main.py`
- webcam/video detection
- OCR
- remote sending with `--server-url`

This is better than running OCR fully in the browser because OCR and OpenCV are heavy for small devices.

## Deploy The Dashboard

Upload these files to GitHub:

- `web_dashboard.py`
- `templates/`
- `static/`
- `src/`
- `requirements-web.txt`
- `Procfile`

Do not upload:

- `outputs/`
- `.venv/`
- videos
- model files

On a hosting service, use:

```text
Build command: pip install -r requirements-web.txt
Start command: gunicorn web_dashboard:app
```

Set environment variables:

```text
CHECKPOINT_API_KEY=choose-any-secret-key
SECRET_KEY=choose-any-secret-key
```

After deployment, your dashboard will have a URL like:

```text
https://your-app-name.example.com
```

## Send Detections From A Checkpoint

On each checkpoint device, run:

```powershell
python main.py --source 0 --display --frame-skip 5 --max-candidates 2 --fast-ocr --cpu --min-confidence 0.65 --duplicate-seconds 15 --save-crops --server-url "https://your-online-dashboard-url" --api-key "your-secret-key" --camera-location "Main Gate"
```

For another checkpoint:

```powershell
python main.py --source 0 --display --frame-skip 5 --max-candidates 2 --fast-ocr --cpu --min-confidence 0.65 --duplicate-seconds 15 --save-crops --server-url "https://your-online-dashboard-url" --api-key "your-secret-key" --camera-location "Parking Entry"
```

## Test API Manually

```powershell
Invoke-RestMethod -Method Post "https://your-online-dashboard-url/api/detections" -Headers @{"X-API-Key"="your-secret-key"} -ContentType "application/json" -Body '{"plate_text":"AP03TV6368","confidence":0.95,"camera_location":"Main Gate","source":"test"}'
```

## Important Note

Free hosting may reset local SQLite files when the server restarts. For a real production system, use a hosted PostgreSQL database. For college demo and prototype usage, SQLite is fine if the hosting platform supports persistent disk.
