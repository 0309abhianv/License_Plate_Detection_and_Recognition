from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import sys

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from license_plate_recognition.storage import PlateLogger


DB_PATH = Path(os.environ.get("PLATE_DB_PATH", PROJECT_ROOT / "outputs" / "plates.db"))
CSV_PATH = Path(os.environ.get("PLATE_CSV_PATH", PROJECT_ROOT / "outputs" / "plate_log.csv"))
API_KEY = os.environ.get("CHECKPOINT_API_KEY", "")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "license-plate-dashboard")


def get_connection() -> sqlite3.Connection:
    PlateLogger(csv_path=CSV_PATH, db_path=DB_PATH)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def api_authorized() -> bool:
    return not API_KEY or request.headers.get("X-API-Key") == API_KEY


@app.get("/")
def dashboard():
    search = request.args.get("search", "").strip().upper()
    status = request.args.get("status", "all")

    where = []
    params: list[object] = []
    if search:
        where.append("d.plate_text LIKE ?")
        params.append(f"%{search}%")
    if status == "authorized":
        where.append("d.is_authorized = 1")
    elif status == "unauthorized":
        where.append("d.is_authorized = 0")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with get_connection() as connection:
        stats = connection.execute(
            """
            SELECT
                COUNT(*) AS total_detections,
                COUNT(DISTINCT plate_text) AS total_vehicles,
                SUM(CASE WHEN is_authorized = 1 THEN 1 ELSE 0 END) AS authorized_hits,
                SUM(CASE WHEN is_authorized = 0 THEN 1 ELSE 0 END) AS unauthorized_hits
            FROM detections
            """
        ).fetchone()
        detections = connection.execute(
            f"""
            SELECT d.*, a.owner_name, a.vehicle_model
            FROM detections d
            LEFT JOIN authorized_vehicles a ON a.plate_text = d.plate_text
            {where_sql}
            ORDER BY d.timestamp DESC, d.id DESC
            LIMIT 100
            """,
            params,
        ).fetchall()
        vehicles = connection.execute(
            """
            SELECT
                d.plate_text,
                COUNT(*) AS detection_count,
                MAX(d.timestamp) AS last_seen,
                (
                    SELECT camera_location
                    FROM detections latest
                    WHERE latest.plate_text = d.plate_text
                    ORDER BY latest.timestamp DESC, latest.id DESC
                    LIMIT 1
                ) AS last_location,
                MAX(d.is_authorized) AS is_authorized,
                a.owner_name,
                a.vehicle_model
            FROM detections d
            LEFT JOIN authorized_vehicles a ON a.plate_text = d.plate_text
            GROUP BY d.plate_text
            ORDER BY last_seen DESC
            LIMIT 50
            """
        ).fetchall()

    return render_template(
        "dashboard.html",
        detections=detections,
        vehicles=vehicles,
        stats=stats,
        search=search,
        status=status,
    )


@app.post("/authorized")
def add_authorized_vehicle():
    plate_text = request.form["plate_text"].strip().upper().replace(" ", "")
    owner_name = request.form.get("owner_name", "").strip()
    vehicle_model = request.form.get("vehicle_model", "").strip()
    note = request.form.get("note", "").strip()
    if not plate_text:
        flash("Plate number is required.")
        return redirect(url_for("dashboard"))

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO authorized_vehicles (
                plate_text, owner_name, vehicle_model, note, created_at
            )
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(plate_text) DO UPDATE SET
                owner_name = excluded.owner_name,
                vehicle_model = excluded.vehicle_model,
                note = excluded.note
            """,
            (plate_text, owner_name, vehicle_model, note),
        )
        connection.execute(
            "UPDATE detections SET is_authorized = 1 WHERE plate_text = ?",
            (plate_text,),
        )
    return redirect(url_for("dashboard", search=plate_text))


@app.post("/api/detections")
def api_add_detection():
    if not api_authorized():
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    plate_text = str(data.get("plate_text", "")).strip().upper().replace(" ", "")
    if not plate_text:
        return jsonify({"error": "plate_text is required"}), 400

    timestamp = str(data.get("timestamp") or "")
    source = str(data.get("source") or "remote")
    camera_location = str(data.get("camera_location") or "Remote Checkpoint")
    crop_path = str(data.get("crop_path") or "")

    with get_connection() as connection:
        authorized = connection.execute(
            "SELECT 1 FROM authorized_vehicles WHERE plate_text = ?",
            (plate_text,),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO detections (
                timestamp, source, camera_location, frame_number, elapsed_ms,
                plate_text, confidence, bbox_x, bbox_y, bbox_width, bbox_height,
                crop_path, is_authorized
            )
            VALUES (
                COALESCE(NULLIF(?, ''), datetime('now')),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                timestamp,
                source,
                camera_location,
                int(data.get("frame_number") or 0),
                int(data.get("elapsed_ms") or 0),
                plate_text,
                float(data.get("confidence") or 0),
                int(data.get("bbox_x") or 0),
                int(data.get("bbox_y") or 0),
                int(data.get("bbox_width") or 0),
                int(data.get("bbox_height") or 0),
                crop_path,
                1 if authorized else 0,
            ),
        )

    return jsonify({"ok": True, "plate_text": plate_text})


@app.get("/vehicle/<plate_text>")
def vehicle_profile(plate_text: str):
    plate_text = plate_text.upper()
    with get_connection() as connection:
        vehicle = connection.execute(
            """
            SELECT
                d.plate_text,
                COUNT(*) AS detection_count,
                MIN(d.timestamp) AS first_seen,
                MAX(d.timestamp) AS last_seen,
                MAX(d.is_authorized) AS is_authorized,
                a.owner_name,
                a.vehicle_model,
                a.note
            FROM detections d
            LEFT JOIN authorized_vehicles a ON a.plate_text = d.plate_text
            WHERE d.plate_text = ?
            GROUP BY d.plate_text
            """,
            (plate_text,),
        ).fetchone()
        history = connection.execute(
            """
            SELECT *
            FROM detections
            WHERE plate_text = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 100
            """,
            (plate_text,),
        ).fetchall()

    if vehicle is None:
        return redirect(url_for("dashboard", search=plate_text))
    return render_template("vehicle.html", vehicle=vehicle, history=history)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
