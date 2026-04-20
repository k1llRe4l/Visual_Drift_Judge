import os
os.environ["CORE_MODEL_SAM3_ENABLED"] = "True"
import json
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "resources" / "videos"
UPLOAD_DIR = VIDEOS_DIR / "uploads"
OUTPUT_DIR = VIDEOS_DIR / "processed"
CONFIGS_DIR = BASE_DIR / "resources" / "configs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

ACTIVE_CONFIG_INPUT = {
    "configuration_name": "",
    "clipping_points": 0,
    "points": [],
}
PROCESS_JOBS = {}
PROCESS_JOBS_LOCK = threading.Lock()


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def try_delete_file(path, attempts=5, delay=0.2):
    file_path = Path(path)
    for attempt in range(attempts):
        try:
            if file_path.exists():
                file_path.unlink()
            return
        except PermissionError:
            if attempt == attempts - 1:
                app.logger.warning("Could not delete locked file: %s", file_path)
                return
            time.sleep(delay)


def empty_directory(directory):
    directory_path = Path(directory)
    if not directory_path.exists():
        return

    for item in directory_path.iterdir():
        if item.is_file():
            try_delete_file(item)


def update_process_job(job_id, **updates):
    with PROCESS_JOBS_LOCK:
        job = PROCESS_JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)


def append_process_log(job_id, message):
    print(message)
    with PROCESS_JOBS_LOCK:
        job = PROCESS_JOBS.get(job_id)
        if job is None:
            return
        job.setdefault("logs", []).append(message)


def run_process_job(job_id, input_path, output_path, output_name):
    started_at = time.time()

    try:
        from model import process_video

        append_process_log(job_id, f"Starting video evaluation for {Path(input_path).name}")

        def progress_callback(evaluated_frames, total_evaluated_frames, elapsed_seconds):
            eta_seconds = None
            if evaluated_frames > 0 and total_evaluated_frames >= evaluated_frames:
                eta_seconds = max(
                    (elapsed_seconds / evaluated_frames) * (total_evaluated_frames - evaluated_frames),
                    0.0,
                )

            update_process_job(
                job_id,
                status="processing",
                evaluated_frames=evaluated_frames,
                total_evaluated_frames=total_evaluated_frames,
                eta_seconds=eta_seconds,
            )

        def log_callback(message):
            append_process_log(job_id, message)

        process_video(
            input_path,
            output_path,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

        append_process_log(job_id, f"Processing complete. Saved as {output_name} in the processed folder.")
        append_process_log(job_id, f"Output path: {output_path}")

        update_process_job(
            job_id,
            status="completed",
            processed_file=output_name,
            processed_path=str(output_path),
            completed_at=time.time(),
            elapsed_seconds=time.time() - started_at,
        )
    except Exception as exc:
        empty_directory(OUTPUT_DIR)
        append_process_log(job_id, f"Processing failed: {exc}")
        update_process_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=time.time(),
            elapsed_seconds=time.time() - started_at,
        )
    finally:
        empty_directory(UPLOAD_DIR)


@app.route("/")
def index():
    config_names = sorted(path.stem for path in CONFIGS_DIR.glob("*.json"))
    return render_template("config_select.html", config_names=config_names)


@app.route("/config")
def config_page():
    return render_template("config_edit.html")


@app.route("/upload")
def upload_page():
    return render_template("video_upload.html")


def get_config_path(config_name):
    safe_name = secure_filename(config_name).strip()
    if not safe_name:
        raise ValueError("Configuration name is required.")

    return CONFIGS_DIR / f"{safe_name}.json"


@app.get("/api/configs/<config_name>")
def load_config(config_name):
    config_path = get_config_path(config_name)
    if not config_path.exists():
        return jsonify({"error": "Configuration file not found."}), 404

    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    return jsonify(data)


@app.post("/api/configs")
def save_config():
    payload = request.get_json(silent=True) or {}
    config_name = (payload.get("configuration_name") or "").strip()
    clipping_points = payload.get("clipping_points")
    points = payload.get("points") or []

    if not config_name:
        return jsonify({"error": "Configuration name is required."}), 400

    try:
        clipping_points = int(clipping_points)
    except (TypeError, ValueError):
        return jsonify({"error": "Number of clipping points must be a whole number."}), 400

    if clipping_points < 1:
        return jsonify({"error": "Number of clipping points must be at least 1."}), 400

    if len(points) != clipping_points:
        return jsonify({"error": "Each clipping point must have its own settings."}), 400

    normalized_points = []
    for index, point in enumerate(points, start=1):
        wheel_reference = point.get("wheel_reference", "back wheels")
        if wheel_reference not in {"back wheels", "front wheels"}:
            return jsonify({"error": f"Invalid wheel setting for clipping point {index}."}), 400

        normalized_points.append(
            {
                "index": index,
                "target_angle": point.get("target_angle", ""),
                "wheel_reference": wheel_reference,
            }
        )

    config_data = {
        "configuration_name": config_name,
        "clipping_points": clipping_points,
        "points": normalized_points,
    }

    config_path = get_config_path(config_name)
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, ensure_ascii=False, indent=2)

    return jsonify(
        {
            "message": "Configuration saved successfully.",
            "filename": config_path.name,
        }
    )


@app.post("/api/configs/continue")
def continue_with_config():
    payload = request.get_json(silent=True) or {}
    config_name = (payload.get("configuration_name") or "").strip()
    clipping_points = payload.get("clipping_points")
    points = payload.get("points") or []

    if not config_name:
        return jsonify({"error": "Configuration name is required."}), 400

    try:
        clipping_points = int(clipping_points)
    except (TypeError, ValueError):
        return jsonify({"error": "Number of clipping points must be a whole number."}), 400

    if clipping_points < 1:
        return jsonify({"error": "Number of clipping points must be at least 1."}), 400

    if len(points) != clipping_points:
        return jsonify({"error": "Each clipping point must have its own settings."}), 400

    normalized_points = []
    for index, point in enumerate(points, start=1):
        target_angle = str(point.get("target_angle", "")).strip()
        wheel_reference = point.get("wheel_reference", "back wheels")

        if target_angle == "":
            return jsonify({"error": f"Target angle is required for clipping point {index}."}), 400

        if wheel_reference not in {"back wheels", "front wheels"}:
            return jsonify({"error": f"Invalid wheel setting for clipping point {index}."}), 400

        normalized_points.append(
            {
                "index": index,
                "target_angle": target_angle,
                "wheel_reference": wheel_reference,
            }
        )

    ACTIVE_CONFIG_INPUT["configuration_name"] = config_name
    ACTIVE_CONFIG_INPUT["clipping_points"] = clipping_points
    ACTIVE_CONFIG_INPUT["points"] = normalized_points
    # TODO: продумать логику передачи информации о текущем конфиге в модель и способ оценки заезда

    print("Received config input from page:")
    print(f"  configuration_name: {ACTIVE_CONFIG_INPUT['configuration_name']}")
    print(f"  clipping_points: {ACTIVE_CONFIG_INPUT['clipping_points']}")
    print("  points:")
    for point in ACTIVE_CONFIG_INPUT["points"]:
        print(
            f"    - index={point['index']}, "
            f"target_angle={point['target_angle']}, "
            f"wheel_reference={point['wheel_reference']}"
        )

    return jsonify(
        {
            "message": "Configuration moved into app state.",
            "active_config": ACTIVE_CONFIG_INPUT,
        }
    )


@app.post("/process")
def process():
    file = request.files.get("video")
    if file is None or file.filename == "":
        return jsonify({"error": "Please upload a video file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    safe_name = secure_filename(file.filename)
    stem = Path(safe_name).stem or "video"
    job_id = uuid.uuid4().hex

    input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    output_name = f"{stem}_processed.mp4"
    output_path = OUTPUT_DIR / f"{job_id}_{output_name}"

    file.save(input_path)
    with PROCESS_JOBS_LOCK:
        PROCESS_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "uploaded_file": input_path.name,
            "uploaded_path": str(input_path),
            "processed_file": output_name,
            "processed_path": str(output_path),
            "evaluated_frames": 0,
            "total_evaluated_frames": 0,
            "eta_seconds": None,
            "logs": [],
        }

    worker = threading.Thread(
        target=run_process_job,
        args=(job_id, input_path, output_path, output_name),
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id}), 202


@app.get("/process/<job_id>")
def process_status(job_id):
    with PROCESS_JOBS_LOCK:
        job = PROCESS_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Processing job not found."}), 404
        return jsonify(job)


if __name__ == "__main__":
    app.run(debug=True)
