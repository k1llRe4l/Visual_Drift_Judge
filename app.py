import json
import os
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_file
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
    "max_trajectory_score": 0,
    "max_angle_score": 0,
    "points": [],
}
PROCESS_JOBS = {}
PROCESS_JOBS_LOCK = threading.Lock()
MODEL_WARMUP_STARTED = False
MODEL_WARMUP_LOCK = threading.Lock()


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_score_summary(base_scores=None, style_score=0.0):
    scores = dict(base_scores or {})
    line_score = round(_to_float(scores.get("line_score")), 2)
    angle_score = round(_to_float(scores.get("angle_score")), 2)
    style_score = round(_to_float(scores.get("style_score"), style_score), 2)
    total_score = round(line_score + angle_score + style_score, 2)

    return {
        "line_score": line_score,
        "angle_score": angle_score,
        "style_score": style_score,
        "total_score": total_score,
    }


def validate_style_score(style_score, score_limits, current_scores):
    try:
        style_score = float(style_score)
    except (TypeError, ValueError):
        raise ValueError("Style score must be a number.")

    if style_score < 0:
        raise ValueError("Style score must be at least 0.")

    max_trajectory_score = _to_float((score_limits or {}).get("max_trajectory_score"))
    max_angle_score = _to_float((score_limits or {}).get("max_angle_score"))
    max_style_score = max(0.0, 100.0 - max_angle_score - max_trajectory_score)

    if style_score > max_style_score:
        raise ValueError(f"Style score must not exceed {max_style_score:g}.")

    line_score = _to_float((current_scores or {}).get("line_score"))
    angle_score = _to_float((current_scores or {}).get("angle_score"))
    total_score = round(line_score + angle_score + style_score, 2)

    if total_score > 100:
        raise ValueError("Total score must not exceed 100.")

    return round(style_score, 2), total_score, round(max_style_score, 2)


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


def _startup_log(message):
    print(message)
    app.logger.info(message)


def warmup_models_in_background():
    global MODEL_WARMUP_STARTED

    with MODEL_WARMUP_LOCK:
        if MODEL_WARMUP_STARTED:
            return
        MODEL_WARMUP_STARTED = True

    def worker():
        try:
            _startup_log("Starting background model warmup...")
            from model import warmup_models

            warmup_models(log_callback=_startup_log)
        except Exception as exc:
            _startup_log(f"Background model warmup failed: {exc}")

    threading.Thread(target=worker, daemon=True).start()


def run_process_job(job_id, input_path, output_path, output_name):
    started_at = time.time()

    try:
        update_process_job(
            job_id,
            status="initializing",
            evaluated_frames=0,
            total_evaluated_frames=0,
            eta_seconds=None,
        )
        append_process_log(job_id, f"Starting video evaluation for {Path(input_path).name}")
        append_process_log(job_id, "Initializing processing job...")
        append_process_log(job_id, "Importing video processing pipeline...")

        from model import process_video

        append_process_log(job_id, "Video processing pipeline imported.")

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

        active_config = {
            "configuration_name": ACTIVE_CONFIG_INPUT["configuration_name"],
            "clipping_points": ACTIVE_CONFIG_INPUT["clipping_points"],
            "max_trajectory_score": ACTIVE_CONFIG_INPUT["max_trajectory_score"],
            "max_angle_score": ACTIVE_CONFIG_INPUT["max_angle_score"],
            "points": [point.copy() for point in ACTIVE_CONFIG_INPUT["points"]],
        }

        append_process_log(job_id, "Preparing active judging configuration...")
        append_process_log(
            job_id,
            (
                "Configuration ready: "
                f"{active_config['configuration_name']} "
                f"with {active_config['clipping_points']} clipping points."
            ),
        )
        append_process_log(job_id, "Loading detection models and warming up inference...")

        process_result = process_video(
            input_path,
            output_path,
            active_config=active_config,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
        with PROCESS_JOBS_LOCK:
            existing_job = PROCESS_JOBS.get(job_id) or {}
            existing_scores = dict(existing_job.get("scores") or {})

        score_summary = build_score_summary(
            (process_result or {}).get("scores"),
            style_score=existing_scores.get("style_score", 0.0),
        )

        append_process_log(job_id, "Frame processing finished. Finalizing output files...")
        append_process_log(job_id, f"Processing complete. Saved as {output_name} in the processed folder.")
        append_process_log(job_id, f"Output path: {output_path}")

        update_process_job(
            job_id,
            status="completed",
            processed_file=output_name,
            processed_path=str(output_path),
            scores=score_summary,
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


@app.route("/result/<job_id>")
def result_page(job_id):
    with PROCESS_JOBS_LOCK:
        job = PROCESS_JOBS.get(job_id)
        if job is None:
            return render_template("result_view.html", job_id=job_id, job=None), 404

        job_snapshot = dict(job)
        job_snapshot["scores"] = dict(job.get("scores") or {})

    return render_template("result_view.html", job_id=job_id, job=job_snapshot)


def get_config_path(config_name):
    safe_name = secure_filename(config_name).strip()
    if not safe_name:
        raise ValueError("Configuration name is required.")

    return CONFIGS_DIR / f"{safe_name}.json"


def validate_score_limits(payload):
    try:
        max_trajectory_score = int(payload.get("max_trajectory_score"))
        max_angle_score = int(payload.get("max_angle_score"))
    except (TypeError, ValueError):
        raise ValueError("Max. trajectory score and Max. angle score must be whole numbers.")

    if max_trajectory_score < 0 or max_angle_score < 0:
        raise ValueError("Max. trajectory score and Max. angle score must be at least 0.")

    if max_trajectory_score + max_angle_score > 100:
        raise ValueError("Сумма баллов по двум критериям не должна превышать 100")

    return max_trajectory_score, max_angle_score


@app.post("/process/<job_id>/style")
def update_style_score(job_id):
    payload = request.get_json(silent=True) or {}

    with PROCESS_JOBS_LOCK:
        job = PROCESS_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Processing job not found."}), 404

        score_limits = dict(job.get("score_limits") or {})
        current_scores = dict(job.get("scores") or {})

    try:
        style_score, total_score, max_style_score = validate_style_score(
            payload.get("style_score"),
            score_limits,
            current_scores,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with PROCESS_JOBS_LOCK:
        job = PROCESS_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Processing job not found."}), 404

        updated_scores = dict(job.get("scores") or {})
        updated_scores["style_score"] = style_score
        updated_scores["total_score"] = total_score
        job["scores"] = updated_scores

    return jsonify(
        {
            "message": "Style score updated successfully.",
            "scores": updated_scores,
            "max_style_score": max_style_score,
        }
    )


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

    try:
        max_trajectory_score, max_angle_score = validate_score_limits(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

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
        "max_trajectory_score": max_trajectory_score,
        "max_angle_score": max_angle_score,
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

    try:
        max_trajectory_score, max_angle_score = validate_score_limits(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

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
    ACTIVE_CONFIG_INPUT["max_trajectory_score"] = max_trajectory_score
    ACTIVE_CONFIG_INPUT["max_angle_score"] = max_angle_score
    ACTIVE_CONFIG_INPUT["points"] = normalized_points

    print("Received config input from page:")
    print(f"  configuration_name: {ACTIVE_CONFIG_INPUT['configuration_name']}")
    print(f"  clipping_points: {ACTIVE_CONFIG_INPUT['clipping_points']}")
    print(f"  max_trajectory_score: {ACTIVE_CONFIG_INPUT['max_trajectory_score']}")
    print(f"  max_angle_score: {ACTIVE_CONFIG_INPUT['max_angle_score']}")
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
            "score_limits": {
                "max_trajectory_score": ACTIVE_CONFIG_INPUT["max_trajectory_score"],
                "max_angle_score": ACTIVE_CONFIG_INPUT["max_angle_score"],
            },
            "evaluated_frames": 0,
            "total_evaluated_frames": 0,
            "eta_seconds": None,
            "scores": build_score_summary(),
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


@app.get("/download/<job_id>")
def download_processed_files(job_id):
    with PROCESS_JOBS_LOCK:
        job = PROCESS_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Processing job not found."}), 404
        if job.get("status") != "completed":
            return jsonify({"error": "Processing is not completed yet."}), 400
        processed_path = Path(job.get("processed_path") or "")
        download_name = job.get("processed_file") or processed_path.name

    if not processed_path.exists() or not processed_path.is_file():
        return jsonify({"error": "Processed file not found."}), 404

    response = send_file(
        processed_path,
        as_attachment=True,
        download_name=download_name,
        mimetype="video/mp4",
    )

    @response.call_on_close
    def cleanup_download_artifacts():
        empty_directory(OUTPUT_DIR)

    return response


if __name__ == "__main__":
    debug_mode = True
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not debug_mode:
        warmup_models_in_background()
    app.run(debug=debug_mode)
