import math
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
import torch
from inference import get_model
from shapely.geometry import Point, Polygon


ROBOFLOW_API_KEY = "0BHeHlDTavVYGbSOAiQa"
CAR_MODEL_ID = "driftcars/5"
CLIP_MODEL_ID = "clippingpoints/4"
DEFAULT_CAR_CONFIDENCE = 0.7
DEFAULT_CLIP_CONFIDENCE = 0.5
DEFAULT_SKIP_FACTOR = 2
DEFAULT_OUTPUT_FPS = 30
ZONE_MISSED_DISTANCE_DELTA = 75
ZONE_EXIT_BUFFER_THRESHOLD = 5

FRONT_IDX = 0
REAR_IDX = 4
CAR_SKELETON_EDGES = [(1, 2), (1, 3), (1, 4), (4, 5), (5, 6), (5, 7)]
ZONE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]

_car_model = None
_clip_model = None
_car_model_lock = threading.Lock()
_clip_model_lock = threading.Lock()


def _emit_log(message, log_callback=None):
    print(message)
    if log_callback is not None:
        log_callback(message)


def _print_cuda_status(log_callback=None):
    if torch.cuda.is_available():
        _emit_log(f"CUDA Available: True ({torch.cuda.get_device_name(0)})", log_callback)
    else:
        _emit_log("CUDA Available: False", log_callback)


def _get_car_model(log_callback=None):
    global _car_model
    if _car_model is None:
        with _car_model_lock:
            if _car_model is None:
                _emit_log(f"Loading car detection model: {CAR_MODEL_ID}", log_callback)
                _print_cuda_status(log_callback)
                _car_model = get_model(model_id=CAR_MODEL_ID, api_key=ROBOFLOW_API_KEY)
                _emit_log("Car detection model loaded.", log_callback)
    return _car_model


def _get_clip_model(log_callback=None):
    global _clip_model
    if _clip_model is None:
        with _clip_model_lock:
            if _clip_model is None:
                _emit_log(f"Loading clipping-point model: {CLIP_MODEL_ID}", log_callback)
                _clip_model = get_model(model_id=CLIP_MODEL_ID, api_key=ROBOFLOW_API_KEY)
                _emit_log("Clipping-point model loaded.", log_callback)
    return _clip_model


def warmup_models(log_callback=None):
    started_at = time.time()
    _emit_log("Starting model warmup...", log_callback)
    _get_car_model(log_callback)
    _get_clip_model(log_callback)
    _emit_log(f"Model warmup complete in {time.time() - started_at:.2f}s.", log_callback)


def _calculate_drift_angle(points):
    if len(points) <= REAR_IDX:
        return 0.0

    front_point = points[FRONT_IDX]
    rear_point = points[REAR_IDX]
    if np.all(front_point == 0) or np.all(rear_point == 0):
        return 0.0

    dx = front_point[0] - rear_point[0]
    dy = front_point[1] - rear_point[1]
    angle_degrees = math.degrees(math.atan2(dx, dy))
    return 180 - abs(angle_degrees)


def _draw_drift_angle(frame, points, drift_angle):
    if drift_angle <= 0:
        return frame

    front_point = points[FRONT_IDX]
    text = f"Angle: {drift_angle:.2f} deg"
    pos = (int(front_point[0]), int(front_point[1] - 20))

    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return frame


def _draw_zone_hit_banner(frame, frame_width):
    text = "ZONE HIT"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = frame_width - text_size[0] - 50

    cv2.putText(frame, text, (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(frame, text, (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return frame


def _draw_scoreboard(frame, judging_state, total_zones):
    angle_score = sum(judging_state["zone_angle_scores"].values())
    line_score = sum(judging_state["zone_line_scores"].values())
    total_score = angle_score + line_score

    angle_text = f"Angle Score: {angle_score:.1f}"
    line_text = f"Line Score: {line_score:.1f}"
    total_text = f"Total: {total_score:.1f}"

    if judging_state["current_zone_idx"] >= total_zones:
        zone_text = "FINISH"
    else:
        zone_text = f"Next Zone: {judging_state['current_zone_idx'] + 1}/{total_zones}"

    cv2.putText(frame, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, line_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, total_text, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, zone_text, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def _build_score_summary(judging_state):
    angle_score = round(sum(judging_state["zone_angle_scores"].values()), 2)
    line_score = round(sum(judging_state["zone_line_scores"].values()), 2)
    total_score = round(angle_score + line_score, 2)
    return {
        "angle_score": angle_score,
        "line_score": line_score,
        "total_score": total_score,
    }


def _advance_zone(judging_state):
    judging_state["current_zone_idx"] += 1
    judging_state["is_inside_zone"] = False
    judging_state["best_angle_in_zone"] = 0.0
    judging_state["exit_buffer"] = 0
    judging_state["closest_zone_distance"] = None


def process_video(
    input_path,
    output_path,
    active_config=None,
    car_confidence=DEFAULT_CAR_CONFIDENCE,
    clip_confidence=DEFAULT_CLIP_CONFIDENCE,
    skip_factor=DEFAULT_SKIP_FACTOR,
    progress_callback=None,
    log_callback=None,
):
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_file}")

    out = None
    judging_state = {
        "current_zone_idx": 0,
        "zone_angle_scores": {},
        "zone_line_scores": {},
        "is_inside_zone": False,
        "best_angle_in_zone": 0.0,
        "exit_buffer": 0,
        "closest_zone_distance": None,
        "zone_visible_in_frame": False,
    }

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_evaluated_frames = (
            max((total_source_frames + max(skip_factor, 1) - 1) // max(skip_factor, 1), 0)
            if total_source_frames > 0
            else 0
        )
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        base_fps = float(source_fps) if source_fps and source_fps > 0 else DEFAULT_OUTPUT_FPS
        output_fps = max(base_fps / max(skip_factor, 1), 1.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_file), fourcc, output_fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_file}")

        car_model = _get_car_model(log_callback)
        clip_model = _get_clip_model(log_callback)
        _emit_log(
            f"Video opened: {input_file.name} ({width}x{height}), {total_evaluated_frames} evaluated frames expected.",
            log_callback,
        )

        if active_config is not None:
            _emit_log(
                (
                    "Active config received: "
                    f"{active_config.get('configuration_name', '')}, "
                    f"clipping_points={active_config.get('clipping_points', 0)}, "
                    f"max_trajectory_score={active_config.get('max_trajectory_score', 0)}, "
                    f"max_angle_score={active_config.get('max_angle_score', 0)}"
                ),
                log_callback,
            )

        points_config = active_config.get("points", []) if active_config is not None else []
        total_zones = len(points_config)
        max_angle_score = float(active_config.get("max_angle_score", 0) or 0) if active_config is not None else 0.0
        max_trajectory_score = (
            float(active_config.get("max_trajectory_score", 0) or 0) if active_config is not None else 0.0
        )
        max_angle_per_zone = (max_angle_score / total_zones) if total_zones > 0 else 0.0
        traj_score_per_zone = (max_trajectory_score / total_zones) if total_zones > 0 else 0.0

        car_vertex_ann = sv.VertexAnnotator(color=sv.Color.RED, radius=5)
        car_edge_ann = sv.EdgeAnnotator(
            color=sv.Color.from_rgb_tuple((255, 0, 255)),
            thickness=2,
            edges=CAR_SKELETON_EDGES,
        )
        incomplete_ann = sv.EdgeAnnotator(
            color=sv.Color.RED,
            thickness=2,
            edges=ZONE_EDGES,
        )
        complete_ann = sv.EdgeAnnotator(
            color=sv.Color.GREEN,
            thickness=4,
            edges=ZONE_EDGES,
        )

        frame_count = 0
        evaluated_frames = 0
        started_at = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count % skip_factor != 0:
                frame_count += 1
                continue

            annotated_frame = frame.copy()
            hit_detected = False
            car_center = None
            drift_angle = 0.0
            front_points = []
            rear_points = []
            judging_state["zone_visible_in_frame"] = False

            car_results = car_model.infer(frame, confidence=car_confidence)[0]
            car_kp = sv.KeyPoints.from_inference(car_results)

            if not car_kp.is_empty():
                car_points = car_kp.xy[0]
                if len(car_points) > REAR_IDX and not np.all(car_points[REAR_IDX] == 0):
                    car_center = car_points[REAR_IDX]

                for idx, point in enumerate(car_points):
                    if np.all(point == 0):
                        continue

                    shapely_point = Point(point[0], point[1])
                    if idx <= 3:
                        front_points.append(shapely_point)
                    else:
                        rear_points.append(shapely_point)

                drift_angle = _calculate_drift_angle(car_points)
                annotated_frame = _draw_drift_angle(annotated_frame, car_points, drift_angle)
                annotated_frame = car_edge_ann.annotate(
                    scene=annotated_frame,
                    key_points=car_kp,
                )
                annotated_frame = car_vertex_ann.annotate(
                    scene=annotated_frame,
                    key_points=car_kp,
                )

            clip_results = clip_model.infer(frame, confidence=clip_confidence)[0]
            clip_kp = sv.KeyPoints.from_inference(clip_results)

            if not clip_kp.is_empty() and car_center is not None and judging_state["current_zone_idx"] < total_zones:
                target_zone_data = points_config[judging_state["current_zone_idx"]]
                valid_polygons = []

                for index in range(len(clip_kp)):
                    individual_clip = clip_kp[index : index + 1]
                    coords = individual_clip.xy[0]
                    valid_count = np.sum(np.any(coords != 0, axis=1))

                    if valid_count == 4:
                        polygon = Polygon([tuple(point) for point in coords])
                        distance = np.linalg.norm(car_center - np.mean(coords, axis=0))
                        valid_polygons.append((polygon, distance, index))

                valid_polygons.sort(key=lambda item: item[1])

                if valid_polygons:
                    judging_state["zone_visible_in_frame"] = True
                    active_poly, best_dist, active_clip_idx = valid_polygons[0]
                    if judging_state["closest_zone_distance"] is None:
                        judging_state["closest_zone_distance"] = best_dist
                    else:
                        judging_state["closest_zone_distance"] = min(
                            judging_state["closest_zone_distance"],
                            best_dist,
                        )
                    wheel_reference = target_zone_data.get("wheel_reference", "back wheels")
                    reference_points = front_points if wheel_reference == "front wheels" else rear_points
                    is_hitting = any(active_poly.contains(car_point) for car_point in reference_points)
                    active_clip = clip_kp[active_clip_idx : active_clip_idx + 1]

                    moving_away_without_hit = (
                        not judging_state["is_inside_zone"]
                        and judging_state["closest_zone_distance"] is not None
                        and best_dist > judging_state["closest_zone_distance"] + ZONE_MISSED_DISTANCE_DELTA
                    )

                    if is_hitting:
                        hit_detected = True
                        judging_state["is_inside_zone"] = True
                        judging_state["exit_buffer"] = 0
                        judging_state["best_angle_in_zone"] = max(
                            judging_state["best_angle_in_zone"],
                            drift_angle,
                        )
                        annotated_frame = complete_ann.annotate(
                            scene=annotated_frame,
                            key_points=active_clip,
                        )
                    else:
                        if judging_state["is_inside_zone"]:
                            judging_state["exit_buffer"] += 1
                        elif moving_away_without_hit:
                            judging_state["exit_buffer"] += 1
                        else:
                            judging_state["exit_buffer"] = 0
                        annotated_frame = incomplete_ann.annotate(
                            scene=annotated_frame,
                            key_points=active_clip,
                        )

                    if judging_state["is_inside_zone"] and (
                        judging_state["exit_buffer"] > ZONE_EXIT_BUFFER_THRESHOLD
                    ):
                        zone_index = target_zone_data["index"]
                        target_angle = float(target_zone_data["target_angle"])
                        actual_angle = judging_state["best_angle_in_zone"]

                        diff = max(0.0, target_angle - actual_angle)
                        angle_score = round(max(0.0, max_angle_per_zone * (1 - (diff / 25))), 2)
                        line_score = round(traj_score_per_zone, 2)

                        judging_state["zone_angle_scores"][zone_index] = angle_score
                        judging_state["zone_line_scores"][zone_index] = line_score

                        _emit_log(
                            f"Zone {zone_index} finished. Angle Score: {angle_score} Line Score: {line_score}",
                            log_callback,
                        )
                        _advance_zone(judging_state)
                    elif (
                        not judging_state["is_inside_zone"]
                        and moving_away_without_hit
                        and judging_state["exit_buffer"] > ZONE_EXIT_BUFFER_THRESHOLD
                    ):
                        zone_index = target_zone_data["index"]
                        _emit_log(f"Zone {zone_index} missed.", log_callback)
                        _advance_zone(judging_state)

            if judging_state["is_inside_zone"] and not judging_state["zone_visible_in_frame"]:
                judging_state["exit_buffer"] += 1

                if judging_state["exit_buffer"] > ZONE_EXIT_BUFFER_THRESHOLD and judging_state["current_zone_idx"] < total_zones:
                    target_zone_data = points_config[judging_state["current_zone_idx"]]
                    zone_index = target_zone_data["index"]
                    target_angle = float(target_zone_data["target_angle"])
                    actual_angle = judging_state["best_angle_in_zone"]

                    diff = max(0.0, target_angle - actual_angle)
                    angle_score = round(max(0.0, max_angle_per_zone * (1 - (diff / 25))), 2)
                    line_score = round(traj_score_per_zone, 2)

                    judging_state["zone_angle_scores"][zone_index] = angle_score
                    judging_state["zone_line_scores"][zone_index] = line_score

                    _emit_log(
                        f"Zone {zone_index} finished. Angle Score: {angle_score} Line Score: {line_score}",
                        log_callback,
                    )
                    _advance_zone(judging_state)

            if hit_detected:
                annotated_frame = _draw_zone_hit_banner(annotated_frame, width)

            if total_zones > 0:
                annotated_frame = _draw_scoreboard(annotated_frame, judging_state, total_zones)

            out.write(annotated_frame)
            evaluated_frames += 1
            if progress_callback is not None:
                progress_callback(
                    evaluated_frames,
                    total_evaluated_frames,
                    time.time() - started_at,
                )
            frame_count += 1

    finally:
        cap.release()
        if out is not None:
            out.release()

    _emit_log("Process complete. Review output file.", log_callback)
    return {
        "output_file": str(output_file),
        "scores": _build_score_summary(judging_state),
    }
