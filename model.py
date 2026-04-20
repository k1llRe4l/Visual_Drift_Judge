import math
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
CLIP_MODEL_ID = "clippingpoints/3"
DEFAULT_CONFIDENCE = 0.7
DEFAULT_CLIP_CONFIDENCE = 0.5
DEFAULT_SKIP_FACTOR = 2
DEFAULT_OUTPUT_FPS = 30

FRONT_IDX = 0
REAR_IDX = 4
CAR_SKELETON_EDGES = [(1, 2), (1, 3), (1, 4), (4, 5), (5, 6), (5, 7)]
ZONE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]

_car_model = None
_clip_model = None


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
        _print_cuda_status(log_callback)
        _car_model = get_model(model_id=CAR_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    return _car_model


def _get_clip_model(log_callback=None):
    global _clip_model
    if _clip_model is None:
        _clip_model = get_model(model_id=CLIP_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    return _clip_model


def _draw_drift_angle(frame, points):
    if len(points) <= REAR_IDX:
        return frame

    p1, p2 = points[FRONT_IDX], points[REAR_IDX]
    if np.all(p1 == 0) or np.all(p2 == 0):
        return frame

    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    angle_degrees = math.degrees(math.atan2(dx, dy))
    drift_angle = 180 - abs(angle_degrees)

    text = f"Angle: {drift_angle:.2f} deg"
    pos = (int(p1[0]), int(p1[1] - 20))

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


def process_video(
    input_path,
    output_path,
    confidence=DEFAULT_CONFIDENCE,
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
            all_car_points = []

            car_results = car_model.infer(frame, confidence=confidence)[0]
            car_kp = sv.KeyPoints.from_inference(car_results)

            if not car_kp.is_empty():
                car_points = car_kp.xy[0]

                for point in car_points:
                    if not np.all(point == 0):
                        all_car_points.append(Point(point[0], point[1]))

                annotated_frame = _draw_drift_angle(annotated_frame, car_points)
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

            if not clip_kp.is_empty():
                for index in range(len(clip_kp)):
                    individual_clip = clip_kp[index : index + 1]
                    coords = individual_clip.xy[0]
                    valid_count = np.sum(np.any(coords != 0, axis=1))

                    if valid_count == 4:
                        zone = Polygon([tuple(point) for point in coords])
                        if any(zone.contains(car_point) for car_point in all_car_points):
                            hit_detected = True
                        annotated_frame = complete_ann.annotate(
                            scene=annotated_frame,
                            key_points=individual_clip,
                        )
                    else:
                        annotated_frame = incomplete_ann.annotate(
                            scene=annotated_frame,
                            key_points=individual_clip,
                        )

            if hit_detected:
                annotated_frame = _draw_zone_hit_banner(annotated_frame, width)

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

    return output_file
