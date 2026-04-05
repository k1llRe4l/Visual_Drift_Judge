import math
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from inference import get_model


ROBOFLOW_API_KEY = "0BHeHlDTavVYGbSOAiQa"
MODEL_ID = "driftcars/5"
DEFAULT_CONFIDENCE = 0.7
DEFAULT_SKIP_FACTOR = 2
DEFAULT_OUTPUT_FPS = 30

FRONT_IDX = 0
REAR_IDX = 4
SKELETON_EDGES = [(1, 2), (1, 3), (1, 4), (4, 5), (5, 6), (5, 7)]

_model = None


def _get_model():
    global _model
    if _model is None:
        if torch.cuda.is_available():
            print(f"CUDA Available: True ({torch.cuda.get_device_name(0)})")
        else:
            print("CUDA Available: False")

        _model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)

    return _model


def process_video(input_path, output_path, confidence=DEFAULT_CONFIDENCE, skip_factor=DEFAULT_SKIP_FACTOR):
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
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = int(source_fps) if source_fps and source_fps > 0 else DEFAULT_OUTPUT_FPS

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_file}")

        model = _get_model()
        keypoint_annotator = sv.VertexAnnotator(color=sv.Color.WHITE, radius=5)
        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.WHITE,
            thickness=2,
            edges=SKELETON_EDGES,
        )

        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            annotated_frame = frame.copy()

            if frame_count % skip_factor == 0:
                results = model.infer(frame, confidence=confidence)[0]
                keypoints = sv.KeyPoints.from_inference(results)

                if not keypoints.is_empty():
                    points = keypoints.xy[0]

                    if len(points) > REAR_IDX:
                        p1, p2 = points[FRONT_IDX], points[REAR_IDX]

                        if not np.all(p1 == 0) and not np.all(p2 == 0):
                            dx = p1[0] - p2[0]
                            dy = p1[1] - p2[1]
                            angle_degrees = math.degrees(math.atan2(dx, dy))
                            drift_angle = 180 - abs(angle_degrees)

                            text = f"Angle: {drift_angle:.2f} deg"
                            pos = (int(p1[0]), int(p1[1] - 20))

                            cv2.putText(
                                annotated_frame,
                                text,
                                pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (79, 79, 79),
                                4,
                            )
                            cv2.putText(
                                annotated_frame,
                                text,
                                pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 255),
                                2,
                            )

                    annotated_frame = edge_annotator.annotate(
                        scene=annotated_frame,
                        key_points=keypoints,
                    )
                    annotated_frame = keypoint_annotator.annotate(
                        scene=annotated_frame,
                        key_points=keypoints,
                    )

            out.write(annotated_frame)
            frame_count += 1
    finally:
        cap.release()
        if out is not None:
            out.release()

    return output_file
