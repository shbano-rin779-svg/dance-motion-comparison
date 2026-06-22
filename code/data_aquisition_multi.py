

"""
Detect up to two people with YOLO and extract pose landmark time-series for each person.

This first version assigns person_id by horizontal position in each frame:
    person_0: left person
    person_1: right person

Before running, place the MediaPipe model file here:
    models/pose_landmarker_lite.task

Install dependencies:
    pip install ultralytics opencv-python mediapipe pandas

Run as a script:
    python code/data_aquisition_multi.py

Use as a library:
    from data_aquisition_multi import run_multi_person_pose_extraction

    run_multi_person_pose_extraction(
        video_path=Path("video/test_multi.mov"),
        output_dir=Path("outputs/multi"),
    )

Outputs:
    <output_dir>/person_0_pose.csv
    <output_dir>/person_1_pose.csv
    <output_dir>/multi_person_annotated.mp4
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


VIDEO_PATH = Path("video/test_multi.mov")
POSE_MODEL_PATH = Path("models/pose_landmarker_lite.task")
YOLO_MODEL_PATH = "yolo11n.pt"
OUTPUT_DIR = Path("outputs/multi")
OUTPUT_VIDEO_FILENAME = "multi_person_annotated.mp4"
NUM_PERSONS = 2
PERSON_CONF_THRESHOLD = 0.4
CROP_MARGIN_RATIO = 0.15

LANDMARK_NAMES: List[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]


BBox = Tuple[int, int, int, int]


def build_csv_header() -> List[str]:
    """Create CSV columns for frame metadata, bbox, and pose landmarks."""
    header = [
        "person_id",
        "frame_index",
        "timestamp_sec",
        "pose_detected",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_conf",
    ]
    for name in LANDMARK_NAMES:
        header.extend(
            [
                f"{name}_x",
                f"{name}_y",
                f"{name}_z",
                f"{name}_visibility",
            ]
        )
    return header


def make_empty_row(
    person_id: int,
    frame_index: int,
    timestamp_sec: float,
    bbox: Optional[BBox] = None,
    bbox_conf: Optional[float] = None,
) -> Dict[str, float | int | str]:
    """Create an empty CSV row when pose is not detected."""
    row: Dict[str, float | int | str] = {
        "person_id": person_id,
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "pose_detected": 0,
        "bbox_x1": "" if bbox is None else bbox[0],
        "bbox_y1": "" if bbox is None else bbox[1],
        "bbox_x2": "" if bbox is None else bbox[2],
        "bbox_y2": "" if bbox is None else bbox[3],
        "bbox_conf": "" if bbox_conf is None else bbox_conf,
    }

    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = ""
        row[f"{name}_y"] = ""
        row[f"{name}_z"] = ""
        row[f"{name}_visibility"] = ""

    return row


def expand_bbox(
    bbox: BBox,
    frame_width: int,
    frame_height: int,
    margin_ratio: float = CROP_MARGIN_RATIO,
) -> BBox:
    """Expand bbox by a margin while keeping it inside the frame."""
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    margin_x = int(bbox_width * margin_ratio)
    margin_y = int(bbox_height * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(frame_width, x2 + margin_x)
    y2 = min(frame_height, y2 + margin_y)

    return x1, y1, x2, y2


def detect_people_yolo(
    model: YOLO,
    frame_bgr,
    frame_width: int,
    frame_height: int,
    person_conf_threshold: float = PERSON_CONF_THRESHOLD,
    num_persons: int = NUM_PERSONS,
    crop_margin_ratio: float = CROP_MARGIN_RATIO,
) -> List[Tuple[BBox, float]]:
    """Detect people using YOLO and return up to `num_persons` bboxes sorted from left to right."""
    results = model.predict(frame_bgr, conf=person_conf_threshold, verbose=False)
    if not results:
        return []

    detections: List[Tuple[BBox, float]] = []
    boxes = results[0].boxes

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        # COCO class 0 is person.
        if class_id != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width, x2))
        y2 = max(0, min(frame_height, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        bbox = expand_bbox(
            (x1, y1, x2, y2),
            frame_width,
            frame_height,
            margin_ratio=crop_margin_ratio,
        )
        detections.append((bbox, confidence))

    detections = sorted(detections, key=lambda item: (item[0][0] + item[0][2]) / 2)
    return detections[:num_persons]


def pose_result_to_full_frame_row(
    person_id: int,
    frame_index: int,
    timestamp_sec: float,
    detection_result,
    bbox: BBox,
    bbox_conf: float,
    frame_width: int,
    frame_height: int,
) -> Dict[str, float | int | str]:
    """Convert crop-level pose landmarks to full-frame normalized coordinates."""
    row = make_empty_row(person_id, frame_index, timestamp_sec, bbox, bbox_conf)

    if not detection_result.pose_landmarks:
        return row

    x1, y1, x2, y2 = bbox
    crop_width = x2 - x1
    crop_height = y2 - y1
    pose_landmarks = detection_result.pose_landmarks[0]

    row["pose_detected"] = 1
    for name, landmark in zip(LANDMARK_NAMES, pose_landmarks):
        full_x = (x1 + landmark.x * crop_width) / frame_width
        full_y = (y1 + landmark.y * crop_height) / frame_height

        row[f"{name}_x"] = full_x
        row[f"{name}_y"] = full_y
        row[f"{name}_z"] = landmark.z
        row[f"{name}_visibility"] = landmark.visibility

    return row


def draw_pose_from_row(frame_bgr, row: Dict[str, float | int | str], color: Tuple[int, int, int]) -> None:
    """Draw pose landmarks from a CSV row on the original frame."""
    if row["pose_detected"] != 1:
        return

    height, width = frame_bgr.shape[:2]

    points = []
    for name in LANDMARK_NAMES:
        x = row[f"{name}_x"]
        y = row[f"{name}_y"]
        visibility = row[f"{name}_visibility"]

        if x == "" or y == "" or visibility == "" or float(visibility) < 0.5:
            points.append(None)
            continue

        points.append((int(float(x) * width), int(float(y) * height)))

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_point = points[start_idx]
        end_point = points[end_idx]
        if start_point is None or end_point is None:
            continue
        cv2.line(frame_bgr, start_point, end_point, color, 2)

    for point in points:
        if point is None:
            continue
        cv2.circle(frame_bgr, point, 3, color, -1)


def draw_bbox(frame_bgr, person_id: int, bbox: BBox, confidence: float, color: Tuple[int, int, int]) -> None:
    """Draw YOLO person bbox and person_id."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame_bgr,
        f"person_{person_id} conf={confidence:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def create_pose_landmarker_options(model_path: Path) -> vision.PoseLandmarkerOptions:
    """Create MediaPipe Pose Landmarker options."""
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    return vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def run_multi_person_pose_extraction(
    video_path: Path = VIDEO_PATH,
    output_dir: Path = OUTPUT_DIR,
    pose_model_path: Path = POSE_MODEL_PATH,
    yolo_model_path: str = YOLO_MODEL_PATH,
    num_persons: int = NUM_PERSONS,
    person_conf_threshold: float = PERSON_CONF_THRESHOLD,
    crop_margin_ratio: float = CROP_MARGIN_RATIO,
) -> Dict[str, Path]:
    """Run YOLO person detection and pose extraction for up to `num_persons` people.

    This function is designed to be called from a Web app such as Streamlit.
    It also returns paths to generated outputs.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not pose_model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {pose_model_path}\n"
            "Download pose_landmarker_lite.task and place it in the models directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / OUTPUT_VIDEO_FILENAME

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    yolo_model = YOLO(yolo_model_path)
    pose_options = create_pose_landmarker_options(pose_model_path)

    csv_files = {}
    csv_writers = {}
    csv_header = build_csv_header()
    person_csv_paths: Dict[int, Path] = {}

    for person_id in range(num_persons):
        csv_path = output_dir / f"person_{person_id}_pose.csv"
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        csv_writer.writeheader()
        csv_files[person_id] = csv_file
        csv_writers[person_id] = csv_writer
        person_csv_paths[person_id] = csv_path

    colors = {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (0, 255, 255),
    }

    landmarkers = []

    try:
        for _ in range(num_persons):
            landmarkers.append(vision.PoseLandmarker.create_from_options(pose_options))

        frame_index = 0
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            timestamp_sec = frame_index / fps
            timestamp_ms = int(timestamp_sec * 1000)

            people = detect_people_yolo(
                yolo_model,
                frame_bgr,
                frame_width,
                frame_height,
                person_conf_threshold=person_conf_threshold,
                num_persons=num_persons,
                crop_margin_ratio=crop_margin_ratio,
            )
            annotated_frame = frame_bgr.copy()

            for person_id in range(num_persons):
                if person_id >= len(people):
                    row = make_empty_row(person_id, frame_index, timestamp_sec)
                    csv_writers[person_id].writerow(row)
                    continue

                bbox, confidence = people[person_id]
                x1, y1, x2, y2 = bbox
                crop_bgr = frame_bgr[y1:y2, x1:x2]

                if crop_bgr.size == 0:
                    row = make_empty_row(person_id, frame_index, timestamp_sec, bbox, confidence)
                    csv_writers[person_id].writerow(row)
                    continue

                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                detection_result = landmarkers[person_id].detect_for_video(mp_image, timestamp_ms)

                row = pose_result_to_full_frame_row(
                    person_id=person_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    detection_result=detection_result,
                    bbox=bbox,
                    bbox_conf=confidence,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                csv_writers[person_id].writerow(row)

                color = colors.get(person_id, (255, 255, 255))
                draw_bbox(annotated_frame, person_id, bbox, confidence, color)
                draw_pose_from_row(annotated_frame, row, color)

            writer.write(annotated_frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()
        for landmarker in landmarkers:
            landmarker.close()
        for csv_file in csv_files.values():
            csv_file.close()

    outputs: Dict[str, Path] = {
        "annotated_video": output_video_path,
    }
    for person_id, csv_path in person_csv_paths.items():
        outputs[f"person_{person_id}_pose_csv"] = csv_path

    print(f"Saved: {output_video_path}")
    for person_id in range(num_persons):
        print(f"Saved: {person_csv_paths[person_id]}")

    return outputs


def extract_multi_person_pose() -> None:
    """Backward-compatible script wrapper."""
    run_multi_person_pose_extraction()


def main() -> None:
    extract_multi_person_pose()


if __name__ == "__main__":
    main()