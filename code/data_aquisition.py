"""
Extract pose landmark time-series from a dance video using MediaPipe Pose.

Before running, place the MediaPipe model file here:
    models/pose_landmarker_lite.task

Run:
    python code/data_aquisition.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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


def build_csv_header() -> List[str]:
    """Create CSV columns for frame metadata and all pose landmarks."""
    header = ["frame_index", "timestamp_sec", "pose_detected"]
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


def landmarks_to_row(frame_index: int, timestamp_sec: float, detection_result) -> Dict[str, float | int | str]:
    """Convert MediaPipe Tasks pose landmarks for one frame into a CSV row."""
    row: Dict[str, float | int | str] = {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "pose_detected": 0,
    }

    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = ""
        row[f"{name}_y"] = ""
        row[f"{name}_z"] = ""
        row[f"{name}_visibility"] = ""

    if not detection_result.pose_landmarks:
        return row

    # This script currently uses the first detected person.
    # For solo dance videos, this is usually sufficient.
    pose_landmarks = detection_result.pose_landmarks[0]
    row["pose_detected"] = 1

    for name, landmark in zip(LANDMARK_NAMES, pose_landmarks):
        row[f"{name}_x"] = landmark.x
        row[f"{name}_y"] = landmark.y
        row[f"{name}_z"] = landmark.z
        row[f"{name}_visibility"] = landmark.visibility

    return row


def draw_pose_landmarks(frame_bgr, detection_result) -> None:
    """Draw the first detected person's pose landmarks and connections in place."""
    if not detection_result.pose_landmarks:
        return

    height, width = frame_bgr.shape[:2]
    pose_landmarks = detection_result.pose_landmarks[0]

    for start_idx, end_idx in POSE_CONNECTIONS:
        start = pose_landmarks[start_idx]
        end = pose_landmarks[end_idx]

        if start.visibility < 0.5 or end.visibility < 0.5:
            continue

        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame_bgr, start_point, end_point, (0, 255, 0), 2)

    for landmark in pose_landmarks:
        if landmark.visibility < 0.5:
            continue

        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame_bgr, center, 3, (0, 0, 255), -1)


def extract_pose_timeseries(video_path: Path, output_csv_path: Path, output_video_path: Path) -> None:
    """Extract pose landmarks from a video and save CSV plus annotated video."""
    model_path = Path("models/pose_landmarker_lite.task")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download pose_landmarker_lite.task and place it in the models directory."
        )

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    header = build_csv_header()

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=header)
        csv_writer.writeheader()

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0

            while True:
                success, frame_bgr = cap.read()
                if not success:
                    break

                timestamp_sec = frame_index / fps

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(timestamp_sec * 1000)
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                row = landmarks_to_row(frame_index, timestamp_sec, detection_result)
                csv_writer.writerow(row)

                annotated_frame = frame_bgr.copy()
                draw_pose_landmarks(annotated_frame, detection_result)
                writer.write(annotated_frame)
                frame_index += 1

    cap.release()
    writer.release()

    print(f"Saved pose CSV: {output_csv_path}")
    print(f"Saved annotated video: {output_video_path}")


def main() -> None:
    video_path = Path("video/test.mov")
    output_csv_path = Path("outputs/test_pose.csv")
    output_video_path = Path("outputs/test_annotated.mp4")

    extract_pose_timeseries(
        video_path,
        output_csv_path,
        output_video_path,
    )


if __name__ == "__main__":
    main()