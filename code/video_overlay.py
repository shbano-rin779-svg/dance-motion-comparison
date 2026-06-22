"""
Create an annotated comparison video highlighting pose-angle differences.

Inputs:
    data/input/test.mov
    outputs/multi/person_0_pose.csv  # target dancer
    outputs/multi/person_1_pose.csv  # learner dancer

Output:
    outputs/multi/overlay/angle_difference_overlay.mp4

Run:
    python code/video_overlay.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


VIDEO_PATH = Path("video/test_multi.mov")
TARGET_CSV_PATH = Path("outputs/multi/person_0_pose.csv")
LEARNER_CSV_PATH = Path("outputs/multi/person_1_pose.csv")
OUTPUT_DIR = Path("outputs/multi/overlay")
OUTPUT_VIDEO_FILENAME = "angle_difference_overlay.mp4"

ANGLE_DIFF_THRESHOLD_DEG = 25.0
VISIBILITY_THRESHOLD = 0.5

TARGET_COLOR = (0, 255, 0)
LEARNER_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)

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

# angle_name: (point_a, vertex_point, point_c)
ANGLE_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
}

ANGLE_LABELS = {
    "left_elbow_angle": "L elbow",
    "right_elbow_angle": "R elbow",
    "left_knee_angle": "L knee",
    "right_knee_angle": "R knee",
    "left_shoulder_angle": "L shoulder",
    "right_shoulder_angle": "R shoulder",
    "left_hip_angle": "L hip",
    "right_hip_angle": "R hip",
}


def load_pose_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column != "pose_detected":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["pose_detected"] = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0).astype(int)
    return df


def get_row_by_frame(df: pd.DataFrame, frame_index: int) -> pd.Series | None:
    rows = df[df["frame_index"] == frame_index]
    if rows.empty:
        return None
    return rows.iloc[0]


def get_joint_xy(row: pd.Series, joint_name: str, frame_width: int, frame_height: int) -> np.ndarray | None:
    if row is None or row.get("pose_detected", 0) != 1:
        return None

    x = row.get(f"{joint_name}_x")
    y = row.get(f"{joint_name}_y")
    visibility = row.get(f"{joint_name}_visibility", 1.0)

    if pd.isna(x) or pd.isna(y) or pd.isna(visibility):
        return None
    if float(visibility) < VISIBILITY_THRESHOLD:
        return None

    return np.array([int(float(x) * frame_width), int(float(y) * frame_height)], dtype=int)


def calculate_three_point_angle(point_a: np.ndarray, vertex: np.ndarray, point_c: np.ndarray) -> float:
    vector_a = point_a.astype(float) - vertex.astype(float)
    vector_c = point_c.astype(float) - vertex.astype(float)

    norm_a = np.linalg.norm(vector_a)
    norm_c = np.linalg.norm(vector_c)
    if norm_a == 0 or norm_c == 0:
        return np.nan

    cos_angle = np.dot(vector_a, vector_c) / (norm_a * norm_c)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_angle(row: pd.Series, angle_name: str, frame_width: int, frame_height: int) -> float:
    joint_a, joint_b, joint_c = ANGLE_DEFINITIONS[angle_name]
    point_a = get_joint_xy(row, joint_a, frame_width, frame_height)
    point_b = get_joint_xy(row, joint_b, frame_width, frame_height)
    point_c = get_joint_xy(row, joint_c, frame_width, frame_height)

    if point_a is None or point_b is None or point_c is None:
        return np.nan

    return calculate_three_point_angle(point_a, point_b, point_c)


def draw_text_with_background(
    frame,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (x - 4, y - text_height - baseline - 4),
        (x + text_width + 4, y + baseline + 4),
        TEXT_BG_COLOR,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def draw_pose(row: pd.Series, frame, color: Tuple[int, int, int]) -> None:
    if row is None or row.get("pose_detected", 0) != 1:
        return

    frame_height, frame_width = frame.shape[:2]

    points = []
    for joint_name in LANDMARK_NAMES:
        point = get_joint_xy(row, joint_name, frame_width, frame_height)
        points.append(point)

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_point = points[start_idx]
        end_point = points[end_idx]
        if start_point is None or end_point is None:
            continue
        cv2.line(frame, tuple(start_point), tuple(end_point), color, 2)

    for point in points:
        if point is None:
            continue
        cv2.circle(frame, tuple(point), 3, color, -1)


def get_angle_difference_events(
    target_row: pd.Series,
    learner_row: pd.Series,
    frame_width: int,
    frame_height: int,
    angle_diff_threshold_deg: float = ANGLE_DIFF_THRESHOLD_DEG,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []

    if target_row is None or learner_row is None:
        return events
    if target_row.get("pose_detected", 0) != 1 or learner_row.get("pose_detected", 0) != 1:
        return events

    for angle_name in ANGLE_DEFINITIONS:
        target_angle = calculate_angle(target_row, angle_name, frame_width, frame_height)
        learner_angle = calculate_angle(learner_row, angle_name, frame_width, frame_height)

        if np.isnan(target_angle) or np.isnan(learner_angle):
            continue

        diff = abs(learner_angle - target_angle)
        if diff < angle_diff_threshold_deg:
            continue

        vertex_joint = ANGLE_DEFINITIONS[angle_name][1]
        learner_vertex = get_joint_xy(learner_row, vertex_joint, frame_width, frame_height)
        target_vertex = get_joint_xy(target_row, vertex_joint, frame_width, frame_height)

        events.append(
            {
                "angle_name": angle_name,
                "label": ANGLE_LABELS.get(angle_name, angle_name),
                "target_angle": target_angle,
                "learner_angle": learner_angle,
                "diff": diff,
                "learner_vertex": learner_vertex,
                "target_vertex": target_vertex,
            }
        )

    events = sorted(events, key=lambda event: event["diff"], reverse=True)
    return events


def draw_angle_highlights(frame, events: List[Dict[str, object]]) -> None:
    if not events:
        return

    # Draw top 3 largest angle differences to avoid clutter.
    for event in events[:3]:
        learner_vertex = event["learner_vertex"]
        if learner_vertex is None:
            continue

        point = tuple(learner_vertex)
        cv2.circle(frame, point, 18, HIGHLIGHT_COLOR, 3)
        cv2.circle(frame, point, 6, HIGHLIGHT_COLOR, -1)

        label = f"{event['label']} diff={event['diff']:.0f}deg"
        text_x = min(point[0] + 20, frame.shape[1] - 260)
        text_y = max(point[1] - 10, 30)
        draw_text_with_background(frame, label, (text_x, text_y), font_scale=0.55, thickness=2)


def draw_frame_info(frame, frame_index: int, timestamp_sec: float, events: List[Dict[str, object]]) -> None:
    draw_text_with_background(
        frame,
        f"time={timestamp_sec:.2f}s frame={frame_index}",
        (20, 30),
        font_scale=0.65,
        thickness=2,
    )
    draw_text_with_background(
        frame,
        "target=green learner=red highlight=yellow",
        (20, 60),
        font_scale=0.55,
        thickness=2,
    )

    if events:
        top_event = events[0]
        draw_text_with_background(
            frame,
            f"largest: {top_event['label']} {top_event['diff']:.1f} deg",
            (20, 90),
            font_scale=0.55,
            thickness=2,
        )


def create_angle_overlay_video(
    video_path: Path = VIDEO_PATH,
    target_csv_path: Path = TARGET_CSV_PATH,
    learner_csv_path: Path = LEARNER_CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
    angle_diff_threshold_deg: float = ANGLE_DIFF_THRESHOLD_DEG,
) -> Dict[str, Path]:
    """Create an annotated comparison video highlighting pose-angle differences.

    Designed to be called from a Web app such as Streamlit.
    Returns paths to generated outputs.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / OUTPUT_VIDEO_FILENAME

    target_df = load_pose_csv(target_csv_path)
    learner_df = load_pose_csv(learner_csv_path)

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

    frame_index = 0
    while True:
        success, frame_bgr = cap.read()
        if not success:
            break

        timestamp_sec = frame_index / fps
        annotated_frame = frame_bgr.copy()

        target_row = get_row_by_frame(target_df, frame_index)
        learner_row = get_row_by_frame(learner_df, frame_index)

        draw_pose(target_row, annotated_frame, TARGET_COLOR)
        draw_pose(learner_row, annotated_frame, LEARNER_COLOR)

        events = get_angle_difference_events(
            target_row,
            learner_row,
            frame_width,
            frame_height,
            angle_diff_threshold_deg=angle_diff_threshold_deg,
        )
        draw_angle_highlights(annotated_frame, events)
        draw_frame_info(annotated_frame, frame_index, timestamp_sec, events)

        writer.write(annotated_frame)
        frame_index += 1

    cap.release()
    writer.release()

    outputs: Dict[str, Path] = {
        "angle_overlay_video": output_video_path,
    }

    print(f"Saved: {output_video_path}")
    return outputs


def create_overlay_video() -> None:
    """Backward-compatible script wrapper."""
    create_angle_overlay_video()


def main() -> None:
    create_overlay_video()


if __name__ == "__main__":
    main()