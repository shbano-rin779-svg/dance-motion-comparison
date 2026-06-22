"""
Create an annotated comparison video visualizing trajectory and dynamics differences.

Inputs:
    data/input/test.mov
    outputs/multi/person_0_pose.csv  # target dancer
    outputs/multi/person_1_pose.csv  # learner dancer

Output:
    outputs/multi/overlay/trajectory_dynamics_overlay.mp4

Run:
    python code/trajectory_dynamics_overlay.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


VIDEO_PATH = Path("video/test_multi.mov")
TARGET_CSV_PATH = Path("outputs/multi/person_0_pose.csv")
LEARNER_CSV_PATH = Path("outputs/multi/person_1_pose.csv")
OUTPUT_DIR = Path("outputs/multi/overlay")
OUTPUT_VIDEO_FILENAME = "trajectory_dynamics_overlay.mp4"

VISIBILITY_THRESHOLD = 0.5
TRAIL_SEC = 1.0
DYNAMICS_DIFF_THRESHOLD_RATIO = 0.35
TRAJECTORY_DIFF_THRESHOLD_PX = 80.0

TARGET_COLOR = (0, 255, 0)
LEARNER_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)

TRACK_JOINTS = [
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
]

JOINT_LABELS = {
    "left_wrist": "L wrist",
    "right_wrist": "R wrist",
    "left_ankle": "L ankle",
    "right_ankle": "R ankle",
}

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


def calculate_speed(
    history: Deque[Tuple[float, np.ndarray]],
) -> float | None:
    if len(history) < 2:
        return None

    t0, p0 = history[-2]
    t1, p1 = history[-1]
    dt = t1 - t0
    if dt <= 0:
        return None

    return float(np.linalg.norm(p1.astype(float) - p0.astype(float)) / dt)


def calculate_trail_distance(history: Deque[Tuple[float, np.ndarray]]) -> float:
    if len(history) < 2:
        return 0.0

    distance = 0.0
    for idx in range(1, len(history)):
        _, previous_point = history[idx - 1]
        _, current_point = history[idx]
        distance += float(np.linalg.norm(current_point.astype(float) - previous_point.astype(float)))
    return distance


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
        cv2.line(frame, tuple(start_point), tuple(end_point), color, 1)

    for point in points:
        if point is None:
            continue
        cv2.circle(frame, tuple(point), 2, color, -1)


def draw_trail(
    frame,
    history: Deque[Tuple[float, np.ndarray]],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if len(history) < 2:
        return

    points = [point for _, point in history]
    for idx in range(1, len(points)):
        alpha = idx / max(len(points) - 1, 1)
        line_thickness = max(1, int(thickness * alpha))
        cv2.line(frame, tuple(points[idx - 1]), tuple(points[idx]), color, line_thickness)


def update_histories(
    histories: Dict[str, Deque[Tuple[float, np.ndarray]]],
    row: pd.Series,
    timestamp_sec: float,
    frame_width: int,
    frame_height: int,
    trail_sec: float = TRAIL_SEC,
) -> None:
    for joint_name in TRACK_JOINTS:
        point = get_joint_xy(row, joint_name, frame_width, frame_height)
        if point is None:
            continue

        history = histories[joint_name]
        history.append((timestamp_sec, point))

        while history and timestamp_sec - history[0][0] > trail_sec:
            history.popleft()


def analyze_joint_differences(
    target_histories: Dict[str, Deque[Tuple[float, np.ndarray]]],
    learner_histories: Dict[str, Deque[Tuple[float, np.ndarray]]],
    dynamics_diff_threshold_ratio: float = DYNAMICS_DIFF_THRESHOLD_RATIO,
    trajectory_diff_threshold_px: float = TRAJECTORY_DIFF_THRESHOLD_PX,
) -> List[Dict[str, object]]:
    events = []

    for joint_name in TRACK_JOINTS:
        target_history = target_histories[joint_name]
        learner_history = learner_histories[joint_name]

        if not target_history or not learner_history:
            continue

        target_current = target_history[-1][1]
        learner_current = learner_history[-1][1]
        trajectory_diff = float(np.linalg.norm(learner_current.astype(float) - target_current.astype(float)))

        target_speed = calculate_speed(target_history)
        learner_speed = calculate_speed(learner_history)
        if target_speed is None or learner_speed is None:
            speed_diff_ratio = 0.0
        else:
            denominator = max(abs(target_speed), 1e-6)
            speed_diff_ratio = (learner_speed - target_speed) / denominator

        target_trail_distance = calculate_trail_distance(target_history)
        learner_trail_distance = calculate_trail_distance(learner_history)
        trail_distance_diff = learner_trail_distance - target_trail_distance

        if (
            trajectory_diff >= trajectory_diff_threshold_px
            or abs(speed_diff_ratio) >= dynamics_diff_threshold_ratio
        ):
            events.append(
                {
                    "joint_name": joint_name,
                    "label": JOINT_LABELS.get(joint_name, joint_name),
                    "trajectory_diff": trajectory_diff,
                    "target_speed": target_speed,
                    "learner_speed": learner_speed,
                    "speed_diff_ratio": speed_diff_ratio,
                    "target_trail_distance": target_trail_distance,
                    "learner_trail_distance": learner_trail_distance,
                    "trail_distance_diff": trail_distance_diff,
                    "learner_point": learner_current,
                }
            )

    events = sorted(
        events,
        key=lambda event: max(
            event["trajectory_diff"] / max(trajectory_diff_threshold_px, 1e-6),
            abs(event["speed_diff_ratio"]) / max(dynamics_diff_threshold_ratio, 1e-6),
        ),
        reverse=True,
    )
    return events


def draw_joint_difference_highlights(frame, events: List[Dict[str, object]]) -> None:
    for event in events[:4]:
        point = event["learner_point"]
        if point is None:
            continue

        cv2.circle(frame, tuple(point), 18, HIGHLIGHT_COLOR, 3)

        label = (
            f"{event['label']} traj={event['trajectory_diff']:.0f}px "
            f"spd={event['speed_diff_ratio'] * 100:+.0f}%"
        )
        text_x = min(point[0] + 20, frame.shape[1] - 360)
        text_y = max(point[1] - 10, 30)
        draw_text_with_background(frame, label, (text_x, text_y), font_scale=0.52, thickness=2)


def draw_summary_panel(
    frame,
    frame_index: int,
    timestamp_sec: float,
    events: List[Dict[str, object]],
    dynamics_diff_threshold_ratio: float = DYNAMICS_DIFF_THRESHOLD_RATIO,
) -> None:
    draw_text_with_background(
        frame,
        f"time={timestamp_sec:.2f}s frame={frame_index}",
        (20, 30),
        font_scale=0.65,
        thickness=2,
    )
    draw_text_with_background(
        frame,
        "target trail=green learner trail=red highlight=yellow",
        (20, 60),
        font_scale=0.55,
        thickness=2,
    )

    if not events:
        draw_text_with_background(
            frame,
            "trajectory/dynamics difference: small",
            (20, 90),
            font_scale=0.55,
            thickness=2,
        )
        return

    top_event = events[0]
    if abs(top_event["speed_diff_ratio"]) >= dynamics_diff_threshold_ratio:
        if top_event["speed_diff_ratio"] > 0:
            comment = f"{top_event['label']}: learner faster than target"
        else:
            comment = f"{top_event['label']}: learner slower than target"
    else:
        comment = f"{top_event['label']}: trajectory is different"

    draw_text_with_background(
        frame,
        comment,
        (20, 90),
        font_scale=0.55,
        thickness=2,
    )


def create_trajectory_dynamics_overlay_video(
    video_path: Path = VIDEO_PATH,
    target_csv_path: Path = TARGET_CSV_PATH,
    learner_csv_path: Path = LEARNER_CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
    trail_sec: float = TRAIL_SEC,
    dynamics_diff_threshold_ratio: float = DYNAMICS_DIFF_THRESHOLD_RATIO,
    trajectory_diff_threshold_px: float = TRAJECTORY_DIFF_THRESHOLD_PX,
) -> Dict[str, Path]:
    """Create an annotated comparison video visualizing trajectory and dynamics differences.

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

    target_histories: Dict[str, Deque[Tuple[float, np.ndarray]]] = {
        joint_name: deque() for joint_name in TRACK_JOINTS
    }
    learner_histories: Dict[str, Deque[Tuple[float, np.ndarray]]] = {
        joint_name: deque() for joint_name in TRACK_JOINTS
    }

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

        update_histories(
            target_histories,
            target_row,
            timestamp_sec,
            frame_width,
            frame_height,
            trail_sec=trail_sec,
        )
        update_histories(
            learner_histories,
            learner_row,
            timestamp_sec,
            frame_width,
            frame_height,
            trail_sec=trail_sec,
        )

        for joint_name in TRACK_JOINTS:
            draw_trail(annotated_frame, target_histories[joint_name], TARGET_COLOR, thickness=3)
            draw_trail(annotated_frame, learner_histories[joint_name], LEARNER_COLOR, thickness=3)

        events = analyze_joint_differences(
            target_histories,
            learner_histories,
            dynamics_diff_threshold_ratio=dynamics_diff_threshold_ratio,
            trajectory_diff_threshold_px=trajectory_diff_threshold_px,
        )
        draw_joint_difference_highlights(annotated_frame, events)
        draw_summary_panel(
            annotated_frame,
            frame_index,
            timestamp_sec,
            events,
            dynamics_diff_threshold_ratio=dynamics_diff_threshold_ratio,
        )

        writer.write(annotated_frame)
        frame_index += 1

    cap.release()
    writer.release()

    outputs: Dict[str, Path] = {
        "trajectory_dynamics_overlay_video": output_video_path,
    }

    print(f"Saved: {output_video_path}")
    return outputs


def main() -> None:
    create_trajectory_dynamics_overlay_video()


if __name__ == "__main__":
    main()