

"""
Create pose snapshot comparison images at a specified timestamp.

Inputs:
    video/test_multi.mov
    outputs/multi/person_0_pose.csv  # target dancer
    outputs/multi/person_1_pose.csv  # learner dancer

Outputs:
    outputs/multi/snapshot/pose_snapshot_<time>.png
    outputs/multi/snapshot/pose_snapshot_<time>_angles.csv

Run:
    python code/pose_snapshot.py
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
OUTPUT_DIR = Path("outputs/multi/snapshot")

SNAPSHOT_TIME_SEC = 10.0
ANGLE_DIFF_THRESHOLD_DEG = 20.0
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


Frame = np.ndarray


def load_pose_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column != "pose_detected":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["pose_detected"] = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0).astype(int)
    return df


def get_video_info(video_path: Path) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        fps = 30.0

    return fps, frame_count, frame_width, frame_height


def read_frame_at_time(video_path: Path, time_sec: float) -> Tuple[Frame, int, float]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    fps, frame_count, _, _ = get_video_info(video_path)
    frame_index = int(round(time_sec * fps))
    frame_index = max(0, min(frame_index, max(frame_count - 1, 0)))
    actual_time_sec = frame_index / fps

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame_bgr = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Failed to read frame at {time_sec:.2f}s")

    return frame_bgr, frame_index, actual_time_sec


def get_nearest_row(df: pd.DataFrame, frame_index: int) -> pd.Series | None:
    if df.empty:
        return None

    exact_rows = df[df["frame_index"] == frame_index]
    if not exact_rows.empty:
        return exact_rows.iloc[0]

    nearest_index = (df["frame_index"] - frame_index).abs().idxmin()
    return df.loc[nearest_index]


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


def compare_angles(
    target_row: pd.Series,
    learner_row: pd.Series,
    frame_width: int,
    frame_height: int,
    angle_diff_threshold_deg: float,
) -> pd.DataFrame:
    rows = []

    for angle_name, (_, vertex_joint, _) in ANGLE_DEFINITIONS.items():
        target_angle = calculate_angle(target_row, angle_name, frame_width, frame_height)
        learner_angle = calculate_angle(learner_row, angle_name, frame_width, frame_height)

        if np.isnan(target_angle) or np.isnan(learner_angle):
            continue

        signed_diff = learner_angle - target_angle
        abs_diff = abs(signed_diff)
        is_large = abs_diff >= angle_diff_threshold_deg

        rows.append(
            {
                "angle_name": angle_name,
                "angle_label": ANGLE_LABELS.get(angle_name, angle_name),
                "vertex_joint": vertex_joint,
                "target_angle_deg": target_angle,
                "learner_angle_deg": learner_angle,
                "signed_diff_deg": signed_diff,
                "abs_diff_deg": abs_diff,
                "is_large_difference": is_large,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("abs_diff_deg", ascending=False).reset_index(drop=True)


def draw_text_with_background(
    frame: Frame,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.55,
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


def draw_pose(row: pd.Series, frame: Frame, color: Tuple[int, int, int], thickness: int = 2) -> None:
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
        cv2.line(frame, tuple(start_point), tuple(end_point), color, thickness)

    for point in points:
        if point is None:
            continue
        cv2.circle(frame, tuple(point), 4, color, -1)


def draw_angle_highlights(
    learner_row: pd.Series,
    frame: Frame,
    angle_df: pd.DataFrame,
    max_highlights: int = 5,
) -> None:
    if angle_df.empty:
        return

    frame_height, frame_width = frame.shape[:2]
    highlight_df = angle_df[angle_df["is_large_difference"]].head(max_highlights)

    for _, row in highlight_df.iterrows():
        vertex_joint = row["vertex_joint"]
        point = get_joint_xy(learner_row, vertex_joint, frame_width, frame_height)
        if point is None:
            continue

        cv2.circle(frame, tuple(point), 22, HIGHLIGHT_COLOR, 4)
        cv2.circle(frame, tuple(point), 7, HIGHLIGHT_COLOR, -1)

        label = f"{row['angle_label']} {row['signed_diff_deg']:+.0f}deg"
        text_x = min(point[0] + 24, frame_width - 240)
        text_y = max(point[1] - 12, 30)
        draw_text_with_background(frame, label, (text_x, text_y), font_scale=0.55, thickness=2)


# ---------------------- Normalized skeleton overlay helpers ----------------------

def get_normalized_joint_coordinates(
    row: pd.Series,
    joint_names: List[str],
) -> Dict[str, np.ndarray]:
    """Normalize pose by hip center and shoulder width.

    This removes the influence of dancer position and body scale so that form differences
    are easier to compare.
    """
    if row is None or row.get("pose_detected", 0) != 1:
        return {}

    def get_raw_point(joint_name: str) -> np.ndarray | None:
        x = row.get(f"{joint_name}_x")
        y = row.get(f"{joint_name}_y")
        visibility = row.get(f"{joint_name}_visibility", 1.0)
        if pd.isna(x) or pd.isna(y) or pd.isna(visibility):
            return None
        if float(visibility) < VISIBILITY_THRESHOLD:
            return None
        return np.array([float(x), float(y)], dtype=float)

    left_hip = get_raw_point("left_hip")
    right_hip = get_raw_point("right_hip")
    left_shoulder = get_raw_point("left_shoulder")
    right_shoulder = get_raw_point("right_shoulder")

    if left_hip is None or right_hip is None or left_shoulder is None or right_shoulder is None:
        return {}

    hip_center = (left_hip + right_hip) / 2
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width <= 1e-6:
        return {}

    normalized_points: Dict[str, np.ndarray] = {}
    for joint_name in joint_names:
        point = get_raw_point(joint_name)
        if point is None:
            continue
        normalized_points[joint_name] = (point - hip_center) / shoulder_width

    return normalized_points


def normalized_to_canvas_point(
    normalized_point: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    scale: float,
) -> Tuple[int, int]:
    center = np.array([canvas_width / 2, canvas_height * 0.62], dtype=float)
    point = center + normalized_point * scale
    return int(point[0]), int(point[1])


def draw_normalized_skeleton(
    canvas: Frame,
    normalized_points: Dict[str, np.ndarray],
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    scale = min(canvas_width, canvas_height) * 0.36

    canvas_points: Dict[str, Tuple[int, int]] = {}
    for joint_name, normalized_point in normalized_points.items():
        canvas_points[joint_name] = normalized_to_canvas_point(
            normalized_point,
            canvas_width,
            canvas_height,
            scale,
        )

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_name = LANDMARK_NAMES[start_idx]
        end_name = LANDMARK_NAMES[end_idx]
        if start_name not in canvas_points or end_name not in canvas_points:
            continue
        cv2.line(canvas, canvas_points[start_name], canvas_points[end_name], color, thickness)

    for point in canvas_points.values():
        cv2.circle(canvas, point, 5, color, -1)


def draw_normalized_angle_highlights(
    canvas: Frame,
    learner_points: Dict[str, np.ndarray],
    angle_df: pd.DataFrame,
    max_highlights: int = 5,
) -> None:
    if angle_df.empty or not learner_points:
        return

    canvas_height, canvas_width = canvas.shape[:2]
    scale = min(canvas_width, canvas_height) * 0.36
    highlight_df = angle_df[angle_df["is_large_difference"]].head(max_highlights)

    for _, row in highlight_df.iterrows():
        vertex_joint = row["vertex_joint"]
        if vertex_joint not in learner_points:
            continue

        point = normalized_to_canvas_point(
            learner_points[vertex_joint],
            canvas_width,
            canvas_height,
            scale,
        )
        cv2.circle(canvas, point, 24, HIGHLIGHT_COLOR, 4)
        cv2.circle(canvas, point, 7, HIGHLIGHT_COLOR, -1)

        label = f"{row['angle_label']} {row['signed_diff_deg']:+.0f}deg"
        text_x = min(point[0] + 24, canvas_width - 250)
        text_y = max(point[1] - 12, 35)
        draw_text_with_background(canvas, label, (text_x, text_y), font_scale=0.55, thickness=2)


def create_normalized_overlay_panel(
    target_row: pd.Series,
    learner_row: pd.Series,
    angle_df: pd.DataFrame,
    panel_width: int,
    panel_height: int,
) -> Frame:
    """Create a clean panel showing only normalized skeletons overlaid."""
    canvas = np.full((panel_height, panel_width, 3), 245, dtype=np.uint8)

    target_points = get_normalized_joint_coordinates(target_row, LANDMARK_NAMES)
    learner_points = get_normalized_joint_coordinates(learner_row, LANDMARK_NAMES)

    draw_text_with_background(
        canvas,
        "normalized skeleton overlay",
        (20, 35),
        font_scale=0.7,
        thickness=2,
    )
    draw_text_with_background(
        canvas,
        "target=green learner=red highlight=yellow",
        (20, 68),
        font_scale=0.55,
        thickness=2,
    )

    # Draw target first and learner second so the learner difference is easy to see.
    draw_normalized_skeleton(canvas, target_points, TARGET_COLOR, thickness=3)
    draw_normalized_skeleton(canvas, learner_points, LEARNER_COLOR, thickness=3)
    draw_normalized_angle_highlights(canvas, learner_points, angle_df)

    # Draw hip-center reference point.
    center_point = (panel_width // 2, int(panel_height * 0.62))
    cv2.circle(canvas, center_point, 6, (80, 80, 80), -1)
    draw_text_with_background(canvas, "hip center", (center_point[0] + 12, center_point[1] + 5), font_scale=0.45, thickness=1)

    return canvas


def create_side_by_side_snapshot(
    frame_bgr: Frame,
    target_row: pd.Series,
    learner_row: pd.Series,
    angle_df: pd.DataFrame,
    time_sec: float,
    frame_index: int,
) -> Frame:
    """Create a snapshot with video comparison and normalized skeleton overlay.

    Left: original frame with both skeletons overlaid.
    Right: normalized skeleton-only overlay for easier form comparison.
    Bottom: top angle differences.
    """
    comparison_frame = frame_bgr.copy()
    draw_pose(target_row, comparison_frame, TARGET_COLOR, thickness=2)
    draw_pose(learner_row, comparison_frame, LEARNER_COLOR, thickness=2)
    draw_angle_highlights(learner_row, comparison_frame, angle_df)

    draw_text_with_background(
        comparison_frame,
        "video overlay: target=green learner=red",
        (20, 35),
        font_scale=0.65,
        thickness=2,
    )
    draw_text_with_background(
        comparison_frame,
        f"time={time_sec:.2f}s frame={frame_index}",
        (20, 68),
        font_scale=0.6,
        thickness=2,
    )

    frame_height, frame_width = frame_bgr.shape[:2]
    normalized_overlay = create_normalized_overlay_panel(
        target_row=target_row,
        learner_row=learner_row,
        angle_df=angle_df,
        panel_width=frame_width,
        panel_height=frame_height,
    )

    snapshot = np.hstack([comparison_frame, normalized_overlay])

    top_diffs = angle_df.head(5)
    panel_height = max(150, 45 + len(top_diffs) * 28)
    panel = np.zeros((panel_height, snapshot.shape[1], 3), dtype=np.uint8)

    draw_text_with_background(
        panel,
        "Pose Snapshot: angle differences",
        (20, 35),
        font_scale=0.75,
        thickness=2,
    )

    if top_diffs.empty:
        draw_text_with_background(panel, "No valid angle comparison.", (20, 75), font_scale=0.6, thickness=2)
    else:
        y = 75
        for _, row in top_diffs.iterrows():
            label = (
                f"{row['angle_label']}: target={row['target_angle_deg']:.1f}deg, "
                f"learner={row['learner_angle_deg']:.1f}deg, diff={row['signed_diff_deg']:+.1f}deg"
            )
            draw_text_with_background(panel, label, (20, y), font_scale=0.55, thickness=1)
            y += 28

    output_image = np.vstack([snapshot, panel])
    return output_image


def run_pose_snapshot(
    video_path: Path = VIDEO_PATH,
    target_csv_path: Path = TARGET_CSV_PATH,
    learner_csv_path: Path = LEARNER_CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
    time_sec: float = SNAPSHOT_TIME_SEC,
    angle_diff_threshold_deg: float = ANGLE_DIFF_THRESHOLD_DEG,
    decimals: int = 3,
) -> Dict[str, Path]:
    """Create a side-by-side pose snapshot and angle comparison CSV.

    Designed to be called from a Web app such as Streamlit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_bgr, frame_index, actual_time_sec = read_frame_at_time(video_path, time_sec)
    frame_height, frame_width = frame_bgr.shape[:2]

    target_df = load_pose_csv(target_csv_path)
    learner_df = load_pose_csv(learner_csv_path)
    target_row = get_nearest_row(target_df, frame_index)
    learner_row = get_nearest_row(learner_df, frame_index)

    angle_df = compare_angles(
        target_row=target_row,
        learner_row=learner_row,
        frame_width=frame_width,
        frame_height=frame_height,
        angle_diff_threshold_deg=angle_diff_threshold_deg,
    )

    safe_time = f"{actual_time_sec:.2f}".replace(".", "_")
    snapshot_path = output_dir / f"pose_snapshot_{safe_time}s.png"
    angles_csv_path = output_dir / f"pose_snapshot_{safe_time}s_angles.csv"

    snapshot_image = create_side_by_side_snapshot(
        frame_bgr=frame_bgr,
        target_row=target_row,
        learner_row=learner_row,
        angle_df=angle_df,
        time_sec=actual_time_sec,
        frame_index=frame_index,
    )

    cv2.imwrite(str(snapshot_path), snapshot_image)
    angle_df.round(decimals).to_csv(angles_csv_path, index=False)

    outputs: Dict[str, Path] = {
        "pose_snapshot_image": snapshot_path,
        "pose_snapshot_angles_csv": angles_csv_path,
    }

    for output_path in outputs.values():
        print(f"Saved: {output_path}")

    return outputs


def main() -> None:
    run_pose_snapshot(
        video_path=VIDEO_PATH,
        target_csv_path=TARGET_CSV_PATH,
        learner_csv_path=LEARNER_CSV_PATH,
        output_dir=OUTPUT_DIR,
        time_sec=SNAPSHOT_TIME_SEC,
        angle_diff_threshold_deg=ANGLE_DIFF_THRESHOLD_DEG,
    )


if __name__ == "__main__":
    main()