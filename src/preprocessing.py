import numpy as np
import pandas as pd

# Landmark indices (MediaPipe)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def compute_center(p1, p2):
    return (p1 + p2) / 2


def compute_torso_vector(shoulder_center, hip_center):
    return shoulder_center - hip_center


def get_rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def normalize_pose(df: pd.DataFrame):
    """
    Input:
        df with columns: [frame, landmark, x, y, z, visibility]

    Output:
        normalized dataframe
    """

    normalized_data = []

    grouped = df.groupby("frame")

    for frame_id, group in grouped:

        group = group.sort_values("landmark")

        coords = group[["x", "y", "z"]].values

        # Skip frames with missing key joints
        try:
            left_shoulder = coords[LEFT_SHOULDER]
            right_shoulder = coords[RIGHT_SHOULDER]
            left_hip = coords[LEFT_HIP]
            right_hip = coords[RIGHT_HIP]
        except:
            continue

        # If NaNs exist → skip frame
        if np.isnan(left_shoulder).any() or np.isnan(right_shoulder).any() \
           or np.isnan(left_hip).any() or np.isnan(right_hip).any():
            continue

        # 1. Translation (hip center → origin)
        hip_center = compute_center(left_hip, right_hip)
        translated = coords - hip_center

        # 2. Scaling (torso length)
        shoulder_center = compute_center(left_shoulder, right_shoulder)
        torso_vec = compute_torso_vector(shoulder_center, hip_center)
        torso_length = np.linalg.norm(torso_vec)

        if torso_length < 1e-6:
            continue

        scaled = translated / torso_length

        # 3. Rotation (align torso vertically)
        torso_vec_scaled = compute_torso_vector(
            compute_center(scaled[LEFT_SHOULDER], scaled[RIGHT_SHOULDER]),
            np.array([0, 0, 0])  # hip center now at origin
        )

        angle = np.arctan2(torso_vec_scaled[1], torso_vec_scaled[0])

        # target: vertical (pointing up → angle = pi/2)
        rotation_angle = (np.pi / 2) - angle

        R = get_rotation_matrix(rotation_angle)

        rotated_xy = np.dot(scaled[:, :2], R.T)

        # 4. Z normalization (scale only)
        z_scaled = scaled[:, 2].reshape(-1, 1)

        final_coords = np.hstack([rotated_xy, z_scaled])

        # Save
        for i, landmark_id in enumerate(group["landmark"].values):
            normalized_data.append({
                "frame": frame_id,
                "landmark": landmark_id,
                "x": final_coords[i, 0],
                "y": final_coords[i, 1],
                "z": final_coords[i, 2],
                "visibility": group.iloc[i]["visibility"]
            })

    normalized_df = pd.DataFrame(normalized_data)

    return normalized_df


def apply_smoothing(df: pd.DataFrame, window=5):
    """
    Apply rolling mean smoothing per landmark
    """
    smoothed = []

    for landmark_id, group in df.groupby("landmark"):
        group = group.sort_values("frame")

        group["x"] = group["x"].rolling(window, center=True, min_periods=1).mean()
        group["y"] = group["y"].rolling(window, center=True, min_periods=1).mean()
        group["z"] = group["z"].rolling(window, center=True, min_periods=1).mean()

        smoothed.append(group)

    return pd.concat(smoothed).reset_index(drop=True)