import numpy as np
import pandas as pd

def compute_angle(a, b, c):
    """
    Angle at point b (in degrees)
    a, b, c are (x, y, z)
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return np.degrees(angle)


# Landmark indices
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28


def extract_features(df: pd.DataFrame):
    features = []

    grouped = df.groupby("frame")

    for frame_id, group in grouped:
        group = group.sort_values("landmark")

        coords = group[["x", "y", "z"]].values

        # Right side only (simpler, consistent)
        shoulder = coords[RS]
        elbow = coords[RE]
        wrist = coords[RW]
        hip = coords[RH]
        knee = coords[RK]
        ankle = coords[RA]

        # 1. Elbow angle
        elbow_angle = compute_angle(shoulder, elbow, wrist)

        # 2. Shoulder angle
        shoulder_angle = compute_angle(elbow, shoulder, hip)

        # 3. Hip angle
        hip_angle = compute_angle(shoulder, hip, knee)

        # 4. Body alignment (shoulder-hip-ankle)
        alignment = compute_angle(shoulder, hip, ankle)

        # 5. Depth (use shoulder height)
        depth = -(shoulder[1] - wrist[1])  # y after normalization

        features.append({
            "frame": frame_id,
            "elbow_angle": elbow_angle,
            "shoulder_angle": shoulder_angle,
            "hip_angle": hip_angle,
            "alignment": alignment,
            "depth": depth
        })

    return pd.DataFrame(features)