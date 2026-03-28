import os
import glob
import pandas as pd

from pose import extract_pose
from preprocessing import normalize_pose, apply_smoothing
from features import extract_features
from aggregate import aggregate_features    

def get_label(video_path: str) -> int:
    """
    Assign label based on folder name
    correct → 1
    incorrect → 0
    """
    folder = os.path.basename(os.path.dirname(video_path)).lower()
    
    if folder == "correct":
        return 1
    elif folder == "incorrect":
        return 0
    else:
        raise ValueError(f"Unknown label folder: {folder}")

def process_video(video_path: str) -> dict:
    """
    Full pipeline for a single video:
    pose → normalize → smooth → features → aggregate
    """

    try:
        # 1. Extract pose
        pose_df = extract_pose(video_path)

        if pose_df.empty:
            print(f"[WARN] Empty pose: {video_path}")
            return None

        # 2. Normalize
        norm_df = normalize_pose(pose_df)

        if norm_df.empty:
            print(f"[WARN] Normalization failed: {video_path}")
            return None

        # 3. Smooth
        smooth_df = apply_smoothing(norm_df)

        # 4. Feature extraction
        features_df = extract_features(smooth_df)

        if features_df.empty:
            print(f"[WARN] Feature extraction failed: {video_path}")
            return None

        # 5. Aggregate
        aggregated = aggregate_features(features_df)

        # 6. Add metadata
        aggregated["label"] = get_label(video_path)
        aggregated["video_path"] = video_path

        return aggregated

    except Exception as e:
        print(f"[ERROR] Failed processing {video_path}: {e}")
        return None

def build_dataset(data_dir: str, save_path=None) -> pd.DataFrame:
    """
    Process all videos and build dataset
    """
    

    video_paths = glob.glob(os.path.join(data_dir, "*", "*.mp4"))

    print(f"[INFO] Found {len(video_paths)} videos")

    data = []

    for i, path in enumerate(video_paths):
        print(f"[INFO] Processing {i+1}/{len(video_paths)}: {path}")

        result = process_video(path)

        if result is not None:
            data.append(result)

    df = pd.DataFrame(data)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[INFO] Saved dataset to {save_path}")

    print(f"[INFO] Successfully processed {len(df)} videos")

    return df