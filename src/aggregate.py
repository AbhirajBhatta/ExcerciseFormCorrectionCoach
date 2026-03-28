def aggregate_features(features_df):
    return {
        "min_elbow_angle": features_df["elbow_angle"].min(),
        "max_elbow_angle": features_df["elbow_angle"].max(),

        "max_depth": features_df["depth"].max(),
        "min_depth": features_df["depth"].min(),

        "avg_alignment": features_df["alignment"].mean(),
        "min_alignment": features_df["alignment"].min(),

        "min_hip_angle": features_df["hip_angle"].min(),
        "avg_hip_angle": features_df["hip_angle"].mean(),

        # Optional: dynamics
        "elbow_range": features_df["elbow_angle"].max() - features_df["elbow_angle"].min(),
        "depth_range": features_df["depth"].max() - features_df["depth"].min(),
    }