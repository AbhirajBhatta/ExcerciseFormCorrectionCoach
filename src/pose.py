import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_pose(video_path):
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False, # for video processing
        model_complexity=1, # to reduce computation, higher means more accuracy
        enable_segmentation=False, # whether we want to separate the shilloute of the person
        min_detection_confidence=0.5, #the threshold which decides whether mediapipe classifies the frame as having a human or not
        min_tracking_confidence=0.5 # confidence for pose tracking, if the confidence is below thresh, mp runs the pose detection again
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Check the path video not loaded")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data = []
    frame_id = 0

    print("Processing video...")
    with tqdm(total=frame_count) as pbar:

        while True:
            ret, frame = cap.read() # ret is true if frame exists, helps determine when vid ends

            if not ret:
                break

            # Convert BGR → RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(image) # run the nueral network to identify the pose

            if results.pose_landmarks: # mp estimates 33 human pose landmarks, stores each as dict in a list, else none

                for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):

                    data.append({
                        "frame": frame_id,
                        "landmark": landmark_id,
                        "x": landmark.x,
                        "y": landmark.y, # normalized coordinates 0->1, x=0.5 means centre of frame
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
            else: # no frame found
                for landmark_id in range(33):
                    data.append({
                        "frame": frame_id,
                        "landmark": landmark_id,
                        "x": np.nan,
                        "y": np.nan,
                        "z": np.nan,
                        "visibility": 0
                    })

            frame_id += 1
            pbar.update(1) #progress bar update for tqdm

    cap.release() #close vid file

    df = pd.DataFrame(data) 
    return df
    