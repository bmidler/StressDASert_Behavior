"""
reprojection_comparison.py

Takes both the original tracks and tracks reprojected from the 3D pose and overlays them on the original video.
The objective is to enable visual comparison.

NOTE: assumes filename/directory conventions as produced by behavior_pipeline.py.
"""

### Imports.

import os
import cv2
import h5py
import numpy as np
from tqdm import trange

### Global variables.

# Global path to the session directory.
SESSION_DIRECTORY = "/mnt/cup/labs/witten/Ben/Projects/StressDASert/Behavior/Data/SleapTrainVideos/2024-09-25/Bl6SW_3/[2024-09-25_14-00-35]-SleapTrain_Bl6SW_3"
OUTPUT = "../Figures/ReprojectionComparison"


def make_comparison_video(video_filepath, black_original_tracks, white_original_tracks, black_reprojected_tracks, white_reprojected_tracks):
    """
    Makes a video comparing the tracked points of original inference and reprojected inference.

    Parameters:
        - video_filepath (str): path to the original video.
        - black_original_tracks (np.ndarray): original black tracks.
        - white_original_tracks (np.ndarray): original white tracks.
        - black_reprojected_tracks (np.ndarray): reprojected black tracks.
        - white_reprojected_tracks (np.ndarray): reprojected white tracks.

    Returns:
        - None.
    """

    ### Open the video.

    cap = cv2.VideoCapture(video_filepath)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = cap.get(5)
    frame_count = int(cap.get(7))

    ### Create video with original and reprojected tracks overlaid.

    # Create the output video (using OUTPUT global variable).
    video_filename = os.path.basename(video_filepath.split("/")[-1])
    video_filename = video_filename.replace(".mp4", "_reprojection_comparison.mp4")
    output_filepath = os.path.join(OUTPUT, video_filename)
    out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    # Loop through each frame overlaying tracks.
    # Original black: orange; reprojected black: red.
    # Original white: purple; reprojected white: blue.
    for i in trange(frame_count, desc="Making comparison video"):

        # Read the frame.
        ret, frame = cap.read()
        if not ret:
            break

        # Get the original and reprojected tracks for this frame.
        black_original = black_original_tracks[i]
        white_original = white_original_tracks[i]
        black_reprojected = black_reprojected_tracks[i]
        white_reprojected = white_reprojected_tracks[i]

        # Overlay the tracks.
        for j in range(black_original.shape[0]):

            # Black original.
            try:
                cv2.circle(frame, tuple(black_original[j].astype(int)), 4, (0, 165, 255), -1)
            except:
                if not np.isnan(black_original[j]).any(): # Only print the real errors (not NaNs).
                    print(f"Error at frame {i}, black, original, point {black_original[j]}.")

            # Black reprojected.
            try:
                cv2.circle(frame, tuple(black_reprojected[j].astype(int)), 4, (0, 0, 255), -1)
            except:
                if not np.isnan(black_reprojected[j]).any(): # Only print the real errors (not NaNs).
                    print(f"Error at frame {i}, black, reprojected, point {black_reprojected[j]}.")

        for j in range(white_original.shape[0]):
            # White original.
            try:
                cv2.circle(frame, tuple(white_original[j].astype(int)), 4, (255, 0, 255), -1)
            except:
                if not np.isnan(white_original[j]).any(): # Only print the real errors (not NaNs).
                    print(f"Error at frame {i}, white, original, point {white_original[j]}.")

            # White reprojected.
            try:
                cv2.circle(frame, tuple(white_reprojected[j].astype(int)), 4, (255, 0, 0), -1)
            except:
                if not np.isnan(white_reprojected[j]).any(): # Only print the real errors (not NaNs).
                    print(f"Error at frame {i}, white, reprojected, point {white_reprojected[j]}.")

        # Write the frame to the output video.
        out.write(frame)

    # Release the video.
    cap.release()

    # Release the output video.
    out.release()


def main():

    ### Get the files.

    files = os.listdir(SESSION_DIRECTORY)
    camera_directories = [f for f in files if f.startswith("Camera")]
    black_reprojected_file = [f for f in files if f.endswith("black_triangulated_reprojected.h5")][0]
    white_reprojected_file = [f for f in files if f.endswith("white_triangulated_reprojected.h5")][0]
    black_reprojected_filepath = os.path.join(SESSION_DIRECTORY, black_reprojected_file)
    white_reprojected_filepath = os.path.join(SESSION_DIRECTORY, white_reprojected_file)

    # Get the camera view reprojected tracks.
    black_reprojected_tracks_file = h5py.File(black_reprojected_filepath, "r")
    white_reprojected_tracks_file = h5py.File(white_reprojected_filepath, "r")
    camera_views = [key for key in black_reprojected_tracks_file.keys() if key.startswith("Camera")]

    ### Loop through each camera view and make the video.

    # Sort the camera directories and camera views.
    camera_directories.sort()
    camera_views.sort()

    for i, camera_directory in enumerate(camera_directories):

        camera_directory_contents = os.listdir(os.path.join(SESSION_DIRECTORY, camera_directory))

        # Get the path to the video: an mp4 file without "tracks" in the name.
        video_filename = [f for f in camera_directory_contents if f.endswith(".mp4") and "tracks" not in f][0]
        video_filepath = os.path.join(SESSION_DIRECTORY, camera_directory, video_filename)

        # Get the original tracks: h5 file in the camera directory with "black" or "white" in the name.
        original_black_tracks = [f for f in camera_directory_contents if f.endswith(".h5") and "black" in f][0]
        original_black_tracks_filepath = os.path.join(SESSION_DIRECTORY, camera_directory, original_black_tracks)
        original_white_tracks = [f for f in camera_directory_contents if f.endswith(".h5") and "white" in f][0]
        original_white_tracks_filepath = os.path.join(SESSION_DIRECTORY, camera_directory, original_white_tracks)

        # Get the original tracks files.
        black_original_tracks = h5py.File(original_black_tracks_filepath, "r")["tracks"][0, :, :, :] # Squash along dim 0 (only 1 instance).
        white_original_tracks = h5py.File(original_white_tracks_filepath, "r")["tracks"][0, :, :, :] # Squash along dim 0 (only 1 instance).

        # Swap first and third axes to match the reprojected tracks.
        black_original_tracks = np.swapaxes(black_original_tracks, 0, 2)
        white_original_tracks = np.swapaxes(white_original_tracks, 0, 2)

        # Get the black/white reprojected tracks for this view: corresponding camera view in the reprojected tracks.
        black_reprojected_tracks = black_reprojected_tracks_file[camera_views[i]][:, 0, :, :] # Squash along dim 1 (only 1 instance).
        white_reprojected_tracks = white_reprojected_tracks_file[camera_views[i]][:, 0, :, :] # Squash along dim 1 (only 1 instance).

        # Make the video.
        make_comparison_video(video_filepath, black_original_tracks, white_original_tracks, black_reprojected_tracks, white_reprojected_tracks)


if __name__ == "__main__":
    main()