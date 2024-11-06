"""
dropped_tracks_comparison.py

Makes figure for comparing frame tracking (versus drops) between each camera's individual tracking and the reprojection.

NOTE: only uses the black mouse.
NOTE: assumes file names and directory structure from behavior_pipeline.py.
"""

### Imports.

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

### Global variables.

# Global path to the session directory.
SESSION_DIRECTORY = "/mnt/cup/labs/witten/Ben/Projects/StressDASert/Behavior/Data/SleapTrainVideos/2024-09-25/Bl6SW_3/[2024-09-25_14-00-35]-SleapTrain_Bl6SW_3"
OUTPUT = "../Figures"

### Functions.

def get_original_camera_tracks(camera_directory):
    """
    Given path to camera directory, returns the original tracks inferred on video from that camera.

    Parameters:
        - camera_directory (str): Path to camera directory.

    Returns:
        - black_original_tracks (list): List of tracks for each frame (black mouse only).
    """

    ### Get the original tracks for the camera view (black mouse only).

    camera_directory_contents = os.listdir(os.path.join(SESSION_DIRECTORY, camera_directory))
    original_black_tracks = [f for f in camera_directory_contents if f.endswith(".h5") and "black" in f][0]
    original_black_tracks_filepath = os.path.join(SESSION_DIRECTORY, camera_directory, original_black_tracks)
    black_original_tracks = h5py.File(original_black_tracks_filepath, "r")["tracks"][0, :, :, :] # Squash along dim 0 (only 1 instance).
    black_original_tracks = np.swapaxes(black_original_tracks, 0, 2) # Swap first and third axes to match the reprojected tracks.

    return black_original_tracks


def get_3D_tracks(pose_3D_filepath):
    """
    Gets the 3D pose tracks for the black mouse.

    Parameters:
        - pose_3D_filepath (str): Path to the 3D pose file.

    Returns:
        - tracks_3D (np.ndarray): 3D tracks for the black mouse.
    """

    ### Open the h5 file.

    tracks_3D = h5py.File(pose_3D_filepath, "r")["tracks"][:, 0, :, :] # Squash along dim 1 (only 1 instance).

    return tracks_3D


def main():

    ### Get original tracks for each camera.

    # Get paths to each camera directory.
    camera_directories = [os.path.join(SESSION_DIRECTORY, camera) for camera in os.listdir(SESSION_DIRECTORY) if "Camera" in camera]
    camera_directories.sort()

    # Get the sleap tracks for the camera view.
    original_camera_view_tracks = []
    for camera_directory in camera_directories:
        camera_tracks = get_original_camera_tracks(camera_directory)
        original_camera_view_tracks.append(camera_tracks)

    ### Get the 3D tracks.

    black_3D_pose_filepath = os.path.join(SESSION_DIRECTORY, "black_triangulated.h5")
    tracks_3D = get_3D_tracks(black_3D_pose_filepath)

    ### Check that all original tracks and 3D tracks have the same number of frames.

    assert all([camera_tracks.shape[0] == tracks_3D.shape[0] for camera_tracks in original_camera_view_tracks]), "Original tracks and 3D tracks have different number of frames."

    ### Make 1D binary vector for whether each frame in each track has all tracks or not.

    # For the original tracks.
    original_tracks_frame_success_vector = []
    for original_track in original_camera_view_tracks:
        tracker = np.all(~np.isnan(original_track), axis=(0, 1))
        original_tracks_frame_success_vector.append(tracker)
    original_tracks_frame_success_vector = np.array(original_tracks_frame_success_vector)

    # For the 3D tracks.
    tracker_3D_frame_success_vector = np.all(~np.isnan(tracks_3D), axis=(0, 1, 2))

    ### Make the figure: raster of frame success for each camera view and 3D tracks.

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot the original tracks.
    for i, tracker in enumerate(original_tracks_frame_success_vector):
        ax.plot(tracker + i, color="black", label=f"Camera {i+1}")

    # Plot the 3D tracks.
    ax.plot(tracker_3D_frame_success_vector + len(original_tracks_frame_success_vector), color="blue", label="3D")

    # Set the labels.
    ax.set_xlabel("Frame")
    ax.set_ylabel("Success")
    ax.set_title("Frame Tracking Success")
    ax.legend()

    # Save the figure.
    fig.savefig(os.path.join(OUTPUT, "frame_tracking_success.png"), dpi=300)


if __name__ == "__main__":
    main()

