"""
behavior_pipeline.py

Master python script that runs the pipeline for processing a behavior video recording session.
Takes as global variables:
    - Path to the session folder (folder that contains video folders for each camera).
    - Path to centroid sleap model (black mouse).
    - Path to centered sleap model (black mouse).
    - Path to centroid sleap model (white mouse).
    - Path to centered sleap model (white mouse).
    - Anipose calibration file.

Does the following:
    - Runs sleap inference on each view in the session.
    - Triangulates unified 3D pose based on pose tracking for each camera view in the session.
    - Re-projects poses back to each camera view based on homogenized 3D pose.
    - Outputs video of the tracks.
"""

### Import statements.

import os
import sys
import cv2
import h5py
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt

### Function definitions.

def reset_session(session_folder):
    """
    In case we're re-processing a session, delete all QC video files and track files (for session and individual cameras).

    Parameters:
        - session_folder (str): Path to the session folder containing video folders for each camera.

    Returns:
        - None.
    """

    ### Get list of files in session directory and delete QC videos and tracks.

    files = os.listdir(session_folder)

    # Videos.
    if "skeleton_video.mp4" in files:
        os.remove(os.path.join(session_folder, "skeleton_video.mp4"))
    
    if "topdown_skeleton_video.mp4" in files:
        os.remove(os.path.join(session_folder, "topdown_skeleton_video.mp4"))

    # Tracks.
    if "black_triangulated_reprojected.h5" in files:
        os.remove(os.path.join(session_folder, "black_triangulated_reprojected.h5"))
    if "white_triangulated_reprojected.h5" in files:
        os.remove(os.path.join(session_folder, "white_triangulated_reprojected.h5"))
    if "black_triangulated.h5" in files:
        os.remove(os.path.join(session_folder, "black_triangulated.h5"))
    if "white_triangulated.h5" in files:
        os.remove(os.path.join(session_folder, "white_triangulated.h5"))

    ### Remove tracks and QC video files from each camera's directory.

    camera_directories = [f for f in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, f)) and "Camera" in f]

    for camera_dir in camera_directories:
        files = os.listdir(os.path.join(session_folder, camera_dir))
        # Videos.
        if "tracked_video.mp4" in files:
            os.remove(os.path.join(session_folder, camera_dir, "tracked_video.mp4"))

        # Tracks--delete all h5 and slp files.
        for file in files:
            if file.endswith(".h5") or file.endswith(".slp"):
                os.remove(os.path.join(session_folder, camera_dir, file))

    ### Remove UnusedInference folders.

    for camera_dir in camera_directories:
        unused_inference_path = os.path.join(session_folder, camera_dir, "UnusedInference")
        if os.path.exists(unused_inference_path):
            for file in os.listdir(unused_inference_path):
                os.remove(os.path.join(unused_inference_path, file))
            os.rmdir(unused_inference_path)

def run_sleap_inference(session_folder, centroid_model, centered_model, mouse_color):
    """
    Runs sleap inference on each camera view (video) in the session using the passed centroid and centered models.
    This will need to be called seperatly for black and white mice.

    NOTE: this function works by calling the SBATCH script to run inference using GPUs on spock.

    Parameters:
        - session_folder (str): Path to the session folder containing video folders for each camera.
        - centroid_model (str): Path to the centroid sleap model.
        - centered_model (str): Path to the centered sleap model.
        - mouse_color (str): Color of the mouse (black or white).

    Returns:
        - None.
    """

    ### Get the paths to each video in the session.

    # Get list of folders (one per camera) in session directory.
    video_folders = [f for f in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, f))]

    # In each camera folder, get path to the mp4 file regardless of video file name.
    video_paths = []
    for folder in video_folders:
        for file in os.listdir(os.path.join(session_folder, folder)):
            if file.endswith(".mp4"):
                video_paths.append(os.path.join(session_folder, folder, file))

    # Drop video if name is "tracked_video", in case we're re-processing a session.
    video_paths = [path for path in video_paths if "tracked_video" not in path]

    print(f"\tFound {len(video_paths)} videos in session folder.")

    ### Submit SBATCH jobs for each video.

    for video_path in video_paths:

        # Get video name being processed.
        video_name = video_path.split("/")[-1]

        # Get the directory path of the video.
        video_dir_path = os.path.dirname(video_path)

        # Construct the global path for the tracks file.
        global_tracks_file_path = os.path.join(video_dir_path, f"{mouse_color}_inference")

        # Get the relative path for the tracks file.
        cwd = os.getcwd()
        relative_tracks_file_path = os.path.relpath(global_tracks_file_path, cwd)

        # Check to see if a file with that name already exists, if so, delete.
        if os.path.exists(global_tracks_file_path):
            os.remove(global_tracks_file_path)

        # Get names for output and error files.
        error_file = f"{mouse_color}_{video_name}_errors.txt"
        output_file = f"{mouse_color}_{video_name}_outputs.txt"

        # Construct the command to run the SBATCH script with dynamic error and output file paths.
        command = (
            f"sbatch --error=SBATCH_outputs/{SESSION_NAME}/{error_file} --output=SBATCH_outputs/{SESSION_NAME}/{output_file} "
            f"{INFERENCE_SBATCH_SCRIPT} {video_path} {centroid_model} {centered_model} {relative_tracks_file_path}"
        )

        # Run the command and capture the output and errors.
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Get the output and error messages.
        output = result.stdout.decode("utf-8")
        error = result.stderr.decode("utf-8")
        
        # Print error message if there is one, otherwise print the job ID.
        if result.returncode != 0:
            print(f"\tError: {error}")
            return
        else:
            # Extract and print the job ID.
            if "Submitted batch job" in output:
                job_id = output.split()[-1]

        print(f"\tRunning inference on {video_name} (Job ID: {job_id})")


def wait_for_inference_to_complete(session_folder):
    """
    A loop that checks the directories for each camera view in the session for the inference files for black and white mice.
    Only advances when all inference files have been generated.

    Parameters:
        - session_folder (str): Path to the session folder containing video folders for each camera.

    Returns:
        - None.
    """

    ### Get the paths to each video in the session.

    # Get list of folders (one per camera) in session directory.
    video_folders = [f for f in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, f))]

    ### Loop until all inference files are generated.

    all_inference_files_present = False
    while not all_inference_files_present:  # Keep looping until all inference files are present.

        for folder in video_folders:
            # Check for inference files for black mouse (if there's an h5 file with "black" in the name).
            black_inference_files = [f for f in os.listdir(os.path.join(session_folder, folder)) if "black" in f and f.endswith(".h5")]
            if len(black_inference_files) == 0:
                all_inference_files_present = False
                break
            else:
                all_inference_files_present = True

            # Check for inference files for white mouse (if there's an h5 file with "white" in the name).
            white_inference_files = [f for f in os.listdir(os.path.join(session_folder, folder)) if "white" in f and f.endswith(".h5")]
            if len(white_inference_files) == 0:
                all_inference_files_present = False
                break
            else:
                all_inference_files_present = True

        else:
            all_inference_files_present = True


def run_anipose_triangulation():
    """
    Runs 3D triangulation using anipose on the inference files generated from sleap.
    This function uses the anipose_triangulation sbatch script and also runs reprojection.

    NOTE: because sleap anipose isn't able to differentiate the sleap track files for different mice,
    this function creates a temporary sub-folder to put the sleap tracks for the mouse not being processed.

    Parameters:
        - None.

    Returns:
        - None.
    """

    ### Run triangulation and reprojection using sbatch script.

    # In each camera folder, create new directory to place the inference file not being used.
    video_folders = [f for f in os.listdir(SESSION_FOLDER) if os.path.isdir(os.path.join(SESSION_FOLDER, f))]
    for folder in video_folders:

        # Make sub-folder to hold the inference file not being used.
        os.makedirs(os.path.join(SESSION_FOLDER, folder, "UnusedInference"), exist_ok=True)

    ### Black mouse.

    # Move white mouse inference files to the UnusedInference folders.
    for folder in video_folders:
        # Get list of all files with "white" in the name: move those files.
        files_to_move = [f for f in os.listdir(os.path.join(SESSION_FOLDER, folder)) if "white" in f]
        for file in files_to_move:
            white_inference_file = os.path.join(SESSION_FOLDER, folder, file)
            os.rename(white_inference_file, os.path.join(SESSION_FOLDER, folder, "UnusedInference", file))  # Move to UnusedInference folder.

    # Run anipose triangulation for black mouse using sbatch script.
    print("Running anipose triangulation for black mouse...")
    session_directory = SESSION_FOLDER
    calibration_file = ANIPOSE_CALIBRATION_FILE
    output_filename = session_directory + "/" + "black_triangulated.h5"
    error_file = f"SBATCH_outputs/{SESSION_NAME}/black_triangulation_errors.txt"
    output_file = f"SBATCH_outputs/{SESSION_NAME}/black_triangulation_outputs.txt"
    command = (f"sbatch --error={error_file} --output={output_file} {ANIPOSE_TRIANGULATION_SBATCH_SCRIPT} {session_directory} {calibration_file} {output_filename}")

    # Run the command and capture the output and errors.
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the output and error messages.
    output = result.stdout.decode("utf-8")
    error = result.stderr.decode("utf-8")

    # Print error message if there is one, otherwise print the job ID.
    if result.returncode != 0:
        print(f"\tError: {error}")
        return
    else:
        # Extract and print the job ID.
        if "Submitted batch job" in output:
            job_id = output.split()[-1]

    print(f"\tRunning triangulation/reprojection for black mouse (Job ID: {job_id})")

    # Wait until the subprocess is done.
    while True:
        if not os.path.exists(output_filename):
            time.sleep(10)
        else:
            break

    # Move white mouse inference files back to their original locations.
    for folder in video_folders:
        # Get list of all files in UnusedInference folder: move those files back.
        files_to_move = [f for f in os.listdir(os.path.join(SESSION_FOLDER, folder, "UnusedInference")) if "white" in f]
        for file in files_to_move:
            reverted_path = os.path.join(SESSION_FOLDER, folder, file)
            os.rename(os.path.join(SESSION_FOLDER, folder, "UnusedInference", file), reverted_path)  # Move back to original location.

    ### White mouse.

    # Move black mouse inference files to the UnusedInference folders.
    for folder in video_folders:
        # Get list of all files with "black" in the name: move those files.
        files_to_move = [f for f in os.listdir(os.path.join(SESSION_FOLDER, folder)) if "black" in f]
        for file in files_to_move:
            black_inference_file = os.path.join(SESSION_FOLDER, folder, file)
            os.rename(black_inference_file, os.path.join(SESSION_FOLDER, folder, "UnusedInference", file))  # Move to UnusedInference folder.

    # Run anipose triangulation for white mouse using sbatch script.
    print("Running anipose triangulation for white mouse...")
    session_directory = SESSION_FOLDER
    calibration_file = ANIPOSE_CALIBRATION_FILE
    output_filename = session_directory + "/" + "white_triangulated.h5"
    error_file = f"SBATCH_outputs/{SESSION_NAME}/white_triangulation_errors.txt"
    output_file = f"SBATCH_outputs/{SESSION_NAME}/white_triangulation_outputs.txt"
    command = (f"sbatch --error={error_file} --output={output_file} {ANIPOSE_TRIANGULATION_SBATCH_SCRIPT} {session_directory} {calibration_file} {output_filename}")

    # Run the command and capture the output and errors.
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the output and error messages.
    output = result.stdout.decode("utf-8")
    error = result.stderr.decode("utf-8")

    # Print error message if there is one, otherwise print the job ID.
    if result.returncode != 0:
        print(f"\tError: {error}")
        return
    else:
        # Extract and print the job ID.
        if "Submitted batch job" in output:
            job_id = output.split()[-1]

    print(f"\tRunning triangulation/reprojection for white mouse (Job ID: {job_id})")
    
    # Wait until the subprocess is done.
    while True:
        if not os.path.exists(output_filename):
            time.sleep(10)
        else:
            break

    # Move black mouse inference files back to their original locations.
    for folder in video_folders:
        # Get list of all files in UnusedInference folder: move those files back.
        files_to_move = [f for f in os.listdir(os.path.join(SESSION_FOLDER, folder, "UnusedInference")) if "black" in f]
        for file in files_to_move:
            reverted_path = os.path.join(SESSION_FOLDER, folder, file)
            os.rename(os.path.join(SESSION_FOLDER, folder, "UnusedInference", file), reverted_path)  # Move back to original location.

    # Delete the UnusedInference folders.
    for folder in video_folders:
        os.rmdir(os.path.join(SESSION_FOLDER, folder, "UnusedInference"))


def wait_for_triangulation_to_complete(session_folder):
    """
    A waiting function that makes sure triangulation and reprojection are done before advancing.

    Parameters:
        - session_folder (str): Path to the session folder containing video folders for each camera.

    Returns:
        - None.
    """

    ### Loop until all triangulation/reprojection files are generated.

    all_triangulation_files_present = False
    while not all_triangulation_files_present:  # Keep looping until all triangulation files are present.

        # Get list of files in session directory.
        files = os.listdir(session_folder)

        # Check for black mouse files.
        black_triangulation_files = [f for f in files if "black_triangulated_reprojected" in f]
        black_files_present = len(black_triangulation_files) > 0

        # Check for white mouse files.
        white_triangulation_files = [f for f in files if "white_triangulated_reprojected" in f]
        white_files_present = len(white_triangulation_files) > 0

        # Update the condition to check if both sets of files are present.
        all_triangulation_files_present = black_files_present and white_files_present


def make_single_video(video_path, black_tracks, white_tracks, fname):
    """
    Makes a single video with overlaid tracks for both black and white mice.

    Parameters:
        - video_path (str): Path to the original video.
        - black_tracks (np.array): black tracks [frames, point, x, y]
        - white_tracks (np.array): white tracks [frames, point, x, y]
        - fname (str): Name of the output video.

    Returns:
        - None.
    """

    ### Open the video file.

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video properties.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{fname}.mp4", fourcc, fps, (width, height))
    n_frames = np.shape(black_tracks)[0] # In case I only tracked a subset of the video.

    # Make sure the tracks are the same length.
    assert np.shape(black_tracks)[0] == np.shape(white_tracks)[0], "Black and white tracks are not the same length."

    # Report-out number of frames being tracked vs in the video.
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\tTracking {n_frames} frames out of {n_frames_video} in the video (based on the n frames tracked by sleap [0, n]).")

    ### Loop through the video frames and overlay the tracks.

    frame_num = 0
    while cap.isOpened():
            
            # Read the frame.
            ret, frame = cap.read()
            if not ret:
                break
    
            # Get the tracks for this frame.
            black_frame_tracks = black_tracks[frame_num]
            white_frame_tracks = white_tracks[frame_num]

            # Get points to exclude.
            indices_to_exclude = [POINT_INDICES.index(point) for point in POINTS_TO_EXCLUDE]
    
            # Overlay the tracks on the frame, blue for white mouse, red for black mouse.
            for i in range(black_frame_tracks.shape[0]):
                if i in indices_to_exclude:
                    continue
                x, y = int(black_frame_tracks[i, 0]), int(black_frame_tracks[i, 1])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            for i in range(white_frame_tracks.shape[0]):
                if i in indices_to_exclude:
                    continue
                x, y = int(white_frame_tracks[i, 0]), int(white_frame_tracks[i, 1])
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

            # Draw lines for the connections.
            # Black mouse.
            for connection in CONNECTIONS:
                point1 = black_frame_tracks[POINT_INDICES.index(connection[0])]
                point2 = black_frame_tracks[POINT_INDICES.index(connection[1])]
                if connection[0] in POINTS_TO_EXCLUDE or connection[1] in POINTS_TO_EXCLUDE:
                    continue
                cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1]),), (0, 0, 255), 2)
            # White mouse.
            for connection in CONNECTIONS:
                point1 = white_frame_tracks[POINT_INDICES.index(connection[0])]
                point2 = white_frame_tracks[POINT_INDICES.index(connection[1])]
                if connection[0] in POINTS_TO_EXCLUDE or connection[1] in POINTS_TO_EXCLUDE:
                    continue
                cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1]),), (255, 0, 0), 2)
    
            # Write the frame to the output video.
            out.write(frame)
    
            # Increment the frame number.
            frame_num += 1

            # Break if we've reached the end of the tracks.
            if frame_num >= n_frames:
                break

    ### Release the video and output video.

    cap.release()
    out.release()


def open_h5_file_with_retry(filepath, mode="r", retries=100, delay=2):
    """
    Attempts to open and return an h5 file, but retries if the file is locked (eg.g another process is accessing it).
    Fixes an issue where the h5 reprojection files are locked by the anipose process.

    Parameters:
        - filepath (str): Path to the h5 file.
        - mode (str): Mode to open the file in.
        - retries (int): Number of times to retry opening the file.
        - delay (int): Delay between retries.

    Returns:
        - file (h5py.File): Opened h5 file.
    """
    for attempt in range(retries):
        try:
            file = h5py.File(filepath, mode)
            print(f"\tOpened file {filepath}.")
            return file
        except OSError as e:
            if "unable to lock file" in str(e):
                print(f"\tAttempt {attempt + 1} of {retries}: Unable to open file {filepath}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise
    raise OSError(f"\tFailed to open file {filepath} after {retries} attempts.")


def project_to_2d(points, plane):
    """
    Helper function for the make_skeleton_video function that projects 3D points to 2D using a plane.

    Parameters:
        - points (np.array): 3D points [frames, points, x/y/z].
        - plane (np.array): 2D plane [2, 3].

    Returns:
        - np.array: 2D points [frames, points, x/y].
    """
    return np.dot(points, plane.T)


def rotation_matrix_x(angle):
    """
    Create a rotation matrix for rotating points around the x-axis.

    Parameters:
        - angle (float): The rotation angle in radians.

    Returns:
        - np.array: The rotation matrix.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])

def rotation_matrix_y(angle):
    """
    Create a rotation matrix for rotating points around the y-axis.

    Parameters:
        - angle (float): The rotation angle in radians.

    Returns:
        - np.array: The rotation matrix.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])

def rotation_matrix_z(angle):
    """
    Create a rotation matrix for rotating points around the z-axis.

    Parameters:
        - angle (float): The rotation angle in radians.

    Returns:
        - np.array: The rotation matrix.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])


def make_topdown_skeleton_video(black_reprojection_tracks_file, white_reprojection_tracks_file, topdown_camera_name="Camera0"):
    """
    Makes a video of the skeletons inside a bounding box using the 3D tracks reprojected to the top-down view.

    Parameters:
        - black_reprojection_tracks_file (h5 file): black mouse reprojected tracks file.
        - white_reprojection_tracks_file (h5 file): white mouse reprojected tracks file.
        - topdown_camera_name (str): Name of the top-down camera.

    Returns:
        - None.
    """

    ### Get 3D tracks reprojected to top-down view.

    # Get white and black mouse tracks for top-down view.
    black_reprojection_tracks_data = black_reprojection_tracks_file[topdown_camera_name][:, 0, :, :]  # Squash along dim 1 (only 1 instance).
    white_reprojection_tracks_data = white_reprojection_tracks_file[topdown_camera_name][:, 0, :, :]  # Squash along dim 1 (only 1 instance).

    # Remove points to exclude.
    indices_to_exclude = [POINT_INDICES.index(point) for point in POINTS_TO_EXCLUDE]
    black_reprojection_tracks_data_no_tail = np.delete(black_reprojection_tracks_data, indices_to_exclude, axis=1)
    white_reprojection_tracks_data_no_tail = np.delete(white_reprojection_tracks_data, indices_to_exclude, axis=1)

    # Get the corners.
    max_x = np.max([np.max(black_reprojection_tracks_data_no_tail[:, :, 0]), np.max(white_reprojection_tracks_data_no_tail[:, :, 0])])
    max_y = np.max([np.max(black_reprojection_tracks_data_no_tail[:, :, 1]), np.max(white_reprojection_tracks_data_no_tail[:, :, 1])])
    min_x = np.min([np.min(black_reprojection_tracks_data_no_tail[:, :, 0]), np.min(white_reprojection_tracks_data_no_tail[:, :, 0])])
    min_y = np.min([np.min(black_reprojection_tracks_data_no_tail[:, :, 1]), np.min(white_reprojection_tracks_data_no_tail[:, :, 1])])

    corners = np.array([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, min_y],
        [max_x, max_y]
    ])

    ### Get the top-down video.

    topdown_directory = os.path.join(SESSION_FOLDER, topdown_camera_name)
    video_filename = [f for f in os.listdir(topdown_directory) if f.endswith(".mp4") and "track" not in f][0] # Makes sure we grab the original video.
    video_path = os.path.join(topdown_directory, video_filename)
    fname = os.path.join(SESSION_FOLDER, "topdown_skeleton_video.mp4")

    ### Open the video file.

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video properties.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{fname}", fourcc, fps, (width, height))
    n_frames = np.shape(black_reprojection_tracks_data)[0] # In case I only tracked a subset of the video.

    # Make sure the tracks are the same length.
    assert np.shape(black_reprojection_tracks_data)[0] == np.shape(white_reprojection_tracks_data)[0], "Black and white tracks are not the same length."

    # Report-out number of frames being tracked vs in the video.
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\tTracking {n_frames} frames out of {n_frames_video} in the video (based on the n frames tracked by sleap [0, n]).")

    ### Loop through the video frames and overlay the tracks.

    frame_num = 0
    while cap.isOpened():
            
            # Read the frame.
            ret, frame = cap.read()
            if not ret:
                break

            # Make the frame white.
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
            # Get the tracks for this frame.
            black_frame_tracks = black_reprojection_tracks_data[frame_num]
            white_frame_tracks = white_reprojection_tracks_data[frame_num]
    
            # Overlay the tracks on the frame, blue for white mouse, red for black mouse.
            for i in range(black_frame_tracks.shape[0]):
                if i in indices_to_exclude:
                    continue
                x, y = int(black_frame_tracks[i, 0]), int(black_frame_tracks[i, 1])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            for i in range(white_frame_tracks.shape[0]):
                if i in indices_to_exclude:
                    continue
                x, y = int(white_frame_tracks[i, 0]), int(white_frame_tracks[i, 1])
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

            # Draw lines for the connections.
            # Black mouse.
            for connection in CONNECTIONS:
                point1 = black_frame_tracks[POINT_INDICES.index(connection[0])]
                point2 = black_frame_tracks[POINT_INDICES.index(connection[1])]
                if connection[0] in POINTS_TO_EXCLUDE or connection[1] in POINTS_TO_EXCLUDE:
                    continue
                cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1]),), (0, 0, 255), 2)
            # White mouse.
            for connection in CONNECTIONS:
                point1 = white_frame_tracks[POINT_INDICES.index(connection[0])]
                point2 = white_frame_tracks[POINT_INDICES.index(connection[1])]
                if connection[0] in POINTS_TO_EXCLUDE or connection[1] in POINTS_TO_EXCLUDE:
                    continue
                cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1]),), (255, 0, 0), 2)

            # Draw the bounding box.
            cv2.line(frame, (int(corners[0][0]), int(corners[0][1])), (int(corners[1][0]), int(corners[1][1])), (0, 0, 0), 2)
            cv2.line(frame, (int(corners[0][0]), int(corners[0][1])), (int(corners[2][0]), int(corners[2][1])), (0, 0, 0), 2)
            cv2.line(frame, (int(corners[1][0]), int(corners[1][1])), (int(corners[3][0]), int(corners[3][1])), (0, 0, 0), 2)
            cv2.line(frame, (int(corners[2][0]), int(corners[2][1])), (int(corners[3][0]), int(corners[3][1])), (0, 0, 0), 2)
    
            # Write the frame to the output video.
            out.write(frame)
    
            # Increment the frame number.
            frame_num += 1

            # Break if we've reached the end of the tracks.
            if frame_num >= n_frames:
                break

    ### Release the video and output video.

    cap.release()
    out.release()

    print(f"\tSaved skeleton video to {fname}.")


def make_skeleton_video(black_3D_pose_filepath, white_3D_pose_filepath, session_folder, width, height, fps):
    """
    Makes an animation of just the tracked points.

    Parameters:
        - black_3D_pose_filepath (str): Path to the black mouse 3D pose h5 file.
        - white_3D_pose_filepath (str): Path to the white mouse 3D pose h5 file.
        - session_folder (str): Path to the session folder containing video folders for each camera.
        - width (int): Width of the video.
        - height (int): Height of the video.
        - fps (int): Frames per second of the video.

    Returns:
        - None.
    """

    ### Open the 3D pose files [frames, instances, points, x/y/z].

    # Open files.
    black_3D_pose_file = open_h5_file_with_retry(black_3D_pose_filepath)["tracks"]
    white_3D_pose_file = open_h5_file_with_retry(white_3D_pose_filepath)["tracks"]

    # Collapse the instance dimension (only 1 instance).
    black_3D_pose = black_3D_pose_file[:, 0, :, :]
    white_3D_pose = white_3D_pose_file[:, 0, :, :]

    # Assert the 3D pose files are the same length.
    assert black_3D_pose.shape[0] == white_3D_pose.shape[0], "Black and white mouse 3D pose files do not have the same number of frames."

    ### Get max x, y, and z coordinate to draw a bounding box between all four corners.

    # Get max x, y, and z coordinates--no tail points.
    indices_to_exclude = [POINT_INDICES.index(point) for point in POINTS_TO_EXCLUDE]
    black_3D_pose_no_tail = np.delete(black_3D_pose, indices_to_exclude, axis=1)
    white_3D_pose_no_tail = np.delete(white_3D_pose, indices_to_exclude, axis=1)

    max_x = np.max([np.max(black_3D_pose_no_tail[:, :, 0]), np.max(white_3D_pose_no_tail[:, :, 0])])
    max_y = np.max([np.max(black_3D_pose_no_tail[:, :, 1]), np.max(white_3D_pose_no_tail[:, :, 1])])
    max_z = np.max([np.max(black_3D_pose_no_tail[:, :, 2]), np.max(white_3D_pose_no_tail[:, :, 2])])

    # Get min x, y, and z coordinates--no tail points.
    min_x = np.min([np.min(black_3D_pose_no_tail[:, :, 0]), np.min(white_3D_pose_no_tail[:, :, 0])])
    min_y = np.min([np.min(black_3D_pose_no_tail[:, :, 1]), np.min(white_3D_pose_no_tail[:, :, 1])])
    min_z = np.min([np.min(black_3D_pose_no_tail[:, :, 2]), np.min(white_3D_pose_no_tail[:, :, 2])])

    # Get the corners of the bounding box.
    corners = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z]
    ])

    ### Make animation: for each frame, project 3D points for both mice and box corners to 2D.

    # Create video writer.
    output_video_path = output_video_path = os.path.join(session_folder, "skeleton_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_size = (width, height)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    num_frames = black_3D_pose.shape[0]

    # Loop through frames.
    for frame_num in range(num_frames):

        # Create blank image.
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255

        # Rotation parameters.
        rotation_speed = np.pi / 12
        elapsed_time = frame_num / fps
        angle = rotation_speed * elapsed_time

        # Create the rotation matrices.
        rot_matrix_x = rotation_matrix_x(angle)
        rot_matrix_y = rotation_matrix_y(angle)
        rot_matrix_z = rotation_matrix_z(angle)

        # Combine the rotation matrices.
        rot_matrix = np.dot(rot_matrix_z, np.dot(rot_matrix_y, rot_matrix_x))

        # Define the 2D plane based on the rotation matrix.
        plane = np.dot(rot_matrix, np.array([[1, 0, 0], [0, 1, 0]]).T).T

        # Project 3D points to 2D.
        black_2D = project_to_2d(black_3D_pose[frame_num], plane)
        white_2D = project_to_2d(white_3D_pose[frame_num], plane)
        corners_2D = project_to_2d(corners, plane)

        # Calculate the center of the bounding box in 2D.
        center_2D = np.mean(corners_2D, axis=0)

        # Calculate the translation needed to center the bounding box in the frame.
        translation_x = width / 2 - center_2D[0]
        translation_y = height / 2 - center_2D[1]

        # Apply translation to center the points.
        black_2D[:, 0] += translation_x
        black_2D[:, 1] += translation_y
        white_2D[:, 0] += translation_x
        white_2D[:, 1] += translation_y

        # Apply translation to corners.
        corners_2D[:, 0] += translation_x
        corners_2D[:, 1] += translation_y

        polygons = [
            [corners_2D[0], corners_2D[1], corners_2D[3], corners_2D[2]],
            [corners_2D[4], corners_2D[5], corners_2D[7], corners_2D[6]],
            [corners_2D[0], corners_2D[1], corners_2D[5], corners_2D[4]],
            [corners_2D[2], corners_2D[3], corners_2D[7], corners_2D[6]],
            [corners_2D[1], corners_2D[3], corners_2D[7], corners_2D[5]], # Floor.
        ]

        # Create a blank mask for counting overlaps.
        overlap_mask = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)

        # Define shading values for walls and floor.
        wall_shade = 50 # Lighter.
        floor_shade = 150 # Darker.

        # Fill polygons on the mask and count overlaps.
        for i, polygon in enumerate(polygons):

            # Create a temporary mask for the current polygon.
            temp_mask = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [np.array(polygon, np.int32)], 1)

            # Determine the shade to apply.
            if i < 4:  # Walls
                temp_mask = (temp_mask * wall_shade)
            else:  # Floor
                temp_mask = (temp_mask * floor_shade)

            # Add the temporary mask to the overlap mask (lower values are darker).
            overlap_mask = cv2.add(overlap_mask, temp_mask)

        # Normalize the overlap mask to the range [0, 255].
        shaded_mask = np.clip(overlap_mask, 0, 255)

        # Convert the shaded mask to a 3-channel image.
        shaded_mask_3ch = cv2.merge([shaded_mask, shaded_mask, shaded_mask])

        # Blend the shaded mask with the frame using additive blending.
        frame = cv2.subtract(frame, shaded_mask_3ch)

        # Draw points for mice.
        for i in range(black_2D.shape[0]):
                if i in indices_to_exclude:
                    continue
                x, y = int(black_2D[i, 0]), int(black_2D[i, 1])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # Red for black mouse.
        for i in range(white_2D.shape[0]):
                if i in indices_to_exclude:
                    continue
                x, y = int(white_2D[i, 0]), int(white_2D[i, 1])
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1) # Blue for white mouse.

        # Draw lines for the connections.
        # Black mouse.
        for connection in CONNECTIONS:
            point1 = black_2D[POINT_INDICES.index(connection[0])]
            point2 = black_2D[POINT_INDICES.index(connection[1])]
            if connection[0] in POINTS_TO_EXCLUDE or connection[1] in POINTS_TO_EXCLUDE:
                continue
            cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1]),), (0, 0, 255), 2)
        # White mouse.
        for connection in CONNECTIONS:
            point1 = white_2D[POINT_INDICES.index(connection[0])]
            point2 = white_2D[POINT_INDICES.index(connection[1])]
            if connection[0] in POINTS_TO_EXCLUDE or connection[1] in POINTS_TO_EXCLUDE:
                continue
            cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1]),), (255, 0, 0), 2)

        # Draw bounding box corners in black.
        for corner in corners_2D:
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(frame, (x, y), 10, (0, 0, 0), -1)  # Black for corners.

        # Using the 3D coordinates of the box corners, draw lines between adjacent (sharing at least one x, y, or z coordinate) corners.
        for i in range(8): # Eight for the number of corners.
            for j in range(8): # Will be some redundant drawing.
                if np.sum(np.abs(corners[i] - corners[j])) == max(np.abs(corners[i] - corners[j])):
                    cv2.line(frame, (int(corners_2D[i][0]), int(corners_2D[i][1])), (int(corners_2D[j][0]), int(corners_2D[j][1])), (0, 0, 0), 2)

        # Write frame to video.
        out.write(frame)

    # Release video.
    out.release()

    print(f"\tMade skeleton video: {output_video_path}")


def make_video():
    """
    Uses the 3D triangulated and 2D reprojected pose files to create a video of the tracks both overlaid on the original videos
    and as a video of only the skeletons.

    Parameters:
        - None.

    Returns: 
        - None.
    """

    ### Make separate videos for each camera view with overlaid tracks.

    # Open reprojection h5 files for black and white mice.
    black_reprojection_tracks_filepath = os.path.join(SESSION_FOLDER, "black_triangulated_reprojected.h5")
    white_reprojection_tracks_filepath = os.path.join(SESSION_FOLDER, "white_triangulated_reprojected.h5")

    black_reprojection_tracks_file = open_h5_file_with_retry(black_reprojection_tracks_filepath, "r")
    white_reprojection_tracks_file = open_h5_file_with_retry(white_reprojection_tracks_filepath, "r")

    # Get the list of camera views from the HDF5 file keys
    camera_views = [key for key in black_reprojection_tracks_file.keys() if key.startswith("Camera")]

    # Assert the camera views are the same for both black and white mice.
    assert camera_views == [key for key in white_reprojection_tracks_file.keys() if key.startswith("Camera")], "Camera views are not the same for black and white mice."

    for i, camera_view in enumerate(camera_views):
        print(f"\tMaking video for camera view {i + 1} of {len(camera_views)}...")

        # Parcelate out into individual camera views and squash dim 1 (only 1 instance).
        black_camera_tracks = black_reprojection_tracks_file[camera_view][:, 0, :]
        white_camera_tracks = white_reprojection_tracks_file[camera_view][:, 0, :]

        # Make the video for the current camera view.
        camera_directory = os.path.join(SESSION_FOLDER, camera_view)
        camera_filepath = [f for f in os.listdir(camera_directory) if f.endswith(".mp4")][0]

        # Try making single video and grab the error if there is one.
        try:
            make_single_video(os.path.join(camera_directory, camera_filepath), black_camera_tracks, white_camera_tracks, os.path.join(camera_directory, "tracked_video"))
        except Exception as e:
            print(f"\tError making video for camera view {camera_view}: {e}")

    print("\tTracked point videos done, now making skeleton video.")

    ### Make a video of just the skeletons.

    # Get paths to 3D files.
    black_3D_pose_filepath = os.path.join(SESSION_FOLDER, "black_triangulated.h5")
    white_3D_pose_filepath = os.path.join(SESSION_FOLDER, "white_triangulated.h5")

    # Get video information (width, height, fps) from the first video.
    Camera0_directory = os.path.join(SESSION_FOLDER, "Camera0")
    video_filename = [f for f in os.listdir(Camera0_directory) if f.endswith(".mp4") and "track" not in f][0] # Makes sure we grab the original video.
    video_path = os.path.join(Camera0_directory, video_filename)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Try making the skeleton video and grab the error if there is one.
    try:
        make_skeleton_video(black_3D_pose_filepath, white_3D_pose_filepath, SESSION_FOLDER, width, height, fps)
        make_topdown_skeleton_video(black_reprojection_tracks_file, white_reprojection_tracks_file, TOP_CAMERA_NAME)
    except Exception as e:
        print(f"\tError making skeleton video: {e}")

    print("\tMade skeleton videos.")

    # Close the HDF5 files
    black_reprojection_tracks_file.close()
    white_reprojection_tracks_file.close()


def run_session(cleanup_session=True):
    """
    Runs a single session using the defined global variables.

    Parameters:
        - cleanup_session (bool): Whether to clean up the session folder before running the session. Default is True.

    Returns:
        - None.
    """

    ### Check if there is a folder for SBATCH outputs. Make if not.

    if not os.path.exists(os.path.join("SBATCH_outputs", SESSION_NAME)):  # Check if the output folder exists.
        os.makedirs(os.path.join("SBATCH_outputs", SESSION_NAME))  # Create the output folder.

    ### Run check to make sure the global variable files and SBATCH script exist.

    if not os.path.exists(SESSION_FOLDER):
        raise FileNotFoundError(f"Session folder does not exist: {SESSION_FOLDER}")
    if not os.path.exists(CENTROID_MODEL_BLACK):
        raise FileNotFoundError(f"Centroid model (black) does not exist: {CENTROID_MODEL_BLACK}")
    if not os.path.exists(CENTERED_MODEL_BLACK):
        raise FileNotFoundError(f"Centered model (black) does not exist: {CENTERED_MODEL_BLACK}")
    if not os.path.exists(CENTROID_MODEL_WHITE):
        raise FileNotFoundError(f"Centroid model (white) does not exist: {CENTROID_MODEL_WHITE}")
    if not os.path.exists(CENTERED_MODEL_WHITE):
        raise FileNotFoundError(f"Centered model (white) does not exist: {CENTERED_MODEL_WHITE}")
    if not os.path.exists(ANIPOSE_CALIBRATION_FILE):
        raise FileNotFoundError(f"Anipose calibration file does not exist: {ANIPOSE_CALIBRATION_FILE}")
    if not os.path.exists(INFERENCE_SBATCH_SCRIPT):
        raise FileNotFoundError(f"SBATCH script does not exist: {INFERENCE_SBATCH_SCRIPT}")
    if not os.path.exists(ANIPOSE_TRIANGULATION_SBATCH_SCRIPT):
        raise FileNotFoundError(f"Anipose triangulation SBATCH script does not exist: {ANIPOSE_TRIANGULATION_SBATCH_SCRIPT}")
    
    ### Clean-up any old inference files.

    if cleanup_session:
        reset_session(SESSION_FOLDER)
    
    ### Run sleap inference for each camera view (video) in the session for black and white mice.

    print(f"Running pipeline for session: {SESSION_NAME}")
    print("Using sleap models:")
    print(f"\tBlack mouse centroid model: {CENTROID_MODEL_BLACK.split("/")[-1]}")
    print(f"\tBlack mouse centered model: {CENTERED_MODEL_BLACK.split("/")[-1]}")
    print(f"\tWhite mouse centroid model: {CENTROID_MODEL_WHITE.split("/")[-1]}")
    print(f"\tWhite mouse centered model: {CENTERED_MODEL_WHITE.split("/")[-1]}")

    # Black mouse.
    print("Running sleap inference for black mouse...")
    run_sleap_inference(SESSION_FOLDER, CENTROID_MODEL_BLACK, CENTERED_MODEL_BLACK, "black")

    # White mouse.
    print("Running sleap inference for white mouse...")
    run_sleap_inference(SESSION_FOLDER, CENTROID_MODEL_WHITE, CENTERED_MODEL_WHITE, "white")

    ### Wait until all inference is done running.

    print("Waiting for inference to complete...")
    wait_for_inference_to_complete(SESSION_FOLDER)

    ### Run anipose triangulation.

    run_anipose_triangulation()

    ### Make video of the tracks, if specified.

    if MAKE_VIDEO:

        # Wait for triangulation to complete.
        print("Waiting for triangulation to complete...")
        wait_for_triangulation_to_complete(SESSION_FOLDER)
        print("Making video of the tracks...")
        make_video()

    else:
        print("Not making videos.")


def setup_session(session_folder):
    """
    Takes path to session folder and defines global variables for files in the session, sleap models, etc.
    
    Parameters:
        - session_folder (str): Path to the session folder containing the videos and other files.

    Returns:
        - None.
    """
    
    # Declare all variables as global
    global MAKE_VIDEO, FILENAME_PREFIX, SESSION_FOLDER, CENTROID_MODEL_BLACK
    global CENTERED_MODEL_BLACK, CENTROID_MODEL_WHITE, CENTERED_MODEL_WHITE
    global ANIPOSE_CALIBRATION_FILE, TOP_CAMERA_NAME, SESSION_NAME
    global INFERENCE_SBATCH_SCRIPT, ANIPOSE_TRIANGULATION_SBATCH_SCRIPT
    global POINT_INDICES, POINTS_TO_EXCLUDE, CONNECTIONS
    
    ### Global variables.
    MAKE_VIDEO = True
    FILENAME_PREFIX = "/mnt/cup/labs/witten/"

    # Models.
    SESSION_FOLDER = session_folder
    CENTROID_MODEL_BLACK = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Production/Bl6/Mine+Misael_Bl6_250710_193204.centroid.n=2104"
    CENTERED_MODEL_BLACK = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Production/Bl6/Mine+Misael_Bl6_250710_221747.centered_instance.n=2104"
    CENTROID_MODEL_WHITE = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Production/SW/Mine+Jiaxuan_SW_250714_091628.centroid.n=4099"
    CENTERED_MODEL_WHITE = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Production/SW/Mine+Jiaxuan_SW_250714_153818.centered_instance.n=4099"

    # Calibration file (file in session folder with "calibration" in the name).
    session_folder_contents = os.listdir(SESSION_FOLDER)
    ANIPOSE_CALIBRATION_FILE = None
    for file in session_folder_contents:
        if "calibration" in file and file.endswith(".toml"):
            ANIPOSE_CALIBRATION_FILE = os.path.join(SESSION_FOLDER, file)
            break
    TOP_CAMERA_NAME = "Camera0"  # The top-down camera name.

    # SLURM scripts.
    SESSION_NAME = SESSION_FOLDER.split("/")[-1]
    INFERENCE_SBATCH_SCRIPT = os.path.join(FILENAME_PREFIX, os.getcwd(), "_sleap_inference.sh")
    ANIPOSE_TRIANGULATION_SBATCH_SCRIPT = os.path.join(FILENAME_PREFIX, os.getcwd(), "_anipose_triangulation.sh")

    # Skeleton points and connections.
    POINT_INDICES = ["Nose", "Ear_R", "Ear_L", "TTI", "TailTip", "Head", "Trunk", "Tail0", "Tail1", "Tail2", "Shoulder_left", "Shoulder_right", "Haunch_left", "Haunch_right", "Neck"]
    POINTS_TO_EXCLUDE = ["Tail0", "Tail1", "Tail2", "TailTip"]
    CONNECTIONS = [["Shoulder_left", "Haunch_left"], ["Haunch_right", "Shoulder_right"], ["Ear_L", "Nose"], ["Ear_R", "Nose"], ["Nose", "Head"], ["Ear_L", "Head"], ["Ear_R", "Head"], ["Shoulder_left", "Neck"], ["Haunch_left", "Trunk"], ["Haunch_right", "Trunk"], ["Shoulder_right", "Neck"], ["TTI", "Tail0"], ["Haunch_left", "TTI"], ["Haunch_right", "TTI"], ["Tail0", "Tail1"], ["Tail1", "Tail2"], ["Tail2", "TailTip"], ["Head", "Neck"], ["Neck", "Trunk"], ["Trunk", "TTI"], ["Shoulder_right", "Shoulder_left"], ["Haunch_left", "Haunch_right"]]

    print(f"Processing session: {SESSION_NAME}")

    run_session() # Run the session processing.

    print(f"Completed processing session: {SESSION_NAME}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python behavior_pipeline.py <session_folder>")
        sys.exit(1)
    
    session_folder = sys.argv[1]
    if not os.path.exists(session_folder):
        print(f"Error: Session folder {session_folder} does not exist")
        sys.exit(1)
    
    setup_session(session_folder)


if __name__ == "__main__":
    main()