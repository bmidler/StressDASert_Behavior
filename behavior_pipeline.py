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

NOTE: sleap tracking is contingent on an SBATCH script for running on spock. This file must be in the
same directory as that SBATCH script.
"""

### Import statements.

import os
import time
import subprocess
import numpy as np

### Global variables.

MAKE_VIDEO = True

FILENAME_PREFIX = "/mnt/cup/labs/witten/"
SESSION_FOLDER = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Data/SleapTrainVideos/2024-09-25/Bl6SW_1/[2024-09-25_13-32-57]-SleapTrain_Bl6SW_1"
CENTROID_MODEL_BLACK = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Production/Bl6/FineTuned_241025_175234.centroid.n=717"
CENTERED_MODEL_BLACK = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Production/Bl6/FineTuned_241027_112222.centered_instance.n=717"
CENTROID_MODEL_WHITE = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Baselines/SW/models/SW_centroid_v1/240927_195723.centroid.n=216"
CENTERED_MODEL_WHITE = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Sleap/Models/Baselines/SW/models/SW_centered_v1/240927_200825.centered_instance.n=216"
ANIPOSE_CALIBRATION_FILE = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Data/SleapTrainVideos/2024-09-25/Bl6SW_1/[2024-09-25_13-32-57]-SleapTrain_Bl6SW_1/calibration-2024-09-25.toml"
SBATCH_SCRIPT = FILENAME_PREFIX + "Ben/Projects/StressDASert/Behavior/Scripts/sleap_inference.sh"

### Function definitions.

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

    print(f"\tFound {len(video_paths)} videos in session folder.")

    ### Submit SBATCH jobs for each video.

    for video_path in video_paths:

        # Get video name being processed.
        video_name = video_path.split("/")[-1]

        # Get names for output and error files.
        error_file = f"{mouse_color}_{video_name}_errors.txt"
        output_file = f"{mouse_color}_{video_name}_outputs.txt"

        # Construct the command to run the SBATCH script with dynamic error and output file paths.
        command = (
            f"sbatch --error=SBATCH_outputs/{error_file} --output=SBATCH_outputs/{output_file} "
            f"{SBATCH_SCRIPT} {video_path} {centroid_model} {centered_model} {mouse_color}"
        )

        # Run the command and capture the output and errors.
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Get the output and error messages.
        output = result.stdout.decode("utf-8")
        error = result.stderr.decode("utf-8")
        
        # Print error message if there is one, otherwise print the job ID.
        if result.returncode != 0:
            print(f"\tError: {error}")
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
            # Check for inference files for black mouse.
            black_inference_file = os.path.join(session_folder, folder, "black_inference.h5")
            if not os.path.exists(black_inference_file):
                all_inference_files_present = False
                break

            # Check for inference files for white mouse.
            white_inference_file = os.path.join(session_folder, folder, "white_inference.h5")
            if not os.path.exists(white_inference_file):
                all_inference_files_present = False
                break

        if not all_inference_files_present:
            print("\tInference files not yet generated. Checking again in 60 seconds.")
            time.sleep(60)  # Wait for 60 seconds before checking again.

        else:
            all_inference_files_present = True


def run_anipose_triangulation():
    """
    Runs 3D triangulation using anipose on the inference files generated from sleap.
    This function uses the anipose_triangulation sbatch script and also runs reprojection.

    NOTE: because sleap anipose isn't able to differentiate the sleap track files for different mice,
    this function creates a temporary sub-folder to put the sleap tracks for the mousenot being processed.

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

    # Move white mouse inference files to the UnusedInference folders.
    for folder in video_folders:
        white_inference_file = os.path.join(SESSION_FOLDER, folder, "white_inference.h5")
        os.rename(white_inference_file, os.path.join(SESSION_FOLDER, folder, "UnusedInference", "white_inference.h5"))

    # Run anipose triangulation for black mouse using anipose_triangulation.sh script.
    print("Running anipose triangulation for black mouse...")
    session_directory = SESSION_FOLDER
    calibration_file = ANIPOSE_CALIBRATION_FILE
    output_filename = "black_triangulated.h5"
    command = (f"sbatch anipose_triangulation.sh {session_directory} {calibration_file} {output_filename}")

    # Run the command and capture the output and errors.
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the output and error messages.
    output = result.stdout.decode("utf-8")
    error = result.stderr.decode("utf-8")

    # Print error message if there is one, otherwise print the job ID.
    if result.returncode != 0:
        print(f"\tError: {error}")
    else:
        # Extract and print the job ID.
        if "\tSubmitted batch job" in output:
            job_id = output.split()[-1]

    print(f"\tRunning triangulation/reprojection for black mouse (Job ID: {job_id})")

    # Move white mouse inference files back to their original locations.
    for folder in video_folders:
        reverted_path = os.path.join(SESSION_FOLDER, folder, "white_inference.h5")
        os.rename(os.path.join(SESSION_FOLDER, folder, "UnusedInference", "white_inference.h5"), reverted_path)

    # Move black mouse inference files to the UnusedInference folders.
    for folder in video_folders:
        black_inference_file = os.path.join(SESSION_FOLDER, folder, "black_inference.h5")
        os.rename(black_inference_file, os.path.join(SESSION_FOLDER, folder, "UnusedInference", "black_inference.h5"))

    # Run anipose triangulation for white mouse using anipose_triangulation.sh script.
    print("Running anipose triangulation for white mouse...")
    session_directory = SESSION_FOLDER
    calibration_file = ANIPOSE_CALIBRATION_FILE
    output_filename = "white_triangulated.h5"
    command = (f"sbatch anipose_triangulation.sh {session_directory} {calibration_file} {output_filename}")

    # Run the command and capture the output and errors.
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the output and error messages.
    output = result.stdout.decode("utf-8")
    error = result.stderr.decode("utf-8")

    # Print error message if there is one, otherwise print the job ID.
    if result.returncode != 0:
        print(f"\tError: {error}")
    else:
        # Extract and print the job ID.
        if "\tSubmitted batch job" in output:
            job_id = output.split()[-1]

    print(f"\tRunning triangulation/reprojection for white mouse (Job ID: {job_id})")

    # Move black mouse inference files back to their original locations.
    for folder in video_folders:
        reverted_path = os.path.join(SESSION_FOLDER, folder, "black_inference.h5")
        os.rename(os.path.join(SESSION_FOLDER, folder, "UnusedInference", "black_inference.h5"), reverted_path)

    # Delete the UnusedInference folders.
    for folder in video_folders:
        os.rmdir(os.path.join(SESSION_FOLDER, folder, "UnusedInference"))

    print("Behavior pipeline completed successfully.")


def make_video():
    """
    Uses the 3D triangulated and 2D reprojected pose files to create a video of the tracks both overlaid on the original videos
    and as a video of only the skeletons.

    Parameters:
        - None.

    Returns: 
        - None.
    """

    # TODO.
    pass


def main():

    ### Check if there is a folder for SBATCH outputs. Make if not.

    if not os.path.exists("SBATCH_outputs"):  # Check if the output folder exists.
        os.makedirs("SBATCH_outputs")  # Create the output folder.

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
    if not os.path.exists(SBATCH_SCRIPT):
        raise FileNotFoundError(f"SBATCH script does not exist: {SBATCH_SCRIPT}")
    
    ### Run sleap inference for each camera view (video) in the session for black and white mice.

    # Black mouse.
    print("Running sleap inference for black mouse...")
    run_sleap_inference(SESSION_FOLDER, CENTROID_MODEL_BLACK, CENTERED_MODEL_BLACK, "black")

    # White mouse.
    print("Running sleap inference for white mouse...")
    run_sleap_inference(SESSION_FOLDER, CENTROID_MODEL_WHITE, CENTERED_MODEL_WHITE, "white")

    ### Wait until all inference is done running.

    print("Waiting for inference to complete...")
    wait_for_inference_to_complete(SESSION_FOLDER)
    print("Inference complete and inference files generated.")

    ### Run anipose triangulation.

    run_anipose_triangulation()

    ### Make video of the tracks if specified.

    if MAKE_VIDEO:
        print("Making video of the tracks...")

        make_video()


if __name__ == "__main__":
    main()
