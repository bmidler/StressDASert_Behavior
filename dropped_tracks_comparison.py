"""
dropped_tracks_comparison.py

Makes figure for campring frame tracking (versus drops) between each camera's individual tracking and the reprojection.
"""

### Imports.

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

### Global variables.

# Global path to the session directory.
SESSION_DIRECTORY = "/mnt/cup/labs/witten/Ben/Projects/StressDASert/Behavior/Data/SleapTrainVideos/2024-09-25/Bl6SW_3/[2024-09-25_14-00-35]-SleapTrain_Bl6SW_3"
OUTPUT = "../Figures/ReprojectionComparison"

