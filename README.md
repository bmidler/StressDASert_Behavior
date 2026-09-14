# Behavior Pipeline

Multi-camera 3D pose tracking for social-defeat sessions with two mice (one black, one white).
The pipeline takes raw multi-view video for a session, runs SLEAP inference on every camera
view for each mouse, triangulates the 2D poses into a single 3D pose with `sleap-anipose`,
reprojects that 3D pose back into each camera view, and renders QC videos.

Everything runs on a Slurm cluster (spock/della-style). One session is processed per array
task; within a session, the per-camera inference jobs run in parallel on GPUs.

---

## Contents

| File | Role |
| --- | --- |
| `behavior_pipeline.sh` | **Entry point.** Slurm array job — one array task per session in a day directory. |
| `behavior_pipeline.py` | Per-session orchestrator. Submits and waits on all the child jobs, then renders videos. |
| `_sleap_inference.sh` | Child GPU job. SLEAP inference + tracking for **one camera view, one mouse color**. |
| `_anipose_triangulation.sh` | Child CPU job. `slap-triangulate` + `slap-reproject` for **one mouse color** across all views. |
| `fix_calibration_camera_order.py` | **Pre-processing utility.** Run before the pipeline to place a correctly-ordered calibration `.toml` in every session. |

The leading underscore on the two child scripts is just a naming convention — you never submit
them by hand; `behavior_pipeline.py` `sbatch`es them for you.

---

## What the pipeline does

```
behavior_pipeline.sh  (array task i)
        │
        └── behavior_pipeline.py <session_dir>
                │
                ├── reset_session()          delete stale tracks/videos from a previous run
                │
                ├── run_sleap_inference()    sbatch _sleap_inference.sh once per (camera × color)
                │   └── waits until every CameraN/ has a black_*.h5 and a white_*.h5
                │
                ├── run_anipose_triangulation()
                │   ├── hide white files in CameraN/UnusedInference/, triangulate black, restore
                │   └── hide black files, triangulate white, restore, delete UnusedInference/
                │
                └── make_video()             (if MAKE_VIDEO = True)
                    ├── tracked_video.mp4 per camera  (reprojected points on the real footage)
                    ├── topdown_skeleton_video.mp4    (skeletons only, top-down view, white bg)
                    └── 3D_skeleton_video.mp4         (slowly rotating 3D skeleton in a wireframe box)
```

### Stage 1 — SLEAP inference (`_sleap_inference.sh`)

One GPU job per camera view per mouse color, so a 5-camera session submits 10 jobs.
Each job runs a top-down SLEAP model pair (centroid + centered-instance) with
`--max_instances 1` and simple fixed-window tracking, then exports to a classic
analysis HDF5 with MATLAB axis ordering so `sleap-anipose` can read it.

This script targets **SLEAP 1.6 with the PyTorch `sleap-nn` backend**, installed as a `uv` tool:

```bash
uv tool install --python 3.13 "sleap[nn]" --torch-backend cu130
```

It expects the binary at `$HOME/.local/bin/sleap`. Before doing any real work it runs a GPU
probe and distinguishes three cases:

* **GPU works** → proceed.
* **CPU-only torch wheel / broken install** (exit 2) → fail immediately; another node won't help.
* **No usable CUDA device on this node** (exit 3) → report the node and re-queue elsewhere,
  excluding the bad node. Controlled by `RETRY_MODE` (`resubmit`, the default, gets a new job
  ID; `requeue` keeps the same one) and `MAX_GPU_RETRIES` (default 3).

Outputs, written next to the source video:

```
CameraN/black_inference.slp
CameraN/black_inference.analysis.h5
CameraN/white_inference.slp
CameraN/white_inference.analysis.h5
```

### Stage 2 — Triangulation and reprojection (`_anipose_triangulation.sh`)

`sleap-anipose` cannot tell two mice apart in a directory, so `behavior_pipeline.py` works
around this: it temporarily moves the white mouse's files into `CameraN/UnusedInference/`,
triangulates the black mouse, moves them back, and repeats with the colors reversed.

The job runs in the `sleap-anipose` conda environment and calls, with up to 100 retries each:

```
slap-triangulate --p2d <session> --calib <calibration.toml> --fname <color>_triangulated.h5 \
                 --scale_smooth 4 --n_deriv_smooth 2 --reproj_loss l2 --reproj_error_threshold 15
slap-reproject   --p3d <color>_triangulated.h5 --calib <calibration.toml> \
                 --fname <color>_triangulated_reprojected.h5
```

Outputs, written to the session root:

```
black_triangulated.h5                 # 3D pose  [frames, instances, points, xyz]
black_triangulated_reprojected.h5     # per-camera 2D  (one dataset per CameraN key)
white_triangulated.h5
white_triangulated_reprojected.h5
```

### Stage 3 — QC videos (`behavior_pipeline.py`)

Skeleton definition, colors and excluded points are globals at the bottom of
`behavior_pipeline.py` (`POINT_INDICES`, `CONNECTIONS`, `POINTS_TO_EXCLUDE`,
`AGGRESSOR_RGB`, `EXPERIMENTAL_RGB`). Tail points are excluded from the rendering by default.
Bounding boxes are computed from the 1st–99th percentile of the tracked points so a few
outlier frames don't blow up the scene.

---

## Data organization

The pipeline is strict about directory layout. A **day directory** contains one subdirectory
per session, plus at least one calibration session:

```
Day6/
├── calibration_2/                         # name must contain "calibration"
│   ├── calibration.toml                   # the master calibration file for this day
│   ├── Camera0/
│   │   └── calibration_images/
│   │       └── ...-s#0955-Camera0-calibration.mp4
│   ├── Camera1/ ...
│   └── Camera4/ ...
│
├── Mouse01_Day6_Defeat/                   # an experimental session
│   ├── calibration.toml                   # per-session copy, camera order corrected
│   ├── Camera0/
│   │   └── ...-s#0955-Camera0-....mp4     # exactly one source .mp4
│   ├── Camera1/
│   │   └── ...-s#1183-Camera1-....mp4
│   └── ...
│
├── Mouse02_Day6_Defeat/
└── ...
```

Requirements, in order of how often they bite:

1. **Camera directories must be named `Camera0`, `Camera1`, …** — contiguous from 0. The
   pipeline finds them by the literal substring `Camera`, and `fix_calibration_camera_order.py`
   matches `^Camera(\d+)$` exactly.
2. **Exactly one source `.mp4` per camera directory.** Several places in the code take the
   first `.mp4` found (filtering out names containing `track`), so a second video is a coin flip.
   File *names* are otherwise arbitrary as far as the pipeline is concerned.
3. **Every session directory needs a calibration file** whose name contains `calibration` and
   ends in `.toml`, sitting in the session root. The pipeline raises `FileNotFoundError`
   immediately if it can't find one. It must be *this session's* file — see the next section.
4. **Session directories must not have `calibration` in their name.** `behavior_pipeline.sh`
   uses that substring to decide what to skip, so a session named e.g. `Mouse01_postcalibration`
   will be silently dropped from the array.
5. **`TOP_CAMERA_NAME` must match a real camera directory.** It's hard-coded to `Camera0` in
   `setup_session()`; change it there if your overhead view lives elsewhere.
6. **An `SBATCH_outputs/` directory must exist** in the submission directory. Slurm will not
   create it and the job dies before producing any log if it's missing.
7. For `fix_calibration_camera_order.py` only: **video filenames must contain the camera
   serial as `s#SERIAL`** (e.g. `s#0955`), and the calibration session's videos should live
   in `CameraN/calibration_images/`.

---

## Calibration: run `fix_calibration_camera_order.py` first

This is the step that's easiest to skip and most expensive to get wrong.

The acquisition software names cameras `Camera0..Camera4` by **USB enumeration order**, which
is not stable across reboots. The physical camera in slot `Camera2` during calibration may be
in slot `Camera4` during an experimental session recorded later that day. The calibration
`.toml` is written in terms of the calibration session's slot order, so copying it verbatim
into an experimental session silently attaches the wrong intrinsics and extrinsics to every
camera — triangulation still "succeeds", it's just wrong.

`fix_calibration_camera_order.py` fixes this:

1. Builds `{CameraN → serial}` for the calibration session from the `s#SERIAL` in the
   filenames under `CameraN/calibration_images/`.
2. Builds the same map for each experimental session from the filenames in `CameraN/`.
3. Where the orders differ, permutes the camera blocks of the master `.toml` into the
   session's slot order, relabels each block's `name` field, and writes the corrected file
   into the session (backing up anything already there as `*.orig-<timestamp>.bak`).
4. Writes `camera_order_report.csv` in the day directory listing every session/slot/serial
   and whether it was `match`, `remapped`, or `error`.

Sessions that have no calibration file get the master copied in (unless `--no-copy-missing`).
The corrected file is always regenerated from the master, so **re-running is safe and
idempotent** — it will not double-permute an already-corrected session.

```bash
# Preview, no writes:
python fix_calibration_camera_order.py \
    --day-dir  ../Data/Defeat-Cohorts/Cohort-B/Cohort_B-CSDS/Day6 \
    --calib-dir ../Data/Defeat-Cohorts/Cohort-B/Cohort_B-CSDS/Day6/calibration_2 \
    --dry-run

# Apply:
python fix_calibration_camera_order.py \
    --day-dir  ../Data/Defeat-Cohorts/Cohort-B/Cohort_B-CSDS/Day6 \
    --calib-dir ../Data/Defeat-Cohorts/Cohort-B/Cohort_B-CSDS/Day6/calibration_2
```

Run with no arguments and it opens two tkinter folder pickers instead (day directory, then
calibration session) — convenient on a desktop, useless over a plain SSH session.

Read the printed output before moving on. A session is skipped, not corrected, if its camera
serials don't match the calibration session's set, if a `Camera*` dir has no videos, or if two
`Camera*` dirs report the same serial.

---

## Running the pipeline

From the directory containing the scripts:

```bash
mkdir -p SBATCH_outputs

sbatch --array=0-23 behavior_pipeline.sh ../Data/Defeat-Cohorts/Cohort-B/Cohort_B-CSDS/Day6
```

Where `23` is **the number of sessions minus 1**. Count the non-calibration subdirectories:

```bash
ls -d ../Data/Defeat-Cohorts/Cohort-B/Cohort_B-CSDS/Day6/*/ | grep -vi calibration | wc -l
```

Notes on the invocation:

* **Run it from the script directory.** The child scripts are resolved against `os.getcwd()`,
  and inference output paths are passed to Slurm relative to the submission directory. A
  relative day-directory path (`../Data/...`) is the expected usage.
* Array indices are 0-based and map to the sessions in the order `bash` globs them
  (alphabetical). Out-of-range indices exit with an error rather than doing nothing silently.
* To re-run a single session, pass just its index: `--array=7`.
* Head-job resources: 1 CPU, 32 GB, 48 h wall clock on the `witten` partition. The head job
  mostly sleeps while child jobs run, but it holds that allocation the whole time — a full day
  of sessions submits one long-lived CPU job per session plus bursts of GPU jobs.
* Per-session logs land in `SBATCH_outputs/<SESSION_NAME>/`, with the array job's own logs in
  `SBATCH_outputs/` itself.

### Environments

Three separate environments are involved, and each script activates its own:

| Script | Environment | Needs |
| --- | --- | --- |
| `behavior_pipeline.sh` → `.py` | conda `general` | `numpy`, `opencv-python` (`cv2`), `h5py`, `matplotlib` |
| `_sleap_inference.sh` | `uv` tool at `~/.local/bin/sleap` | `sleap[nn]` 1.6, CUDA-enabled torch |
| `_anipose_triangulation.sh` | conda `sleap-anipose` | `slap-triangulate`, `slap-reproject` |

All three load `anacondapy/2023.07-cuda` as their module base.

---

## Configuration

Most knobs are hard-coded as globals in `setup_session()` near the bottom of
`behavior_pipeline.py`:

| Variable | Meaning |
| --- | --- |
| `MAKE_VIDEO` | Set `False` to stop after triangulation and skip all QC video rendering. |
| `FILENAME_PREFIX` | Cluster mount root, `/mnt/cup/labs/witten/`. |
| `CENTROID_MODEL_BLACK` / `CENTERED_MODEL_BLACK` | Black (Bl6) mouse SLEAP model pair. |
| `CENTROID_MODEL_WHITE` / `CENTERED_MODEL_WHITE` | White (SW) mouse SLEAP model pair. |
| `TOP_CAMERA_NAME` | Overhead camera directory, used for the top-down skeleton video. |
| `POINT_INDICES` | Node order in the SLEAP skeleton — **must match the trained models**. |
| `POINTS_TO_EXCLUDE` | Nodes dropped from rendering and from bounding-box computation (tail points). |
| `CONNECTIONS` | Skeleton edges drawn in the QC videos. |
| `AGGRESSOR_RGB` / `EXPERIMENTAL_RGB` | Overlay colors, in **BGR** order (OpenCV). |

Triangulation smoothing parameters (`--scale_smooth`, `--n_deriv_smooth`, `--reproj_loss`,
`--reproj_error_threshold`) live in `_anipose_triangulation.sh`. Inference parameters
(`--batch_size`, `--peak_threshold`, `--tracking_window_size`) live in `_sleap_inference.sh`.

Changing models means changing `POINT_INDICES` to match the new skeleton — the code indexes
tracks by position in that list, so a mismatch produces plausible-looking but scrambled videos
rather than an error.

---

## Outputs per session

```
<session>/
├── black_triangulated.h5
├── black_triangulated_reprojected.h5
├── white_triangulated.h5
├── white_triangulated_reprojected.h5
├── topdown_skeleton_video.mp4
├── 3D_skeleton_video.mp4
└── CameraN/
    ├── black_inference.slp
    ├── black_inference.analysis.h5
    ├── white_inference.slp
    ├── white_inference.analysis.h5
    └── tracked_video.mp4
```

Re-running a session calls `reset_session()` first, which deletes the triangulated/reprojected
h5s, the skeleton videos, every `.h5`/`.slp` in each camera directory, and any leftover
`UnusedInference/` folders. Anything you want to keep should be copied out first.

---

## Troubleshooting

**The head job hangs forever after submitting inference.**
`wait_for_inference_to_complete()` polls until every camera directory contains both a
`black*.h5` and a `white*.h5`. If an inference job died, that file never appears and the
loop spins indefinitely (it has no sleep and no timeout, so it will also peg a core). Check
`SBATCH_outputs/<SESSION_NAME>/black_<video>_errors.txt` and cancel the head job manually.

**Inference jobs keep bouncing between nodes.**
That's `_sleap_inference.sh` doing its job — a node claimed a GPU that torch couldn't use.
After `MAX_GPU_RETRIES` (3) it gives up and lists the bad nodes in its log. If the probe exits
with code 2 instead, it's a broken *install* (usually a CPU-only torch wheel) and no amount of
retrying will help.

**`--dependency=afterok` chains break on GPU retries.**
In the default `resubmit` mode, a job that hands off to a replacement exits non-zero on
purpose. Use `RETRY_MODE=requeue` if you need the job ID to survive.

**Triangulation succeeds but the 3D pose is nonsense.**
Almost always a calibration camera-order problem. Check `camera_order_report.csv` and confirm
the session's `.toml` was corrected, not just copied.

**Videos aren't produced but tracking is fine.**
Video generation is wrapped in `try/except` and prints `Error making video…` rather than
failing the job — look for that line in the session log.
