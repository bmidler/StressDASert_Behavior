#!/bin/bash

#SBATCH -J sleap_inference
#SBATCH --partition=witten,all
#SBATCH -c 7
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH --requeue

# ---------------------------------------------------------------------------
# SLEAP inference for a single camera view (one mouse color).
#
# For SLEAP 1.6 (PyTorch "sleap-nn" backend), installed as a uv tool:
#   uv tool install --python 3.13 "sleap[nn]" --torch-backend cu130
#
# If the allocated node claims a GPU but torch cannot use it, this script
# reports the bad node and re-queues itself elsewhere instead of dying.

set -uo pipefail

# --- Retry-on-bad-GPU configuration ----------------------------------------
# resubmit : sbatch a brand-new job with --exclude=<bad nodes>.  Portable, no
#            special scontrol permissions, but the job gets a NEW job ID.
# requeue  : scontrol requeue the same job, appending the bad node to its
#            ExcNodeList.  Keeps the SAME job ID, so `sbatch --wait` and
#            --dependency=afterok chains in behavior_pipeline.py keep working.
#            Requires that users may update ExcNodeList on their own jobs.
RETRY_MODE="${RETRY_MODE:-resubmit}"
MAX_GPU_RETRIES="${MAX_GPU_RETRIES:-3}"

# Carried across resubmissions via the exported environment.
GPU_ATTEMPT="${GPU_ATTEMPT:-0}"
GPU_BAD_NODES="${GPU_BAD_NODES:-}"

# Captured before anything can fail, so a retry can pass them on unchanged.
JOB_ARGS=("$@")

# ---------------------------------------------------------------------------
# gpu_retry <reason>
#   Report a node with a non-functional GPU, then put the job back in the
#   queue with that node excluded.  Never returns.
# ---------------------------------------------------------------------------
gpu_retry() {
    local reason="$1"
    local me="${SLURMD_NODENAME:-unknown}"

    echo
    echo "==========================================================="
    echo "  GPU UNUSABLE -- node likely broken"
    echo "  Node:   $me"
    echo "  Job:    ${SLURM_JOB_ID:-none}"
    echo "  Reason: $reason"
    echo "==========================================================="

    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "Not running under Slurm; nothing to re-queue."
        exit 1
    fi
    if [ "$me" = "unknown" ]; then
        echo "Could not determine the node name; refusing to retry blindly."
        exit 1
    fi

    if [ "$RETRY_MODE" = "requeue" ]; then
        _gpu_requeue "$me"
    else
        _gpu_resubmit "$me"
    fi
}

# --- Mode 1: scontrol requeue (same job ID) --------------------------------
_gpu_requeue() {
    local me="$1"
    local attempt="${SLURM_RESTART_COUNT:-0}"

    if [ "$attempt" -ge "$MAX_GPU_RETRIES" ]; then
        echo "Already re-queued $attempt time(s) (max $MAX_GPU_RETRIES)."
        echo "Every node tried so far had a broken GPU. Giving up."
        exit 1
    fi

    # Accumulate exclusions so we never revisit a node we already rejected.
    local prev exclude
    prev=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
           | tr ' ' '\n' | sed -n 's/^ExcNodeList=//p' | head -1)
    [ "$prev" = "(null)" ] && prev=""
    exclude="${prev:+$prev,}$me"

    echo "Re-queueing job $SLURM_JOB_ID (attempt $((attempt + 1))/$MAX_GPU_RETRIES)"
    echo "Excluding nodes: $exclude"

    # requeuehold -> update -> release.  A RUNNING job will not accept an
    # ExcNodeList change; a held PENDING one will.
    if scontrol requeuehold "$SLURM_JOB_ID"; then
        if ! scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$exclude"; then
            echo "WARNING: could not set ExcNodeList."
            echo "         The job may be scheduled back onto $me."
        fi
        scontrol release "$SLURM_JOB_ID"
    else
        echo "ERROR: scontrol requeuehold failed. Falling back to resubmit."
        _gpu_resubmit "$me"
    fi

    # Slurm tears this step down asynchronously; do not fall through into
    # a multi-hour CPU run while we wait for the signal.
    sleep 60
    exit 1
}

# --- Mode 2: sbatch a fresh job (new job ID) -------------------------------
_gpu_resubmit() {
    local me="$1"

    if [ "$GPU_ATTEMPT" -ge "$MAX_GPU_RETRIES" ]; then
        echo "Already resubmitted $GPU_ATTEMPT time(s) (max $MAX_GPU_RETRIES)."
        echo "Bad nodes so far: ${GPU_BAD_NODES:-none}. Giving up."
        exit 1
    fi

    local exclude="${GPU_BAD_NODES:+$GPU_BAD_NODES,}$me"

    # $0 inside a batch job points at Slurm's spooled copy of the script,
    # which is deleted when the job ends -- ask the controller for the real
    # path instead.
    local script
    script=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
             | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
    if [ ! -f "$script" ]; then
        script="${SLURM_SUBMIT_DIR:-.}/$(basename "$0")"
    fi
    if [ ! -f "$script" ]; then
        echo "ERROR: cannot locate this script to resubmit it (tried: $script)."
        exit 1
    fi

    echo "Resubmitting (attempt $((GPU_ATTEMPT + 1))/$MAX_GPU_RETRIES)"
    echo "Script:          $script"
    echo "Excluding nodes: $exclude"

    # Subshell: clear the inherited SLURM_* variables first.  Slurm's
    # precedence is command line > environment > #SBATCH directives, so
    # leftovers like SLURM_MEM_PER_NODE would silently override the
    # directives above in the child job.  Comma-separated values are passed
    # through the environment rather than through --export=VAR=..., which
    # splits on commas.
    (
        unset "${!SLURM_@}"
        export GPU_ATTEMPT=$((GPU_ATTEMPT + 1))
        export GPU_BAD_NODES="$exclude"
        export RETRY_MODE MAX_GPU_RETRIES
        sbatch --exclude="$exclude" --export=ALL "$script" "${JOB_ARGS[@]}"
    )

    if [ $? -ne 0 ]; then
        echo "ERROR: sbatch failed; the work has NOT been rescheduled."
        exit 1
    fi

    # Non-zero on purpose: this job did no work.  Watch out if anything
    # upstream chains on it with --dependency=afterok (see RETRY_MODE).
    exit 1
}

# --- Environment -----------------------------------------------------------
# Absolute path so the job never depends on PATH surviving submission.
# If you redirected uv with UV_TOOL_BIN_DIR, point this there instead.
SLEAP="$HOME/.local/bin/sleap"

if [ ! -x "$SLEAP" ]; then
    echo "ERROR: sleap not found at $SLEAP"
    echo "       Check: uv tool list"
    exit 1
fi

echo "SLEAP binary:     $SLEAP"
"$SLEAP" doctor          # Package versions + GPU detection. No `|| true`.
echo

# --- Fail fast if the GPU is not usable ------------------------------------
SLEAP_PY=$(head -1 "$SLEAP" | sed 's|^#!||' | tr -d '"')

echo "Job ID:               ${SLURM_JOB_ID:-none}"
echo "Node:                 ${SLURMD_NODENAME:-unknown}"
echo "Attempt:              $((GPU_ATTEMPT + 1)) (bad nodes: ${GPU_BAD_NODES:-none})"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS:       ${SLURM_JOB_GPUS:-<unset>}"
nvidia-smi -L 2>&1 || echo "nvidia-smi unavailable on this node"

# Exit codes from the probe below:
#   0  GPU works
#   2  broken *install* (CPU-only wheel) -- another node will not help
#   3  broken *node* (no device, or the device errors out) -- retry elsewhere
"$SLEAP_PY" - <<'PYEOF'
import sys

try:
    import torch
except Exception as e:
    print(f"ABORT: cannot import torch: {e}", file=sys.stderr)
    sys.exit(2)

print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
      f"available: {torch.cuda.is_available()}")

if torch.__version__.endswith("+cpu"):
    print("ABORT: CPU-only torch wheel installed. Reinstall with "
          "--torch-backend cu128 (or cu130 if driver >= 580).", file=sys.stderr)
    sys.exit(2)

if not torch.cuda.is_available():
    print("ABORT: torch sees no CUDA device. Either this node's GPU is "
          "broken, or the driver is too old for the installed wheel "
          "(cu130 needs driver 580+).", file=sys.stderr)
    sys.exit(3)

try:
    a = torch.randn(1024, 1024, device="cuda")
    torch.mm(a, a)
    torch.cuda.synchronize()
    print(f"GPU ready: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"ABORT: GPU visible but unusable: {e}", file=sys.stderr)
    sys.exit(3)
PYEOF

gpu_rc=$?
case "$gpu_rc" in
    0)
        ;;
    2)
        echo "This is an installation problem, not a node problem."
        echo "Retrying on another node would fail the same way. Not re-queueing."
        exit 1
        ;;
    *)
        gpu_retry "torch could not use a GPU here (probe exit $gpu_rc)"
        ;;
esac
echo

# --- Arguments (passed by behavior_pipeline.py, order unchanged) -----------
video=$1            # Path to the camera-view .mp4
centroid_model=$2   # Path to the centroid model directory
centered_model=$3   # Path to the centered-instance model directory
tracks_base=$4      # Output basename, e.g. .../Camera0/black_inference (no ext)

slp_output="${tracks_base}.slp"
analysis_output="${tracks_base}.analysis.h5"

echo "Video:            $video"
echo "Centroid model:   $centroid_model"
echo "Centered model:   $centered_model"
echo "SLP output:       $slp_output"
echo "Analysis output:  $analysis_output"
echo

# --- Inference + tracking --------------------------------------------------
# Flag translation from the old TensorFlow `sleap-track` call:
#   --model / --model                     -> -m / -m              (unchanged)
#   --batch_size 16                       -> --batch_size 16      (unchanged)
#   --peak_threshold 0.3                  -> --peak_threshold 0.3 (unchanged)
#   --tracking.tracker simple             -> --tracking           (fixed_window default)
#   --tracking.track_window 3             -> --tracking_window_size 3
#   --tracking.similarity instance        -> --features keypoints (default)
#   --tracking.clean_instance_count 1     -> --max_instances 1
#   --tracking.post_connect_single_breaks 1 -> REMOVED (no sleap-nn equivalent)
#
# --device cuda (not auto): auto silently falls back to CPU when no GPU is
# visible, which is exactly how a broken install hides for 12 hours.
"$SLEAP" track \
    -i "$video" \
    -m "$centroid_model" \
    -m "$centered_model" \
    -o "$slp_output" \
    --device cuda \
    --batch_size 16 \
    --peak_threshold 0.3 \
    --max_instances 1 \
    --tracking \
    --tracking_window_size 3

track_rc=$?
if [ "$track_rc" -ne 0 ]; then
    # A GPU that passed the probe can still fall over mid-run (ECC errors, a
    # wedged device, another job's OOM). Treat that as a bad node too, but
    # only when the GPU is visibly gone -- otherwise it is a real SLEAP error
    # and re-queueing would just burn hours repeating it.
    if ! nvidia-smi -L >/dev/null 2>&1; then
        gpu_retry "GPU disappeared during inference (sleap track exit $track_rc)"
    fi
    echo "SLEAP inference failed (exit $track_rc)."
    exit 1
fi

# --- Convert .slp -> classic analysis .h5 (input for sleap-anipose) --------
# `matlab` axis ordering keeps the file byte-compatible with sleap-anipose /
# MATLAB -- the same layout the old `sleap-convert --format analysis` wrote.
"$SLEAP" export "$slp_output" -o "$analysis_output" --h5-dim-order matlab

if [ $? -ne 0 ]; then
    echo "Analysis HDF5 export failed."
    exit 1
fi

echo "SLEAP inference completed successfully: $analysis_output"
echo "~~~All done!~~~"
