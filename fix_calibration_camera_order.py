#!/usr/bin/env python3
"""
fix_calibration_camera_order.py

Correct the per-session calibration .toml files so that the intrinsics/extrinsics
attached to each "CameraN" slot belong to the physical camera (identified by its
s#SERIAL) that actually occupied slot CameraN during that experimental session.

Background
----------
Cameras are named Camera0..Camera4 by the acquisition software according to the
order in which they enumerate on the USB bus.  That order is not stable across
reboots, so the physical camera (serial number) sitting in slot "Camera2" during
a calibration session may be in slot "Camera4" during a later experimental
session.  The calibration .toml is written in terms of the *calibration
session's* slot order, so copying it verbatim into an experimental session with
a different enumeration order silently mis-assigns every camera's parameters.

What this does
--------------
1. Reads the calibration session and builds  {CameraN -> serial}  from the mp4
   filenames in  CameraN/calibration_images/.
2. For every experimental session in the day directory, builds the same map from
   the mp4 filenames in  CameraN/.
3. Where the two orders differ, writes a corrected calibration .toml into the
   experimental session: the block for the physical camera with serial S is
   moved into the slot that S occupied during *that session*, and its `name`
   field is relabelled to match.
4. Backs up whatever calibration file was already there, and writes a CSV report.

The corrected file is always regenerated from the master calibration file in the
calibration session directory, so re-running the script is safe and idempotent
(it will not double-permute an already-corrected session).

Usage
-----
    python fix_calibration_camera_order.py
        -> opens two folder pickers: the day directory, then the calibration
           session directory used for that day.

    python fix_calibration_camera_order.py --day-dir /path/to/Day1 \
        --calib-dir /path/to/Day1/calibration_2 [--dry-run]

Options
-------
    --dry-run          Report what would change; write nothing.
    --no-copy-missing  Skip sessions that have no calibration .toml instead of
                       copying one in from the calibration session.
    --report PATH      Where to write the CSV report
                       (default: <day-dir>/camera_order_report.csv).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Filename / directory patterns
# ----------------------------------------------------------------------------

# Matches "s#0955" in ...-{qp24-020fps}-s#0955-Camera0-calibration.mp4
SERIAL_RE = re.compile(r"s#([A-Za-z0-9]+)")
# Matches "Camera0" as a hyphen-delimited field in the same filename
CAMLABEL_RE = re.compile(r"(?:^|[-_])(Camera(\d+))(?=[-_.]|$)")
# Matches a camera directory name, e.g. "Camera3"
CAMDIR_RE = re.compile(r"^Camera(\d+)$")
# A directory that is a calibration session rather than an experimental one
CALIB_DIR_RE = re.compile(r"^calibration", re.IGNORECASE)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


# ----------------------------------------------------------------------------
# Minimal TOML read/write that round-trips the anipose/aniposelib style
# ----------------------------------------------------------------------------

def toml_load(path: Path) -> "OrderedDict[str, OrderedDict]":
    """Parse a calibration .toml into an ordered {section: {key: value}} dict.

    Deliberately hand-rolled rather than using tomllib so that section and key
    order are preserved exactly, which keeps the rewritten file diff-friendly
    against the original.
    """
    text = path.read_text(encoding="utf-8")
    data: "OrderedDict[str, OrderedDict]" = OrderedDict()
    current: "OrderedDict[str, object] | None" = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("[") and raw.endswith("]") and "=" not in raw:
            section = raw[1:-1].strip()
            current = data.setdefault(section, OrderedDict())
            continue
        if "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key, value = key.strip(), value.strip()
        # Continue reading while brackets are unbalanced (multi-line arrays)
        while value.count("[") > value.count("]") and i < len(lines):
            value += " " + lines[i].strip()
            i += 1
        if current is None:
            current = data.setdefault("", OrderedDict())
        try:
            # TOML scalars/arrays used in these files are valid Python literals
            current[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            low = value.lower()
            if low in ("true", "false"):
                current[key] = (low == "true")
            else:
                current[key] = value.strip('"')
    return data


def _fmt_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        return "[ " + ", ".join(_fmt_value(v) for v in value) + ",]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_fmt_value(v)}" for k, v in value.items()) + " }"
    return json.dumps(str(value))


def toml_dumps(data, newline: str = "\n") -> str:
    out = []
    for section, body in data.items():
        if section:
            out.append(f"[{section}]")
        for key, value in body.items():
            out.append(f"{key} = {_fmt_value(value)}")
        out.append("")
    return newline.join(out)


def detect_newline(path: Path) -> str:
    raw = path.read_bytes()
    return "\r\n" if b"\r\n" in raw else "\n"


# ----------------------------------------------------------------------------
# Cataloguing serial <-> camera-slot pairings
# ----------------------------------------------------------------------------

class SessionError(Exception):
    """A session could not be catalogued or corrected."""


def camera_dirs(session_dir: Path):
    """Yield (index, path) for Camera0, Camera1, ... subdirectories, in order."""
    found = []
    for child in sorted(session_dir.iterdir()):
        if not child.is_dir():
            continue
        m = CAMDIR_RE.match(child.name)
        if m:
            found.append((int(m.group(1)), child))
    found.sort(key=lambda t: t[0])
    return found


def _videos_in(directory: Path, recursive: bool):
    if not directory.is_dir():
        return []
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def serial_from_camera_dir(cam_dir: Path, is_calibration: bool, warnings: list) -> str:
    """Read the camera serial out of the video filenames in one Camera* dir."""
    if is_calibration:
        videos = _videos_in(cam_dir / "calibration_images", recursive=False)
        if not videos:
            videos = _videos_in(cam_dir, recursive=True)
    else:
        videos = _videos_in(cam_dir, recursive=False)
        if not videos:
            videos = _videos_in(cam_dir, recursive=True)

    if not videos:
        raise SessionError(f"no video files found under {cam_dir}")

    serials, mismatched_labels = set(), set()
    for video in videos:
        sm = SERIAL_RE.search(video.name)
        if not sm:
            warnings.append(f"no s#SERIAL in filename: {video.name}")
            continue
        serials.add(sm.group(1))
        lm = CAMLABEL_RE.search(video.name)
        if lm and lm.group(1) != cam_dir.name:
            mismatched_labels.add(f"{video.name} (in {cam_dir.name}/)")

    if not serials:
        raise SessionError(f"no s#SERIAL found in any filename under {cam_dir}")
    if len(serials) > 1:
        raise SessionError(
            f"conflicting serials {sorted(serials)} in {cam_dir}"
        )
    for bad in sorted(mismatched_labels):
        warnings.append(f"filename camera label disagrees with its directory: {bad}")
    return serials.pop()


def catalogue_session(session_dir: Path, is_calibration: bool):
    """Return ({CameraN: serial}, warnings) for one session directory."""
    warnings: list = []
    cams = camera_dirs(session_dir)
    if not cams:
        raise SessionError(f"no Camera* subdirectories in {session_dir}")
    mapping: "OrderedDict[str, str]" = OrderedDict()
    for _, cam_dir in cams:
        mapping[cam_dir.name] = serial_from_camera_dir(cam_dir, is_calibration, warnings)
    dupes = [s for s in set(mapping.values()) if list(mapping.values()).count(s) > 1]
    if dupes:
        raise SessionError(f"serial(s) {sorted(dupes)} appear in more than one Camera* dir")
    return mapping, warnings


# ----------------------------------------------------------------------------
# Calibration file handling
# ----------------------------------------------------------------------------

def find_calibration_toml(directory: Path):
    """Find the calibration*.toml in a session directory (not a backup)."""
    hits = [
        p for p in sorted(directory.glob("*.toml"))
        if p.name.lower().startswith("calibration") and ".orig" not in p.name
    ]
    if not hits:
        return None
    if len(hits) > 1:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def camera_index(label: str) -> int:
    m = CAMDIR_RE.match(label)
    return int(m.group(1)) if m else 10**6


def build_corrected_toml(master, cal_map, exp_map):
    """Permute the master calibration data into the experimental camera order.

    cal_map / exp_map are {CameraN: serial}.  Returns (new_data, changed).
    """
    cam_sections = [s for s in master if s.lower().startswith("cam")]
    other_sections = [s for s in master if s not in cam_sections]

    cal_labels = sorted(cal_map, key=camera_index)
    exp_labels = sorted(exp_map, key=camera_index)

    if set(cal_map.values()) != set(exp_map.values()):
        only_cal = sorted(set(cal_map.values()) - set(exp_map.values()))
        only_exp = sorted(set(exp_map.values()) - set(cal_map.values()))
        raise SessionError(
            "camera serials differ from the calibration session "
            f"(only in calibration: {only_cal}; only in session: {only_exp})"
        )
    if len(cam_sections) != len(cal_labels):
        raise SessionError(
            f"calibration file has {len(cam_sections)} camera blocks but the "
            f"calibration session has {len(cal_labels)} cameras"
        )

    # Locate each physical camera's block in the master file.  Prefer matching
    # on the block's `name` field; fall back to positional order if the names
    # are absent or unexpected.
    name_to_section = {}
    for section in cam_sections:
        name = master[section].get("name")
        if isinstance(name, str):
            name_to_section[name] = section
    use_names = all(label in name_to_section for label in cal_labels)

    serial_to_section = {}
    for pos, label in enumerate(cal_labels):
        section = name_to_section[label] if use_names else cam_sections[pos]
        serial_to_section[cal_map[label]] = section

    new_data: "OrderedDict[str, OrderedDict]" = OrderedDict()
    changed = False
    for pos, section in enumerate(cam_sections):
        label = exp_labels[pos]
        source_section = serial_to_section[exp_map[label]]
        body = OrderedDict(master[source_section])
        if "name" in body:
            body["name"] = label
        else:
            body = OrderedDict([("name", label)] + list(body.items()))
        new_data[section] = body
        if source_section != section:
            changed = True

    for section in other_sections:
        new_data[section] = OrderedDict(master[section])

    return new_data, changed


def backup_path(target: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return target.with_name(f"{target.name}.orig-{stamp}.bak")


# ----------------------------------------------------------------------------
# Folder pickers
# ----------------------------------------------------------------------------

def pick_directories(day_dir, calib_dir):
    """Fill in whichever directories were not supplied on the command line."""
    if day_dir and calib_dir:
        return Path(day_dir), Path(calib_dir)
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        sys.exit(
            "tkinter is not available, so the graphical folder picker cannot open.\n"
            "Pass the directories explicitly instead:\n"
            "  python fix_calibration_camera_order.py --day-dir <DAY> --calib-dir <CALIB>"
        )

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    if not day_dir:
        messagebox.showinfo(
            "Select day directory",
            "Select the DAY directory containing all experimental and "
            "calibration sessions.",
        )
        day_dir = filedialog.askdirectory(title="Select the day directory")
        if not day_dir:
            root.destroy()
            sys.exit("No day directory selected; nothing to do.")

    if not calib_dir:
        messagebox.showinfo(
            "Select calibration session",
            "Now select the CALIBRATION SESSION directory that produced the "
            "calibration file used for this day.",
        )
        calib_dir = filedialog.askdirectory(
            title="Select the calibration session used for this day",
            initialdir=day_dir,
        )
        if not calib_dir:
            root.destroy()
            sys.exit("No calibration session selected; nothing to do.")

    root.destroy()
    return Path(day_dir), Path(calib_dir)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fix per-session calibration .toml camera ordering."
    )
    parser.add_argument("--day-dir", help="Day directory (skips the folder picker)")
    parser.add_argument("--calib-dir", help="Calibration session directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report changes without writing any files")
    parser.add_argument("--no-copy-missing", action="store_true",
                        help="Do not copy a calibration file into sessions that lack one")
    parser.add_argument("--report", help="Path for the CSV report")
    args = parser.parse_args(argv)

    day_dir, calib_dir = pick_directories(args.day_dir, args.calib_dir)
    day_dir = day_dir.resolve()
    calib_dir = calib_dir.resolve()

    if not day_dir.is_dir():
        sys.exit(f"Day directory does not exist: {day_dir}")
    if not calib_dir.is_dir():
        sys.exit(f"Calibration session directory does not exist: {calib_dir}")

    print(f"Day directory:        {day_dir}")
    print(f"Calibration session:  {calib_dir}")
    if args.dry_run:
        print("Mode:                 DRY RUN (no files will be written)")
    print()

    # --- Master calibration file -------------------------------------------
    master_toml = find_calibration_toml(calib_dir)
    if master_toml is None:
        sys.exit(f"No calibration*.toml found in {calib_dir}")
    newline = detect_newline(master_toml)
    master = toml_load(master_toml)
    print(f"Master calibration file: {master_toml.name}")

    # --- Catalogue the calibration session ---------------------------------
    try:
        cal_map, cal_warnings = catalogue_session(calib_dir, is_calibration=True)
    except SessionError as exc:
        sys.exit(f"Could not catalogue the calibration session: {exc}")
    for w in cal_warnings:
        print(f"  ! {w}")
    print("Calibration camera order:")
    for label, serial in cal_map.items():
        print(f"  {label:<9} s#{serial}")
    print()

    serial_to_cal_label = {s: l for l, s in cal_map.items()}

    # --- Walk the experimental sessions ------------------------------------
    sessions = [
        d for d in sorted(day_dir.iterdir())
        if d.is_dir()
        and d.resolve() != calib_dir
        and not CALIB_DIR_RE.match(d.name)
    ]
    if not sessions:
        print("No experimental session directories found in the day directory.")

    rows = []
    n_ok = n_fixed = n_failed = 0

    for session in sessions:
        print(f"[{session.name}]")
        try:
            exp_map, warnings = catalogue_session(session, is_calibration=False)
        except SessionError as exc:
            print(f"  SKIPPED: {exc}\n")
            rows.append({
                "session": session.name, "camera_slot": "", "serial": "",
                "calibration_slot": "", "status": "error", "detail": str(exc),
            })
            n_failed += 1
            continue
        for w in warnings:
            print(f"  ! {w}")

        try:
            new_data, changed = build_corrected_toml(master, cal_map, exp_map)
        except SessionError as exc:
            print(f"  SKIPPED: {exc}\n")
            rows.append({
                "session": session.name, "camera_slot": "", "serial": "",
                "calibration_slot": "", "status": "error", "detail": str(exc),
            })
            n_failed += 1
            continue

        for label in sorted(exp_map, key=camera_index):
            serial = exp_map[label]
            cal_label = serial_to_cal_label[serial]
            flag = "" if cal_label == label else f"  <-- was {cal_label} at calibration"
            print(f"  {label:<9} s#{serial}{flag}")
            rows.append({
                "session": session.name,
                "camera_slot": label,
                "serial": serial,
                "calibration_slot": cal_label,
                "status": "match" if cal_label == label else "remapped",
                "detail": "",
            })

        existing = find_calibration_toml(session)
        target = existing if existing is not None else session / master_toml.name

        if existing is None:
            if args.no_copy_missing:
                print("  No calibration file in this session; skipping (--no-copy-missing).\n")
                n_failed += 1
                continue
            print(f"  No calibration file present; will create {target.name}")

        if not changed:
            # Camera order matches, but still make sure a calibration file is there.
            if existing is None and not args.dry_run:
                shutil.copy2(master_toml, target)
                print("  Camera order matches calibration; copied master file in.")
            else:
                print("  Camera order matches calibration; no correction needed.")
            n_ok += 1
            print()
            continue

        new_text = toml_dumps(new_data, newline)
        if existing is not None and existing.read_bytes() == new_text.encode("utf-8"):
            print("  Calibration file is already corrected; left unchanged.")
            n_ok += 1
            print()
            continue

        if args.dry_run:
            print(f"  WOULD REWRITE {target.name} with the corrected camera order.")
        else:
            if existing is not None:
                bak = backup_path(target)
                shutil.copy2(target, bak)
                print(f"  Backed up existing file -> {bak.name}")
            target.write_text(new_text, encoding="utf-8", newline="")
            print(f"  REWROTE {target.name} with the corrected camera order.")
        n_fixed += 1
        print()

    # --- Report -------------------------------------------------------------
    report_path = Path(args.report) if args.report else day_dir / "camera_order_report.csv"
    if not args.dry_run:
        with open(report_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["session", "camera_slot", "serial",
                            "calibration_slot", "status", "detail"],
            )
            writer.writeheader()
            for label, serial in cal_map.items():
                writer.writerow({
                    "session": calib_dir.name, "camera_slot": label, "serial": serial,
                    "calibration_slot": label, "status": "calibration", "detail": "",
                })
            writer.writerows(rows)
        print(f"Report written to {report_path}")

    print(f"\nSummary: {n_ok} already correct, {n_fixed} corrected, {n_failed} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
