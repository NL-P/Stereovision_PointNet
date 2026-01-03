# -*- coding: utf-8 -*-
"""
Disparity -> Depth -> Point cloud - Step 3 of the pipeline

Pipeline:
1) Capture stereo pairs:
   python scripts/capture_stereo_basler.py --num-frames 50

2) Stereo calibration:
   python scripts/stereo_calibrate.py --left-dir data/calibration/left --right-dir data/calibration/right

3) Disparity + depth + PLY:
   python scripts/disparity_to_pointcloud.py \
       --left-dir data/session_001/left \
       --right-dir data/session_001/right \
       --calib-dir outputs/calibration/parameters \
       --out-dir outputs/reconstruction/session_001 \
       --pattern "*.jpg"

Notes:
- This script uses cv2.ximgproc (WLS filter), which requires opencv-contrib-python.
- Depth units: if your calibration square size is in mm, baseline is mm, then depth output is mm.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import glob
import re
import pickle

import cv2
import numpy as np
import pandas as pd


log = logging.getLogger("disparity_to_pointcloud")


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(img_dir: Path, pattern: str) -> list[Path]:
    return sorted([Path(p) for p in glob.glob(str(img_dir / pattern))])


def pair_by_index(left_files: list[Path], right_files: list[Path]) -> list[tuple[Path, Path]]:
    """
    Pair left/right by sorted order. If you need strict pairing by index in filename,
    adapt this to parse indices and match.
    """
    n = min(len(left_files), len(right_files))
    if len(left_files) != len(right_files):
        log.warning("Left/right counts differ (L=%d, R=%d). Using first %d pairs.", len(left_files), len(right_files), n)
    return list(zip(left_files[:n], right_files[:n]))


def safe_imread_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    # WLS + SGBM are happiest with uint8 grayscale
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def save_ply_xyzrgb(points_xyzrgb: np.ndarray, out_path: Path):
    """
    points_xyzrgb: (N, 6) columns: x y z r g b
    """
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points_xyzrgb)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        np.savetxt(f, points_xyzrgb, fmt="%f %f %f %d %d %d")


# -----------------------------
# Calibration loading (NPZ or pickle)
# -----------------------------
def load_calibration(calib_dir: Path):
    """
    Supports:
    - New format (npz):
        intrinsics_left.npz, intrinsics_right.npz
        rectify_maps_left.npz, rectify_maps_right.npz
        stereo_extrinsics_rectify.npz  (contains T)
    - Legacy pickle format:
        CamParasL.p, CamParasR.p, stereoRemap.p, Baseline.p
    """
    calib_dir = Path(calib_dir)

    # Prefer NPZ
    intrL = calib_dir / "intrinsics_left.npz"
    intrR = calib_dir / "intrinsics_right.npz"
    mapsL = calib_dir / "rectify_maps_left.npz"
    mapsR = calib_dir / "rectify_maps_right.npz"
    extr = calib_dir / "stereo_extrinsics_rectify.npz"

    if intrL.exists() and intrR.exists() and mapsL.exists() and mapsR.exists() and extr.exists():
        L = np.load(intrL)
        R = np.load(intrR)
        ML = np.load(mapsL)
        MR = np.load(mapsR)
        EX = np.load(extr)

        mtxL, distL = L["mtx"], L["dist"]
        mtxR, distR = R["mtx"], R["dist"]
        mapxL, mapyL = ML["map1"], ML["map2"]
        mapxR, mapyR = MR["map1"], MR["map2"]
        T = EX["T"]

        log.info("Loaded calibration from NPZ in %s", calib_dir.as_posix())
        return mtxL, distL, mtxR, distR, (mapxL, mapyL, mapxR, mapyR), T

    # Fallback: pickles
    camL = calib_dir / "CamParasL.p"
    camR = calib_dir / "CamParasR.p"
    remap = calib_dir / "stereoRemap.p"
    base = calib_dir / "Baseline.p"

    if camL.exists() and camR.exists() and remap.exists() and base.exists():
        CamParasL = pickle.load(open(camL, "rb"))
        CamParasR = pickle.load(open(camR, "rb"))
        stereoRemap = pickle.load(open(remap, "rb"))
        T = pickle.load(open(base, "rb"))

        mtxL, distL = CamParasL["mtxL"], CamParasL["distL"]
        mtxR, distR = CamParasR["mtxR"], CamParasR["distR"]
        mapxL, mapyL = stereoRemap["mapxL"], stereoRemap["mapyL"]
        mapxR, mapyR = stereoRemap["mapxR"], stereoRemap["mapyR"]

        log.info("Loaded calibration from PICKLES in %s", calib_dir.as_posix())
        return mtxL, distL, mtxR, distR, (mapxL, mapyL, mapxR, mapyR), T

    raise FileNotFoundError(
        "Calibration files not found. Provide calib dir containing either NPZ set or pickle set.\n"
        f"Given calib_dir: {calib_dir}"
    )


# -----------------------------
# Stereo parameters
# -----------------------------
def default_sgbm_params():
    # Reasonable defaults; override by Excel or CLI
    return dict(
        window_size=5,
        min_disp=0,
        num_disp=16 * 10,          # must be divisible by 16
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=5,
        P1=8 * 3 * (5 ** 2),
        P2=32 * 3 * (5 ** 2),
        lmbda=8000.0,
        sigma=1.5,
    )


def load_params_from_excel(excel_path: Path, row: int):
    df = pd.read_excel(excel_path)
    if row < 0 or row >= len(df):
        raise IndexError(f"Row {row} out of range for {excel_path} (len={len(df)})")

    r = df.loc[row]
    p = dict(
        window_size=int(r["window_size"]),
        min_disp=int(r["min_disp"]),
        num_disp=int(r["num_disp"]),
        uniquenessRatio=int(r["uniquenessRatio"]),
        speckleWindowSize=int(r["speckleWindowSize"]),
        speckleRange=int(r["speckleRange"]),
        disp12MaxDiff=int(r["disp12MaxDiff"]),
        P1=int(r["P1"]),
        P2=int(r["P2"]),
        lmbda=float(r["lmbda"]),
        sigma=float(r["sigma"]),
    )

    # SGBM rule: num_disp must be divisible by 16
    if p["num_disp"] % 16 != 0:
        p["num_disp"] = int(np.ceil(p["num_disp"] / 16.0) * 16)

    return p


# -----------------------------
# Core computation
# -----------------------------
def compute_depth_from_disparity(disparity_float: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    # disparity_float is in pixels, float32
    return (fx * baseline) / (disparity_float + 1e-6)


def make_point_cloud(grayL: np.ndarray, depth: np.ndarray, mtxL: np.ndarray,
                     z_min: float, z_max: float) -> np.ndarray:
    """
    Returns (N, 6) xyzrgb with grayscale mapped to RGB.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    fx, fy = float(mtxL[0, 0]), float(mtxL[1, 1])
    cx, cy = float(mtxL[0, 2]), float(mtxL[1, 2])

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    valid = np.isfinite(z) & (z >= z_min) & (z <= z_max)
    x, y, z = x[valid], y[valid], z[valid]
    c = grayL[valid].astype(np.uint8)

    rgb = np.stack([c, c, c], axis=1)  # grayscale -> RGB
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    xyzrgb = np.concatenate([xyz, rgb.astype(np.float32)], axis=1)

    # Cast RGB to ints for PLY writer
    xyzrgb[:, 3:] = np.clip(xyzrgb[:, 3:], 0, 255)
    xyzrgb = np.column_stack([xyzrgb[:, :3], xyzrgb[:, 3:].astype(np.uint8)])
    return xyzrgb


def parse_args():
    p = argparse.ArgumentParser(description="Compute disparity, depth, and point clouds from stereo images.")

    p.add_argument("--left-dir", type=str, required=True, help="Folder of LEFT images.")
    p.add_argument("--right-dir", type=str, required=True, help="Folder of RIGHT images.")
    p.add_argument("--pattern", type=str, default="*.jpg", help="Glob pattern, e.g. *.jpg or *.png")

    p.add_argument("--calib-dir", type=str, required=True,
                   help="Calibration parameter folder (NPZ set or legacy pickle set).")

    p.add_argument("--out-dir", type=str, default="outputs/reconstruction",
                   help="Where outputs (twin image, disparity, depth, pointcloud) are written.")

    p.add_argument("--start", type=int, default=0, help="Start pair index (0-based) after sorting.")
    p.add_argument("--count", type=int, default=-1, help="How many pairs to process (-1 = all).")

    # Depth filtering
    p.add_argument("--z-min", type=float, default=100.0, help="Min depth (same unit as baseline, usually mm).")
    p.add_argument("--z-max", type=float, default=3000.0, help="Max depth (same unit as baseline, usually mm).")

    # Depth preview clamp for colored JPG
    p.add_argument("--depth-vis-min", type=float, default=600.0, help="Clamp min for depth visualization.")
    p.add_argument("--depth-vis-max", type=float, default=1100.0, help="Clamp max for depth visualization.")

    # Parameter sources
    p.add_argument("--param-excel", type=str, default=None,
                   help="Excel file with SGBM+WLS params (optional).")
    p.add_argument("--param-row", type=int, default=1,
                   help="Row index in Excel for params (0-based).")

    # Display/debug
    p.add_argument("--save-disparity", action="store_true", help="Save disparity visualization JPG.")
    p.add_argument("--save-twin", action="store_true", help="Save side-by-side rectified image.")
    p.add_argument("--no-wls", action="store_true", help="Disable WLS filter (use raw left disparity).")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    calib_dir = Path(args.calib_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    out_twin = ensure_dir(out_dir / "ImgTwin")
    out_disp = ensure_dir(out_dir / "Disparity")
    out_depth = ensure_dir(out_dir / "Depth")
    out_pc = ensure_dir(out_dir / "PointCloud")

    # Load calibration
    mtxL, distL, mtxR, distR, (mapxL, mapyL, mapxR, mapyR), T = load_calibration(calib_dir)
    fx, fy = float(mtxL[0, 0]), float(mtxL[1, 1])
    baseline = float(np.linalg.norm(T))

    log.info("fx=%.3f fy=%.3f | baseline=%.6f", fx, fy, baseline)

    # Load params
    params = default_sgbm_params()
    if args.param_excel:
        params = load_params_from_excel(Path(args.param_excel), args.param_row)
        log.info("Loaded SGBM/WLS params from Excel row %d: %s", args.param_row, params)
    else:
        log.info("Using default SGBM/WLS params: %s", params)

    # Ensure OpenCV contrib module exists if WLS is enabled
    if not args.no_wls and not hasattr(cv2, "ximgproc"):
        raise RuntimeError(
            "cv2.ximgproc not found. Install opencv-contrib-python:\n"
            "  pip install opencv-contrib-python"
        )

    # List input pairs
    left_files = list_images(left_dir, args.pattern)
    right_files = list_images(right_dir, args.pattern)
    pairs = pair_by_index(left_files, right_files)

    if args.start < 0 or args.start >= len(pairs):
        raise IndexError(f"--start {args.start} out of range. Total pairs: {len(pairs)}")

    pairs = pairs[args.start:]
    if args.count != -1:
        pairs = pairs[: args.count]

    if len(pairs) == 0:
        raise RuntimeError("No stereo pairs to process. Check your dirs/pattern.")

    # Create matchers
    stereo = cv2.StereoSGBM_create(
        minDisparity=int(params["min_disp"]),
        numDisparities=int(params["num_disp"]),
        blockSize=int(params["window_size"]),
        uniquenessRatio=int(params["uniquenessRatio"]),
        speckleWindowSize=int(params["speckleWindowSize"]),
        speckleRange=int(params["speckleRange"]),
        disp12MaxDiff=int(params["disp12MaxDiff"]),
        P1=int(params["P1"]),
        P2=int(params["P2"]),
    )

    if args.no_wls:
        wls_filter = None
        stereoR = None
    else:
        stereoR = cv2.ximgproc.createRightMatcher(stereo)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(float(params["lmbda"]))
        wls_filter.setSigmaColor(float(params["sigma"]))

    # Process
    for k, (lf, rf) in enumerate(pairs):
        idx = args.start + k
        log.info("Processing pair %d | %s | %s", idx, lf.name, rf.name)

        frameL = safe_imread_gray(lf)
        frameR = safe_imread_gray(rf)

        # Rectify
        Left_nice = cv2.remap(frameL, mapxL, mapyL, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        Right_nice = cv2.remap(frameR, mapxR, mapyR, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

        grayL = Left_nice
        grayR = Right_nice

        # Optional: twin save
        if args.save_twin:
            twin = np.hstack([grayL, grayR])
            cv2.imwrite(str(out_twin / f"imgTwin_{idx:06d}.jpg"), twin)

        # Compute disparity (OpenCV returns fixed-point disparity scaled by 16)
        dispL = stereo.compute(grayL, grayR).astype(np.int16)

        if args.no_wls:
            filtered = dispL
        else:
            dispR = stereoR.compute(grayR, grayL).astype(np.int16)

            # WLS expects:
            # - left disparity: CV_16S
            # - right disparity: CV_16S
            # - left view image: 8-bit single channel
            filtered = wls_filter.filter(dispL, grayL, None, dispR)

        # Convert to float disparity in pixels
        disparity = filtered.astype(np.float32) / 16.0

        # Optional disparity visualization
        if args.save_disparity:
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = disp_vis.astype(np.uint8)
            disp_vis_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(out_disp / f"disparity_{idx:06d}.jpg"), disp_vis_color)

        # Depth (same unit as baseline)
        depth = compute_depth_from_disparity(disparity, fx=fx, baseline=baseline)
        np.save(str(out_depth / f"depth_{idx:06d}.npy"), depth)

        # Depth colored image (clamped for display)
        depth_vis = np.clip(depth, args.depth_vis_min, args.depth_vis_max)
        depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_depth / f"depth_{idx:06d}.jpg"), depth_color)

        # Point cloud
        xyzrgb = make_point_cloud(grayL, depth, mtxL, z_min=args.z_min, z_max=args.z_max)
        ply_path = out_pc / f"pointcloud_{idx:06d}.ply"
        save_ply_xyzrgb(xyzrgb, ply_path)

    cv2.destroyAllWindows()
    log.info("Done. Outputs written to %s", out_dir.as_posix())


if __name__ == "__main__":
    main()
