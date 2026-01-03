# -*- coding: utf-8 -*-
"""
Stereo calibration (OpenCV) - Step 2 of the pipeline

Pipeline:
1) Capture stereo pairs:
   python scripts/capture_stereo_basler.py --num-frames 50

2) Calibrate stereo (this file):
   python scripts/stereo_calibrate.py --left-dir data/calibration/left --right-dir data/calibration/right

3) Rectify + disparity + point cloud:
   python scripts/disparity_to_pointcloud.py --calib-dir outputs/calibration

Repo notes:
- Do NOT commit raw images. Put them under data/ (gitignored).
- Calibration outputs are written to outputs/calibration/ (also gitignored by default).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import glob
import pickle

import cv2
import numpy as np
import pandas as pd


log = logging.getLogger("stereo_calibrate")


def list_images(img_dir: Path, pattern: str) -> list[Path]:
    files = sorted([Path(p) for p in glob.glob(str(img_dir / pattern))])
    return files


def load_list_from_excel(excel_path: Path, img_dir: Path, column: str = "Valid Images") -> list[Path]:
    df = pd.read_excel(excel_path)
    names = df[column].astype(str).tolist()
    return [img_dir / n for n in names]


def ensure_same_length(left_files: list[Path], right_files: list[Path]) -> tuple[list[Path], list[Path]]:
    n = min(len(left_files), len(right_files))
    if len(left_files) != len(right_files):
        log.warning("Left/right counts differ (L=%d, R=%d). Using first %d pairs.", len(left_files), len(right_files), n)
    return left_files[:n], right_files[:n]


def build_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """
    Chessboard inner-corner grid: cols x rows.
    Returns (cols*rows, 3) points on Z=0 plane in millimeters.
    """
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)
    return objp


def parse_args():
    p = argparse.ArgumentParser(description="Stereo calibration using chessboard images (OpenCV).")

    p.add_argument("--left-dir", type=str, required=True, help="Folder containing LEFT calibration images.")
    p.add_argument("--right-dir", type=str, required=True, help="Folder containing RIGHT calibration images.")
    p.add_argument("--pattern", type=str, default="*.jpg", help="Glob pattern for images (e.g., *.jpg, *.png).")

    p.add_argument("--cols", type=int, default=9, help="Number of inner corners along chessboard columns.")
    p.add_argument("--rows", type=int, default=6, help="Number of inner corners along chessboard rows.")
    p.add_argument("--square-size-mm", type=float, default=27.44, help="Chessboard square size in millimeters.")

    p.add_argument("--use-valid-excel", action="store_true",
                   help="If set, read valid image lists from valid_images_<method>.xlsx instead of globbing.")
    p.add_argument("--method", type=int, default=0,
                   help="Version index for valid_images_<method>.xlsx and output valid_images_<method+1>.xlsx")

    p.add_argument("--error-thresh", type=float, default=0.17,
                   help="Reprojection error threshold to mark an image pair as valid.")

    p.add_argument("--output-dir", type=str, default="outputs/calibration",
                   help="Where to write calibration parameters.")
    p.add_argument("--save-excel", action="store_true",
                   help="Also export valid image lists as Excel files (requires openpyxl).")

    p.add_argument("--debug-show", action="store_true",
                   help="Show detected corners during processing (press any key to advance).")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Termination criteria
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = build_object_points(args.cols, args.rows, args.square_size_mm)

    # Load file lists
    if args.use_valid_excel:
        xl_left = left_dir / f"valid_images_{args.method}.xlsx"
        xl_right = right_dir / f"valid_images_{args.method}.xlsx"
        if not xl_left.exists() or not xl_right.exists():
            raise FileNotFoundError(f"Missing valid lists: {xl_left} or {xl_right}")
        left_files = load_list_from_excel(xl_left, left_dir)
        right_files = load_list_from_excel(xl_right, right_dir)
        log.info("Loaded valid lists from Excel: %d pairs", min(len(left_files), len(right_files)))
    else:
        left_files = list_images(left_dir, args.pattern)
        right_files = list_images(right_dir, args.pattern)
        log.info("Loaded by glob: L=%d, R=%d", len(left_files), len(right_files))

    left_files, right_files = ensure_same_length(left_files, right_files)
    if len(left_files) == 0:
        raise RuntimeError("No image pairs found. Check --left-dir/--right-dir and --pattern.")

    objpoints: list[np.ndarray] = []
    imgpointsL: list[np.ndarray] = []
    imgpointsR: list[np.ndarray] = []

    used_left: list[Path] = []
    used_right: list[Path] = []

    # Detect corners
    for i, (lf, rf) in enumerate(zip(left_files, right_files)):
        imgL = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            log.warning("Skip unreadable pair: %s | %s", lf.name, rf.name)
            continue

        # OpenCV expects pattern size (cols, rows)
        retL, cornersL = cv2.findChessboardCorners(imgL, (args.cols, args.rows), None)
        retR, cornersR = cv2.findChessboardCorners(imgR, (args.cols, args.rows), None)

        if retL and retR:
            cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria_subpix)
            cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria_subpix)

            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            used_left.append(lf)
            used_right.append(rf)

            log.info("OK corners: %d | %s | %s", i, lf.name, rf.name)

            if args.debug_show:
                visL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
                visR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(visL, (args.cols, args.rows), cornersL, retL)
                cv2.drawChessboardCorners(visR, (args.cols, args.rows), cornersR, retR)
                cv2.imshow("Left corners", visL)
                cv2.imshow("Right corners", visR)
                cv2.waitKey(0)
        else:
            log.info("No corners: %d | %s | %s", i, lf.name, rf.name)

    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        raise RuntimeError(f"Too few valid corner detections: {len(objpoints)}. Need more images.")

    # Calibrate intrinsics for each camera
    img_size = cv2.imread(str(used_left[0]), cv2.IMREAD_GRAYSCALE).shape[::-1]  # (w, h)

    log.info("Calibrating LEFT intrinsics...")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, img_size, None, None)

    log.info("Calibrating RIGHT intrinsics...")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, img_size, None, None)

    log.info("Stereo calibrate (FIX_INTRINSIC)...")
    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        mtxL,
        distL,
        mtxR,
        distR,
        img_size,
        criteria=criteria_stereo,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    rectify_scale = 0  # 0 = cropped, 1 = full view (may include black borders)
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
        MLS, dLS, MRS, dRS, img_size, R, T, alpha=rectify_scale
    )

    left_map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, img_size, cv2.CV_16SC2)
    right_map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, img_size, cv2.CV_16SC2)

    # Reprojection error per used image
    total_err_L = 0.0
    total_err_R = 0.0
    valid_left_names: list[str] = []
    valid_right_names: list[str] = []

    for i in range(len(objpoints)):
        projL, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
        projR, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)

        errL = cv2.norm(imgpointsL[i], projL, cv2.NORM_L2) / len(projL)
        errR = cv2.norm(imgpointsR[i], projR, cv2.NORM_L2) / len(projR)

        total_err_L += errL
        total_err_R += errR

        nameL = used_left[i].name
        nameR = used_right[i].name

        log.info("Reproj error %d | L=%.6f R=%.6f | %s | %s", i, errL, errR, nameL, nameR)

        if errL < args.error_thresh and errR < args.error_thresh:
            valid_left_names.append(nameL)
            valid_right_names.append(nameR)

    mean_err_L = total_err_L / len(objpoints)
    mean_err_R = total_err_R / len(objpoints)
    log.info("Mean reprojection error LEFT : %.6f", mean_err_L)
    log.info("Mean reprojection error RIGHT: %.6f", mean_err_R)
    log.info("Valid pairs under threshold %.3f: %d", args.error_thresh, len(valid_left_names))

    # Save parameters (npz + pickle for backward compatibility)
    (out_dir / "parameters").mkdir(parents=True, exist_ok=True)
    param_dir = out_dir / "parameters"

    np.savez(param_dir / "intrinsics_left.npz", mtx=mtxL, dist=distL)
    np.savez(param_dir / "intrinsics_right.npz", mtx=mtxR, dist=distR)

    np.savez(
        param_dir / "stereo_extrinsics_rectify.npz",
        R=R, T=T, E=E, F=F,
        RL=RL, RR=RR, PL=PL, PR=PR, Q=Q,
        roiL=np.array(roiL), roiR=np.array(roiR),
    )

    mapL_1, mapL_2 = left_map
    mapR_1, mapR_2 = right_map
    np.savez(param_dir / "rectify_maps_left.npz", map1=mapL_1, map2=mapL_2)
    np.savez(param_dir / "rectify_maps_right.npz", map1=mapR_1, map2=mapR_2)

    # Optional: keep your original pickle outputs (handy if other scripts depend on them)
    pickle.dump({"mtxL": mtxL, "distL": distL, "rvecsL": rvecsL, "tvecsL": tvecsL}, open(param_dir / "CamParasL.p", "wb"))
    pickle.dump({"mtxR": mtxR, "distR": distR, "rvecsR": rvecsR, "tvecsR": tvecsR}, open(param_dir / "CamParasR.p", "wb"))
    pickle.dump({"mapxL": mapL_1, "mapyL": mapL_2, "mapxR": mapR_1, "mapyR": mapR_2}, open(param_dir / "stereoRemap.p", "wb"))
    pickle.dump(Q, open(param_dir / "disp2depth.p", "wb"))
    pickle.dump(T, open(param_dir / "Baseline.p", "wb"))

    # Save valid image lists (CSV always, Excel optional)
    valid_dir = out_dir / "valid_lists"
    valid_dir.mkdir(parents=True, exist_ok=True)

    dfL = pd.DataFrame(valid_left_names, columns=["Valid Images"])
    dfR = pd.DataFrame(valid_right_names, columns=["Valid Images"])

    # Next method index output (like your original behavior)
    next_method = args.method + 1
    csvL = valid_dir / f"valid_images_{next_method}_L.csv"
    csvR = valid_dir / f"valid_images_{next_method}_R.csv"
    dfL.to_csv(csvL, index=False)
    dfR.to_csv(csvR, index=False)

    if args.save_excel:
        xlsL = valid_dir / f"valid_images_{next_method}_L.xlsx"
        xlsR = valid_dir / f"valid_images_{next_method}_R.xlsx"
        dfL.to_excel(xlsL, index=False)
        dfR.to_excel(xlsR, index=False)

    log.info("Saved calibration to: %s", param_dir.as_posix())
    log.info("Saved valid lists to: %s", valid_dir.as_posix())


if __name__ == "__main__":
    main()
