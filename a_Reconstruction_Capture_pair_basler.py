# -*- coding: utf-8 -*-
"""
Stereo capture (Basler + pypylon) - Step 1 of the pipeline

Pipeline:
1) Capture synchronized-ish stereo pairs with this script.
2) Calibrate stereo using your chessboard pairs:
   - Run: scripts/stereo_calibrate.py
   - Output: calibration files (intrinsics/extrinsics)
3) Generate disparity and point cloud:
   - Run: scripts/disparity_to_pointcloud.py

Notes:
- This repo does NOT commit images. Outputs go to ./outputs by default.
- Prefer selecting cameras by serial number to avoid left/right swapping.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
from pypylon import pylon


log = logging.getLogger("stereo_capture")


def make_output_dirs(base: Path, session_name: str) -> tuple[Path, Path]:
    session_dir = base / session_name
    left_dir = session_dir / "left"
    right_dir = session_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    return left_dir, right_dir


def pick_devices_by_serial(devices, left_serial: str | None, right_serial: str | None):
    if left_serial and right_serial:
        dev_map = {d.GetSerialNumber(): d for d in devices}
        if left_serial not in dev_map or right_serial not in dev_map:
            available = [d.GetSerialNumber() for d in devices]
            raise RuntimeError(
                f"Serial not found. Available serials: {available}"
            )
        return dev_map[left_serial], dev_map[right_serial]

    # fallback: first two devices (less safe)
    return devices[0], devices[1]


def configure_camera(cam: pylon.InstantCamera, width: int, height: int, exposure_us: int, fps: float):
    cam.Open()

    # Resolution
    cam.Width.SetValue(width)
    cam.Height.SetValue(height)

    # Exposure
    cam.ExposureTime.SetValue(exposure_us)

    # FPS (if supported)
    if cam.AcquisitionFrameRateEnable.IsWritable():
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(fps)

    # You may also want to set PixelFormat explicitly depending on your camera model:
    # cam.PixelFormat.SetValue("Mono8") or "BayerRG8" etc.


def grab_to_array(grab_result: pylon.GrabResult) -> "cv2.typing.MatLike":
    """
    Convert pypylon grab result to a numpy array that OpenCV can save/display.
    Uses pypylon converter to ensure 8-bit output where possible.
    """
    converter = pylon.ImageFormatConverter()
    # Works well for many mono pipelines:
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    img = converter.Convert(grab_result)
    return img.GetArray()


def parse_args():
    p = argparse.ArgumentParser(description="Capture stereo image pairs from two Basler cameras.")
    p.add_argument("--num-frames", type=int, default=50, help="Number of stereo pairs to capture.")
    p.add_argument("--output-root", type=str, default="outputs", help="Root folder for capture sessions.")
    p.add_argument("--session", type=str, default=None, help="Session name (default: timestamp).")
    p.add_argument("--start-index", type=int, default=0, help="Starting index for filenames.")
    p.add_argument("--width", type=int, default=5000, help="Capture width in pixels.")
    p.add_argument("--height", type=int, default=3660, help="Capture height in pixels.")
    p.add_argument("--exposure-us", type=int, default=30000, help="Exposure time in microseconds.")
    p.add_argument("--fps", type=float, default=18.0, help="Camera acquisition frame rate.")
    p.add_argument(
        "--interval-ms",
        type=int,
        default=0,
        help="Optional delay between saved frames (0 = no enforced delay).",
    )
    p.add_argument("--left-serial", type=str, default=None, help="Serial number for LEFT camera.")
    p.add_argument("--right-serial", type=str, default=None, help="Serial number for RIGHT camera.")
    p.add_argument("--display", action="store_true", help="Show live preview windows.")
    p.add_argument("--preview-width", type=int, default=800, help="Preview window width.")
    p.add_argument("--preview-height", type=int, default=600, help="Preview window height.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    output_root = Path(args.output_root)
    session_name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    left_dir, right_dir = make_output_dirs(output_root, session_name)

    log.info("Output session: %s", (output_root / session_name).as_posix())
    log.info("Left  dir: %s", left_dir.as_posix())
    log.info("Right dir: %s", right_dir.as_posix())

    # Enumerate devices
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if len(devices) < 2:
        raise RuntimeError("At least two cameras are required.")

    left_dev, right_dev = pick_devices_by_serial(devices, args.left_serial, args.right_serial)

    camera_l = pylon.InstantCamera(tl_factory.CreateDevice(left_dev))
    camera_r = pylon.InstantCamera(tl_factory.CreateDevice(right_dev))

    try:
        configure_camera(camera_l, args.width, args.height, args.exposure_us, args.fps)
        configure_camera(camera_r, args.width, args.height, args.exposure_us, args.fps)

        camera_l.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        camera_r.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        log.info("Cameras initialized. Start capturing...")

        idx = args.start_index
        start_time = datetime.now()

        for k in range(args.num_frames):
            t0 = time.time()

            grab_l = camera_l.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            grab_r = camera_r.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            try:
                if not (grab_l.GrabSucceeded() and grab_r.GrabSucceeded()):
                    log.warning("Grab failed at frame %d, skipping.", idx)
                    continue

                frame_l = grab_to_array(grab_l)
                frame_r = grab_to_array(grab_r)

                # Save full-res
                out_l = left_dir / f"frame{idx:06d}_l.jpg"
                out_r = right_dir / f"frame{idx:06d}_r.jpg"
                cv2.imwrite(str(out_l), frame_l)
                cv2.imwrite(str(out_r), frame_r)

                log.info("Saved pair %d -> %s | %s", idx, out_l.name, out_r.name)

                # Optional display
                if args.display:
                    prev_l = cv2.resize(frame_l, (args.preview_width, args.preview_height))
                    prev_r = cv2.resize(frame_r, (args.preview_width, args.preview_height))
                    cv2.imshow("Left", prev_l)
                    cv2.imshow("Right", prev_r)

                    # Press q to quit early
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("Quit requested by user.")
                        break

                idx += 1

            finally:
                grab_l.Release()
                grab_r.Release()

            # Optional interval control
            if args.interval_ms > 0:
                elapsed_ms = (time.time() - t0) * 1000.0
                sleep_ms = max(args.interval_ms - elapsed_ms, 0.0)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)

        total_s = (datetime.now() - start_time).total_seconds()
        if total_s > 0:
            log.info("Done. Effective FPS (saved pairs): %.3f", args.num_frames / total_s)

    finally:
        # Cleanup
        if camera_l.IsGrabbing():
            camera_l.StopGrabbing()
        if camera_r.IsGrabbing():
            camera_r.StopGrabbing()

        if camera_l.IsOpen():
            camera_l.Close()
        if camera_r.IsOpen():
            camera_r.Close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
