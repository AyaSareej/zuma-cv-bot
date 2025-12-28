from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from src.frog.frog_detector import detect_frog_by_contours


def parse_args():
    p = argparse.ArgumentParser(description="Quick test: frog contour detection on a single screenshot.")
    p.add_argument("--image", type=str, required=True, help="Path to screenshot (png/jpg).")
    p.add_argument("--outdir", type=str, default="assets/screenshots/_out", help="Output directory.")
    p.add_argument("--crop", action="store_true", help="If set, crop a manual ROI rectangle (edit values in code).")
    return p.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise RuntimeError("Failed to read image (cv2.imread returned None).")

    # Optional manual crop (ONLY for quick testing).
    # In the real system Member 1 will provide roi_frame already.
    roi = bgr
    if args.crop:
        # Edit these numbers to match your screenshot if needed:
        x, y, w, h = 0, 0, bgr.shape[1], bgr.shape[0]
        roi = bgr[y:y+h, x:x+w].copy()

    # Placeholder HSV range: tune this to your game's frog colors.
    # Tip: start wide, then narrow.
    hsv_lower = (25, 50, 50)
    hsv_upper = (95, 255, 255)

    det = detect_frog_by_contours(
        roi,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        min_area=800,
        max_area=200000,
    )

    # Draw debug visualization
    out = roi.copy()
    if det is not None:
        cx, cy = det.center
        cv2.drawContours(out, [det.contour], -1, (0, 255, 0), 3)
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(out, f"frog_center=({cx},{cy}) score={det.score:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "No frog candidate found (tune HSV/area thresholds)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"{img_path.stem}_frog_debug.png"
    cv2.imwrite(str(out_path), out)

    print("✅ Saved:", out_path)

    # Optional: show window (press any key to close)
    cv2.imshow("frog_debug", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
