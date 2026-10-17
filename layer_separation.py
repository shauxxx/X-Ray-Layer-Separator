
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy.ndimage import shift as image_shift
from scipy.optimize import minimize
from skimage.restoration import denoise_tv_chambolle


# -------------------- CONFIGURATION --------------------

INPUT_DIR = Path("photo_for_test1")
OUTPUT_DIR = Path("output")

# Relative shifts of each layer in pixels for each frame.
# Replace these example values with measured/calculated shifts.
#
# Each row: [dx_top, dy_top, dx_bottom, dy_bottom]
# There must be one row per input image.
SHIFTS = None

# Example only; NOT real measurements:
# SHIFTS = np.array([
#     [0.0, 0.0, 0.0, 0.0],
#     [0.3, 0.0, -0.3, 0.0],
#     [0.6, 0.0, -0.6, 0.0],
# ], dtype=np.float32)

TV_WEIGHT = 0.002
MAX_ITER = 150
DOWNSCALE = 1  # Use 2 or 4 for a faster initial test


def load_frames(folder):
    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in
        {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    )

    if len(paths) < 3:
        raise ValueError("Need at least 3 projection frames.")

    frames = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read {path}")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)

        if DOWNSCALE > 1:
            img = cv2.resize(
                img,
                (img.shape[1] // DOWNSCALE,
                 img.shape[0] // DOWNSCALE),
                interpolation=cv2.INTER_AREA
            )

        frames.append(img)

    shape = frames[0].shape
    if any(f.shape != shape for f in frames):
        raise ValueError("All frames must have identical dimensions.")

    return np.stack(frames), paths


def warp_layer(layer, dx, dy):
    # Positive dx moves content to the right;
    # positive dy moves content downward.
    return image_shift(
        layer, shift=(dy, dx),
        order=1, mode="nearest", prefilter=False
    )


def reconstruct(frames, shifts, tv_weight=TV_WEIGHT):
    """
    Solve:
      min_T,B sum_k ||I_k - W(T, shift_top_k)
                          - W(B, shift_bottom_k)||^2
          + TV_WEIGHT * (TV(T) + TV(B))

    The shifts are assumed known.
    """
    n, h, w = frames.shape
    if shifts.shape != (n, 4):
        raise ValueError(
            f"SHIFTS must have shape ({n}, 4), got {shifts.shape}"
        )

    # Initial estimate: split the average image equally.
    avg = frames.mean(axis=0)
    initial = np.concatenate([
        (avg * 0.5).ravel(),
        (avg * 0.5).ravel()
    ])

    def objective(z):
        top = z[:h * w].reshape(h, w)
        bottom = z[h * w:].reshape(h, w)

        loss = 0.0
        grad_top = np.zeros_like(top)
        grad_bottom = np.zeros_like(bottom)

        for frame, (dxt, dyt, dxb, dyb) in zip(frames, shifts):
            wt = warp_layer(top, dxt, dyt)
            wb = warp_layer(bottom, dxb, dyb)
            residual = wt + wb - frame

            loss += 0.5 * np.sum(residual ** 2)

            # Approximate inverse-warp gradients.
            grad_top += warp_layer(
                residual, -dxt, -dyt
            )
            grad_bottom += warp_layer(
                residual, -dxb, -dyb
            )

        # Smoothness regularization using first differences.
        # This is a quadratic approximation to TV regularization.
        def smoothness(x):
            gx = np.diff(x, axis=1, append=x[:, -1:])
            gy = np.diff(x, axis=0, append=x[-1:, :])
            return 0.5 * np.sum(gx**2 + gy**2)

        # For a robust production implementation, replace this
        # smoothness penalty with a proper TV proximal optimizer.
        loss += tv_weight * (
            smoothness(top) + smoothness(bottom)
        )

        # Gradient of quadratic smoothness.
        def smooth_grad(x):
            return (
                4 * x
                - np.roll(x, 1, axis=0)
                - np.roll(x, -1, axis=0)
                - np.roll(x, 1, axis=1)
                - np.roll(x, -1, axis=1)
            )

        grad_top += tv_weight * smooth_grad(top)
        grad_bottom += tv_weight * smooth_grad(bottom)

        grad = np.concatenate([
            grad_top.ravel(), grad_bottom.ravel()
        ])
        return loss, grad

    result = minimize(
        objective, initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAX_ITER}
    )

    top = result.x[:h * w].reshape(h, w)
    bottom = result.x[h * w:].reshape(h, w)

    return top, bottom, result


def save_image(path, img):
    # Keep the reconstructed data in floating point until export.
    img = np.nan_to_num(img)
    lo, hi = np.percentile(img, [0.5, 99.5])

    if hi <= lo:
        hi = lo + 1.0

    normalized = np.clip((img - lo) / (hi - lo), 0, 1)
    tifffile.imwrite(str(path), normalized.astype(np.float32))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, paths = load_frames(INPUT_DIR)

    print(f"Loaded {len(paths)} frames, shape={frames.shape[1:]}")

    if SHIFTS is None:
        raise RuntimeError(
            "Set SHIFTS to measured/calculated per-frame shifts. "
            "The program cannot infer physical layer separation "
            "from frame count alone."
        )

    shifts = np.asarray(SHIFTS, dtype=np.float32)
    shifts[:, :] /= DOWNSCALE

    top, bottom, result = reconstruct(frames, shifts)

    save_image(OUTPUT_DIR / "top_layer.tif", top)
    save_image(OUTPUT_DIR / "bottom_layer.tif", bottom)

    print("Optimization success:", result.success)
    print("Final loss:", result.fun)
    print("Saved results to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
