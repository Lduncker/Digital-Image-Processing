import cv2
import numpy as np

# -----------------------------
# Utilities: histogram + CDF
# -----------------------------
def hist_256(u8: np.ndarray) -> np.ndarray:
    # 8-bit histogram with 256 bins
    return np.bincount(u8.ravel(), minlength=256).astype(np.float64)

def cdf_from_hist(h: np.ndarray) -> np.ndarray:
    c = np.cumsum(h)
    c /= c[-1] if c[-1] != 0 else 1.0
    return c

# -----------------------------
# histogram matching:
# For each intensity 'a', find 'z' with same CDF height
# (CDF_src(a) ≈ CDF_ref(z)):contentReference[oaicite:6]{index=6}
# -----------------------------
def match_histogram_channel(src_u8: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    h_src = hist_256(src_u8)
    cdf_src = cdf_from_hist(h_src)

    T = np.zeros(256, dtype=np.uint8)
    j = 0
    for a in range(256):
        while j < 255 and ref_cdf[j] < cdf_src[a]:
            j += 1
        T[a] = j

    # "Equalize pixel intensity" step (apply LUT mapping):contentReference[oaicite:7]{index=7}
    return T[src_u8]

def match_histogram_rgb(src_bgr: np.ndarray, ref_cdfs_rgb: list[np.ndarray]) -> np.ndarray:
    # Work in RGB order for clarity
    src_rgb = src_bgr[:, :, ::-1]
    out_rgb = np.empty_like(src_rgb)
    for c in range(3):  # R,G,B
        out_rgb[..., c] = match_histogram_channel(src_rgb[..., c], ref_cdfs_rgb[c])
    return out_rgb[:, :, ::-1]  # back to BGR

# -----------------------------
# Gamma correction (power-law): s = c * r^gamma:contentReference[oaicite:8]{index=8}
# -----------------------------
def gamma_correction_u8(bgr: np.ndarray, gamma: float) -> np.ndarray:
    x = bgr.astype(np.float32) / 255.0
    y = np.power(np.clip(x, 0, 1), gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)

# -----------------------------
# Channel-wise MSE (R,G,B) and overall average
# -----------------------------
def mse_channels_rgb(img1_bgr: np.ndarray, img2_bgr: np.ndarray):
    a = img1_bgr.astype(np.float32)[:, :, ::-1]  # RGB
    b = img2_bgr.astype(np.float32)[:, :, ::-1]
    mse = ((a - b) ** 2).mean(axis=(0, 1))       # [R,G,B]
    return mse, float(mse.mean())

# ============================================================
# TRAINING (allowed to use day):
# store reference CDFs (day image statistics)
# ============================================================
night_path = "night.jpg"
day_path   = "day.jpg"

night = cv2.imread(night_path, cv2.IMREAD_COLOR)  # BGR uint8
day   = cv2.imread(day_path,   cv2.IMREAD_COLOR)

if night is None:
    raise FileNotFoundError(f"Could not load night image from {night_path}")
if day is None:
    raise FileNotFoundError(f"Could not load day image from {day_path}")

day_rgb = day[:, :, ::-1]
ref_cdfs_rgb = [cdf_from_hist(hist_256(day_rgb[..., c])) for c in range(3)]

# Choose fixed params during training (allowed):
GAMMA = 1.1         # tuned on this training pair
USE_MEDIAN = True   # denoise via neighborhood filter (spatial filtering)
MEDIAN_K = 5

# ============================================================
# INFERENCE (night-only input):
# apply fixed denoise + fixed gamma + histogram matching using stored ref CDFs
# ============================================================
x = night.copy()

if USE_MEDIAN:
    x = cv2.medianBlur(x, MEDIAN_K)

if GAMMA != 1.0:
    x = gamma_correction_u8(x, GAMMA)

enhanced = match_histogram_rgb(x, ref_cdfs_rgb)

# Save result
out_path = "enhanced.jpg"
cv2.imwrite(out_path, enhanced)

# Report MSE
base_mse, base_overall = mse_channels_rgb(night, day)
final_mse, final_overall = mse_channels_rgb(enhanced, day)

print("Baseline MSE (R,G,B):", base_mse, "Overall:", base_overall)
print("Final    MSE (R,G,B):", final_mse, "Overall:", final_overall)
print("Saved:", out_path)
