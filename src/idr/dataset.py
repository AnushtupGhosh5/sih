"""
IO-VNBD synchronised dataset loader + mounting-invariant feature engineering.

Input  : smartphone IMU (accelerometer, gyroscope, gravity) -> S-Dataset
Label  : vehicle VBOX forward velocity (m/s)                 -> V-Dataset
The V and S CSVs are row-synchronised (same drive, 10 Hz); we truncate each
pair to the shorter length to absorb the few off-by-N tail mismatches.
"""
import os
import re
import glob
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Column resolution (headers are verbose & unit-suffixed; match by keyword)
# --------------------------------------------------------------------------
def _match(term, col_lower):
    """Single-letter terms (axis x/y/z) must be isolated tokens, not letters
    inside a word -- otherwise 'y' matches the 'y' in 'gyroscope'/'gravity'.
    Longer terms match as plain substrings."""
    if len(term) == 1:
        return re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", col_lower) is not None
    return term in col_lower


def _find(cols, *musts):
    musts = [m.lower() for m in musts]
    for col in cols:
        cl = col.lower()
        if all(_match(m, cl) for m in musts):
            return col
    raise KeyError(f"No column matching {musts} in {list(cols)[:30]}")


def list_pairs():
    """Return {drive_key: (v_path, s_path)} for all paired drives."""
    vf = {os.path.basename(f)[2:-4].lower(): f
          for f in glob.glob(os.path.join(C.V_DIR, "*.csv"))}
    sf = {os.path.basename(f)[2:-4].lower(): f
          for f in glob.glob(os.path.join(C.S_DIR, "*.csv"))}
    keys = sorted(set(vf) & set(sf))
    return {k: (vf[k], sf[k]) for k in keys}


def load_pair(key, v_path, s_path):
    """Load one synchronised drive into a tidy DataFrame with unified columns."""
    v = pd.read_csv(v_path, encoding=C.ENCODING)
    s = pd.read_csv(s_path, encoding=C.ENCODING)
    n = min(len(v), len(s))
    v, s = v.iloc[:n].reset_index(drop=True), s.iloc[:n].reset_index(drop=True)

    # --- Smartphone IMU (inputs) ---
    ax = pd.to_numeric(s[_find(s.columns, "accelerometer", "x")], errors="coerce")
    ay = pd.to_numeric(s[_find(s.columns, "accelerometer", "y")], errors="coerce")
    az = pd.to_numeric(s[_find(s.columns, "accelerometer", "z")], errors="coerce")
    gx = pd.to_numeric(s[_find(s.columns, "gyroscope", "x")], errors="coerce")
    gy = pd.to_numeric(s[_find(s.columns, "gyroscope", "y")], errors="coerce")
    gz = pd.to_numeric(s[_find(s.columns, "gyroscope", "z")], errors="coerce")
    grx = pd.to_numeric(s[_find(s.columns, "gravity", "x")], errors="coerce")
    gry = pd.to_numeric(s[_find(s.columns, "gravity", "y")], errors="coerce")
    grz = pd.to_numeric(s[_find(s.columns, "gravity", "z")], errors="coerce")

    # --- Ground truth (labels / evaluation) ---
    v_vel = pd.to_numeric(v[_find(v.columns, "velocity")], errors="coerce") * C.KMH_TO_MS
    v_lat = pd.to_numeric(v[_find(v.columns, "latitude")], errors="coerce")
    v_lon = pd.to_numeric(v[_find(v.columns, "longitude")], errors="coerce")
    v_head = pd.to_numeric(v[_find(v.columns, "heading")], errors="coerce")

    # Phone compass azimuth (Android gyro+mag fused heading) + raw magnetometer
    azi = pd.to_numeric(s[_find(s.columns, "orientation", "azimuth")], errors="coerce")
    mx = pd.to_numeric(s[_find(s.columns, "magnetic", "x")], errors="coerce")
    my = pd.to_numeric(s[_find(s.columns, "magnetic", "y")], errors="coerce")
    mz = pd.to_numeric(s[_find(s.columns, "magnetic", "z")], errors="coerce")

    df = pd.DataFrame({
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
        "grx": grx, "gry": gry, "grz": grz,
        "mx": mx, "my": my, "mz": mz, "az_deg": azi,
        "speed_ms": v_vel, "lat": v_lat, "lon": v_lon, "heading_deg": v_head,
    })
    df["drive"] = key
    # Drop rows with any NaN in the essential fields.
    df = df.dropna().reset_index(drop=True)

    # Course-over-ground from the accurate VBOX position track. The raw vehicle
    # "Heading" channel glitches at low speed, so the position-derived course is
    # a far more reliable heading reference when the vehicle is moving.
    df["course_deg"] = _course_over_ground(df["lat"].to_numpy(),
                                           df["lon"].to_numpy())
    return df


def _course_over_ground(lat, lon, smooth=5):
    lat0 = np.radians(lat[0])
    e = np.radians(lon - lon[0]) * 6371000.0 * np.cos(lat0)
    n = np.radians(lat - lat[0]) * 6371000.0
    k = np.ones(smooth) / smooth
    e = np.convolve(e, k, "same"); n = np.convolve(n, k, "same")
    course = np.degrees(np.arctan2(np.gradient(e), np.gradient(n))) % 360.0
    return course


# --------------------------------------------------------------------------
# Mounting-invariant feature engineering
# --------------------------------------------------------------------------
def compute_features(df):
    """
    Build 6 rotation/mounting-invariant channels from raw phone IMU.

    Using the provided gravity vector g as the vertical reference makes every
    channel invariant to how the phone is yawed on the mount, and robust to its
    pitch/roll -- directly addressing the "accidental phone misalignment"
    requirement. Vehicle forward speed is encoded in the vibration signature
    (engine/road/wheel harmonics) that these channels expose.
    """
    acc = df[["ax", "ay", "az"]].to_numpy(float)
    gyr = df[["gx", "gy", "gz"]].to_numpy(float)
    grav = df[["grx", "gry", "grz"]].to_numpy(float)

    gnorm = np.linalg.norm(grav, axis=1, keepdims=True)
    gnorm = np.clip(gnorm, 1e-6, None)
    ghat = grav / gnorm                              # unit "down" per sample

    lin = acc - grav                                 # gravity-free linear accel
    a_vert = np.sum(lin * ghat, axis=1, keepdims=True)
    a_horz = lin - a_vert * ghat
    yaw = np.sum(gyr * ghat, axis=1, keepdims=True)  # vehicle yaw rate (rad/s)
    g_horz = gyr - yaw * ghat

    feats = np.concatenate([
        np.linalg.norm(lin, axis=1, keepdims=True),      # |linear accel|
        a_vert,                                          # vertical accel
        np.linalg.norm(a_horz, axis=1, keepdims=True),   # |horizontal accel|
        np.linalg.norm(gyr, axis=1, keepdims=True),      # |gyro|
        yaw,                                             # yaw rate
        np.linalg.norm(g_horz, axis=1, keepdims=True),   # |horizontal gyro|
    ], axis=1)
    assert feats.shape[1] == C.NUM_FEATURES
    return feats.astype(np.float32)


def make_windows(feats, target, window=C.WINDOW, stride=1):
    """Slice [T, C] features into [N, window, C]; label = speed at window end."""
    T = len(feats)
    if T < window:
        return np.empty((0, window, feats.shape[1]), np.float32), np.empty((0,), np.float32)
    idx = np.arange(0, T - window + 1, stride)
    X = np.stack([feats[i:i + window] for i in idx]).astype(np.float32)
    y = target[idx + window - 1].astype(np.float32)
    return X, y


def build_split():
    """
    Load all drives, split by drive, and return windowed train/val + the raw
    per-drive frames for test (needed intact for dead-reckoning).
    """
    pairs = list_pairs()
    train_frames, val_frames, test_frames = [], [], {}
    for key, (vp, sp) in pairs.items():
        if key in C.EXCLUDE_DRIVES:
            continue
        if not key.startswith(C.INCLUDE_PREFIXES):   # within-vehicle (Driver E)
            continue
        df = load_pair(key, vp, sp)
        if len(df) < C.WINDOW + 1:
            continue
        if key in C.TEST_DRIVES:
            test_frames[key] = df
        elif key in C.VAL_DRIVES:
            val_frames.append(df)
        else:
            train_frames.append(df)
    return train_frames, val_frames, test_frames
