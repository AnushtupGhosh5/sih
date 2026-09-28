"""
Dead-reckoning evaluation on IO-VNBD test drives.

Realistic GNSS-blackout model: at blackout onset the last GNSS speed (V0) and
heading (H0) are known. During the blackout we propagate an INS solution using
the smartphone IMU only, having *calibrated* the forward axis, accelerometer
bias and gyroscope bias on a short pre-blackout window where GNSS is available
(exactly what a phone can do just before entering a tunnel).

We report drift as a percentage of distance travelled (the SIH benchmark), and
provide oracle variants to attribute error to speed vs heading.
"""
import numpy as np
from scipy.signal import butter, filtfilt

from . import config as C

R_EARTH = 6371000.0


def latlon_to_enu(lat, lon, lat0, lon0):
    """Equirectangular local ENU (metres) around a reference point."""
    lat0r = np.radians(lat0)
    x = np.radians(lon - lon0) * R_EARTH * np.cos(lat0r)   # east
    y = np.radians(lat - lat0) * R_EARTH                    # north
    return x, y


def _lowpass(x, fc=0.5):
    b, a = butter(2, fc / (C.FS / 2), "low")
    return filtfilt(b, a, x)


def _horiz_basis(ghat):
    ref = np.array([1.0, 0, 0])
    if abs(ghat @ ref) > 0.9:
        ref = np.array([0, 1.0, 0])
    e1 = np.cross(ghat, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ghat, e1)
    return e1, e2


def calibrate(df, cal_slice):
    """
    Estimate forward axis, accel bias and gyro-yaw bias from a GNSS-available
    pre-blackout window.
    """
    acc = df[["ax", "ay", "az"]].to_numpy()[cal_slice]
    gyr = df[["gx", "gy", "gz"]].to_numpy()[cal_slice]
    grav = df[["grx", "gry", "grz"]].to_numpy()[cal_slice]
    sp = df["speed_ms"].to_numpy()[cal_slice]
    head = np.radians(df["course_deg"].to_numpy()[cal_slice])

    ghat = grav.mean(0); ghat /= np.linalg.norm(ghat)
    e1, e2 = _horiz_basis(ghat)
    lin = acc - grav
    c1, c2 = _lowpass(lin @ e1), _lowpass(lin @ e2)
    dv = _lowpass(np.gradient(sp) * C.FS)

    # forward axis = horizontal direction whose accel best matches GNSS dv/dt
    best_r, best_th = -2, 0.0
    for th in np.linspace(0, 2 * np.pi, 180, endpoint=False):
        af = np.cos(th) * c1 + np.sin(th) * c2
        s = af.std()
        if s < 1e-6:
            continue
        r = np.corrcoef(af, dv)[0, 1]
        if r > best_r:
            best_r, best_th = r, th
    fwd = np.cos(best_th) * e1 + np.sin(best_th) * e2

    a_fwd = lin @ fwd
    accel_bias = np.mean(_lowpass(a_fwd) - dv)     # residual forward-accel bias

    # gyro yaw: sign so integrated yaw follows GNSS heading; bias = mean residual
    ghat_t = grav / np.clip(np.linalg.norm(grav, axis=1, keepdims=True), 1e-6, None)
    yaw = np.sum(gyr * ghat_t, axis=1)
    dhead = np.unwrap(head)
    dhead_rate = np.gradient(dhead) * C.FS
    sign = np.sign(np.corrcoef(yaw, dhead_rate)[0, 1] or 1.0)
    if sign == 0:
        sign = 1.0
    yaw_bias = np.mean(sign * yaw - dhead_rate)

    # compass azimuth -> vehicle-heading offset (phone yaw on the mount)
    azi = np.radians(df["az_deg"].to_numpy()[cal_slice])
    off = np.angle(np.mean(np.exp(1j * (head - azi))))   # circular-mean offset

    return {"ghat": ghat, "fwd": fwd, "accel_bias": accel_bias,
            "yaw_sign": sign, "yaw_bias": yaw_bias, "cal_corr": best_r,
            "az_offset": off}


def dead_reckon(df, cal, bo_slice, speed_mode="ins", speed_series=None,
                heading_mode="fused", fuse_gain=0.02):
    """
    Propagate INS through blackout indices `bo_slice`.
      speed_mode  : 'ins' | 'const' | 'oracle' | 'series'
      heading_mode: 'fused'  (gyro integration corrected toward compass azimuth),
                    'gyro'   (pure gyro integration),
                    'compass'(azimuth + calibrated offset only),
                    'oracle' (true GNSS heading).
    Returns estimated ENU trajectory and the ground-truth ENU trajectory.
    """
    idx = np.arange(bo_slice.start, bo_slice.stop)
    acc = df[["ax", "ay", "az"]].to_numpy()[idx]
    gyr = df[["gx", "gy", "gz"]].to_numpy()[idx]
    grav = df[["grx", "gry", "grz"]].to_numpy()[idx]
    sp_true = df["speed_ms"].to_numpy()[idx]
    lat = df["lat"].to_numpy()[idx]; lon = df["lon"].to_numpy()[idx]
    head = np.radians(df["course_deg"].to_numpy()[idx])
    azi = np.radians(df["az_deg"].to_numpy()[idx]) + cal["az_offset"]

    dt = 1.0 / C.FS
    lin = acc - grav
    a_fwd = lin @ cal["fwd"] - cal["accel_bias"]
    a_fwd = _lowpass(a_fwd)
    # yaw rate about the *live* vertical (per-sample gravity) so phone
    # pitch/roll on bumps does not leak into heading.
    ghat_t = grav / np.clip(np.linalg.norm(grav, axis=1, keepdims=True), 1e-6, None)
    yaw = cal["yaw_sign"] * np.sum(gyr * ghat_t, axis=1) - cal["yaw_bias"]

    V0 = sp_true[0]
    H0 = head[0]

    # speed profile
    if speed_mode == "const":
        v = np.full(len(idx), V0)
    elif speed_mode == "oracle":
        v = sp_true.copy()
    elif speed_mode == "series":
        v = np.clip(speed_series, 0, None)
    else:  # ins
        v = np.empty(len(idx)); v[0] = V0
        for t in range(1, len(idx)):
            v[t] = max(0.0, v[t - 1] + a_fwd[t] * dt)

    # heading profile
    h = np.empty(len(idx)); h[0] = H0
    if heading_mode == "oracle":
        h = head.copy()
    elif heading_mode == "compass":
        h = azi.copy()
    elif heading_mode == "hold":
        h = np.full(len(idx), H0)            # constant heading (straight-tunnel prior)
    else:
        for t in range(1, len(idx)):
            h[t] = h[t - 1] + yaw[t] * dt          # gyro propagation
            if heading_mode == "fused":            # pull toward compass
                err = np.angle(np.exp(1j * (azi[t] - h[t])))
                h[t] += fuse_gain * err

    # integrate position in ENU (heading measured from North, clockwise)
    east = np.zeros(len(idx)); north = np.zeros(len(idx))
    for t in range(1, len(idx)):
        east[t] = east[t - 1] + v[t] * np.sin(h[t]) * dt
        north[t] = north[t - 1] + v[t] * np.cos(h[t]) * dt

    gx, gy = latlon_to_enu(lat, lon, lat[0], lon[0])
    return {"east": east, "north": north, "gt_e": gx, "gt_n": gy,
            "v": v, "v_true": sp_true, "dist": np.sum(sp_true) * dt}


def drift_stats(res):
    final_err = np.hypot(res["east"][-1] - res["gt_e"][-1],
                         res["north"][-1] - res["gt_n"][-1])
    dist = max(res["dist"], 1e-6)
    return {"final_error_m": final_err, "distance_m": dist,
            "drift_pct": 100.0 * final_err / dist}
