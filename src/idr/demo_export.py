"""
Export a replay scenario for the browser demo.

Runs a real IO-VNBD drive: GNSS available -> GNSS blackout (tunnel) -> GNSS
recovery, and records three tracks in lat/lon:
  * ground truth (VBOX),
  * plain inertial dead reckoning (drifts), and
  * IDR + map-matching (stays on the road).
Also records speed, per-frame mode and live drift, and the nearby OSM road
network. Output: demo/replay_data.json (consumed by demo/index.html).
"""
import os
import json
import numpy as np

from . import config as C
from . import dataset as D
from . import dead_reckoning as DR
from . import osm
from . import map_matching as MM

CAL = 250          # 25 s pre-blackout calibration (GNSS available)
PRE = 200          # 20 s of GNSS driving shown before the blackout
POST = 120         # 12 s of GNSS recovery shown after
R_EARTH = 6371000.0


def enu_to_latlon(e, n, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(n) / R_EARTH)
    lon = lon0 + np.degrees(np.asarray(e) / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


def _pick_blackout(df, net, dist_m=1000):
    """Choose a curvy 1 km blackout where plain DR fails but map-aided succeeds."""
    sp = df["speed_ms"].to_numpy(); cum = np.cumsum(sp) / C.FS; n = len(df)
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    s = CAL + PRE
    best = None
    while s < n - 5:
        e = int(np.searchsorted(cum, cum[s] + dist_m))
        if e >= n - POST or e - s < 10 or sp[s:e].mean() < 5 or sp[s - CAL:s].min() < 1:
            s += int(C.FS * 6); continue
        cal = DR.calibrate(df, slice(s - CAL, s))
        res = DR.dead_reckon(df, cal, slice(s, e), "const", heading_mode="gyro")
        se, sn = osm.latlon_to_enu(lat[s], lon[s], net.lat0, net.lon0)
        ge, gn = osm.latlon_to_enu(lat[e - 1], lon[e - 1], net.lat0, net.lon0)
        rawd = np.hypot(se + res["east"][-1] - ge, sn + res["north"][-1] - gn)
        mp = MM.map_aided_dr(net, df, cal, s, e, "const")
        if mp is None:
            s += int(C.FS * 6); continue
        mmd = np.hypot(mp[-1][0] - ge, mp[-1][1] - gn)
        course = np.unwrap(np.radians(df["course_deg"].to_numpy()[s:e]))
        turn = np.degrees(np.abs(np.diff(course)).sum())
        rawpct = 100 * rawd / res["dist"]; mmpct = 100 * mmd / res["dist"]
        # demo score: big free-DR failure, tight map-aided, some turning for drama
        turn_bonus = min(turn, 400) * 0.05 if 60 < turn < 720 else -20
        score = min(rawpct, 120) - 8 * mmpct + turn_bonus
        best = best or []
        best.append((score, s, e, cal, res, mp, rawpct, mmpct, turn))
        s = e
    if not best:
        raise RuntimeError("no valid blackout windows")
    best.sort(key=lambda z: z[0], reverse=True)
    top = best[0]
    print(f"  demo blackout: plain DR {top[6]:.0f}% | map-aided {top[7]:.1f}% "
          f"| turning {top[8]:.0f} deg")
    return top[1:6]


def export(drive="vfa02", out="demo/replay_data.json"):
    df = D.load_pair(drive, *D.list_pairs()[drive])
    net = osm.network_for_drive(df, drive)
    s, e, cal, res, mp = _pick_blackout(df, net)
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    sp = df["speed_ms"].to_numpy()
    a = s - PRE; b = min(e + POST, len(df))

    # blackout free-DR + map-aided tracks in net ENU -> lat/lon
    se, sn = osm.latlon_to_enu(lat[s], lon[s], net.lat0, net.lon0)
    free_e = se + res["east"]; free_n = sn + res["north"]
    free_lat, free_lon = enu_to_latlon(free_e, free_n, net.lat0, net.lon0)
    mm_lat, mm_lon = enu_to_latlon(mp[:, 0], mp[:, 1], net.lat0, net.lon0)

    frames = []
    step = 2                                   # 5 Hz frames (smooth, small file)
    for i in range(a, b, step):
        gt = [float(lat[i]), float(lon[i])]
        if i < s:                              # GNSS available (all aligned)
            free = gt; mm = gt; mode = "GNSS"; d_free = d_mm = 0.0
        elif i < e:                            # blackout
            j = i - s
            free = [float(free_lat[j]), float(free_lon[j])]
            mm = [float(mm_lat[j]), float(mm_lon[j])]
            mode = "BLACKOUT"
            ge, gn = osm.latlon_to_enu(lat[i], lon[i], net.lat0, net.lon0)
            d_free = float(np.hypot(free_e[j] - ge, free_n[j] - gn))
            d_mm = float(np.hypot(mp[j, 0] - ge, mp[j, 1] - gn))
        else:                                  # GNSS recovered
            free = gt; mm = gt; mode = "GNSS"; d_free = d_mm = 0.0
        frames.append({"gt": gt, "free": free, "mm": mm, "mode": mode,
                       "spd": round(float(sp[i]) * 3.6, 1),
                       "df": round(d_free, 1), "dm": round(d_mm, 1)})

    # nearby OSM roads (lat/lon polylines) for an offline-safe road layer
    minla, maxla = lat[a:b].min() - 0.01, lat[a:b].max() + 0.01
    minlo, maxlo = lon[a:b].min() - 0.01, lon[a:b].max() + 0.01
    roads = []
    for line in net.edge_lines:
        xs, ys = line.xy                       # xs = east, ys = north
        la, lo = enu_to_latlon(np.array(xs), np.array(ys), net.lat0, net.lon0)
        if minla < la[0] < maxla and minlo < lo[0] < maxlo:
            roads.append([[round(float(la[0]), 6), round(float(lo[0]), 6)],
                          [round(float(la[1]), 6), round(float(lo[1]), 6)]])

    dist = float(res["dist"])
    payload = {
        "drive": drive, "fps": 5, "blackout_dist_m": round(dist),
        "final_free_pct": round(100 * frames_final(frames, "df") / dist, 1),
        "final_mm_pct": round(100 * frames_final(frames, "dm") / dist, 1),
        "center": [float(np.mean(lat[a:b])), float(np.mean(lon[a:b]))],
        "frames": frames, "roads": roads,
    }
    os.makedirs(os.path.dirname(os.path.join(C.ROOT, out)), exist_ok=True)
    path = os.path.join(C.ROOT, out)
    with open(path, "w") as f:
        json.dump(payload, f)
    # also emit a JS file so the UI works by double-clicking index.html (no server)
    js_path = os.path.splitext(path)[0] + ".js"
    with open(js_path, "w") as f:
        f.write("window.REPLAY_DATA = ")
        json.dump(payload, f)
        f.write(";")
    print(f"Exported {len(frames)} frames, {len(roads)} road segments -> {path}")
    print(f"  blackout {dist:.0f} m | plain IMU final {payload['final_free_pct']}% "
          f"| IDR+map final {payload['final_mm_pct']}%")


def frames_final(frames, key):
    for fr in reversed(frames):
        if fr["mode"] == "BLACKOUT":
            return fr[key]
    return 0.0


if __name__ == "__main__":
    export()
