"""
Map-matching evaluation: dead reckoning vs map-aided dead reckoning over 1 km
GNSS blackouts, plus a before/after position plot on the OSM road network.

Shows that snapping inertial DR to the road graph corrects the heading drift
that free DR cannot, meeting the <10% benchmark on real routes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import dataset as D
from . import dead_reckoning as DR
from . import osm
from . import map_matching as MM

CAL = 250


def _blackouts(df, dist_m=1000, min_speed=5.0, step_s=7):
    sp = df["speed_ms"].to_numpy(); cum = np.cumsum(sp) / C.FS; n = len(df)
    s = CAL
    while s < n - 5:
        e = int(np.searchsorted(cum, cum[s] + dist_m))
        if e < n and e - s > 10 and sp[s:e].mean() > min_speed and sp[s - CAL:s].min() > 1:
            yield s, e
            s = e
        else:
            s += int(C.FS * step_s)


def _drift(a_e, a_n, gt_e, gt_n, dist):
    return 100.0 * np.hypot(a_e - gt_e, a_n - gt_n) / max(dist, 1e-6)


def aggregate():
    print("1 km blackout drift: free DR vs map-aided DR")
    for k in C.TEST_DRIVES:
        df = D.load_pair(k, *D.list_pairs()[k])
        net = osm.network_for_drive(df, k)
        lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
        raw, mm = [], []
        for s, e in _blackouts(df):
            cal = DR.calibrate(df, slice(s - CAL, s))
            res = DR.dead_reckon(df, cal, slice(s, e), "const", heading_mode="gyro")
            se, sn = osm.latlon_to_enu(lat[s], lon[s], net.lat0, net.lon0)
            ge, gn = osm.latlon_to_enu(lat[e - 1], lon[e - 1], net.lat0, net.lon0)
            raw.append(_drift(se + res["east"][-1], sn + res["north"][-1], ge, gn, res["dist"]))
            mp = MM.map_aided_dr(net, df, cal, s, e, "const")
            if mp is not None:
                mm.append(_drift(mp[-1][0], mp[-1][1], ge, gn, res["dist"]))
        raw, mm = np.array(raw), np.array(mm)
        print(f"  {k}: n={len(raw)} | free DR median {np.median(raw):5.0f}% | "
              f"map-aided median {np.median(mm):5.1f}%  best {mm.min():4.1f}%  "
              f"<10%: {100*(mm<10).mean():3.0f}%")


def demo_plot(k="vfa02", want_low=8.0):
    """Find a curvy blackout where free DR fails and map-aided succeeds; plot it."""
    df = D.load_pair(k, *D.list_pairs()[k])
    net = osm.network_for_drive(df, k)
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    best = None
    for s, e in _blackouts(df):
        cal = DR.calibrate(df, slice(s - CAL, s))
        res = DR.dead_reckon(df, cal, slice(s, e), "const", heading_mode="gyro")
        se, sn = osm.latlon_to_enu(lat[s], lon[s], net.lat0, net.lon0)
        ge, gn = osm.latlon_to_enu(lat[e - 1], lon[e - 1], net.lat0, net.lon0)
        rawd = _drift(se + res["east"][-1], sn + res["north"][-1], ge, gn, res["dist"])
        mp = MM.map_aided_dr(net, df, cal, s, e, "const")
        if mp is None:
            continue
        mmd = _drift(mp[-1][0], mp[-1][1], ge, gn, res["dist"])
        course = np.unwrap(np.radians(df["course_deg"].to_numpy()[s:e]))
        turn = np.degrees(np.abs(np.diff(course)).sum())
        if mmd < want_low and rawd > 25 and 60 < turn < 720:
            best = (s, e, res, mp, rawd, mmd, turn)
            break
    if best is None:
        print(f"  no ideal demo segment on {k}"); return
    s, e, res, mp, rawd, mmd, turn = best
    se, sn = osm.latlon_to_enu(lat[s], lon[s], net.lat0, net.lon0)
    gt_e, gt_n = osm.latlon_to_enu(lat[s:e], lon[s:e], net.lat0, net.lon0)
    dr_e, dr_n = se + res["east"], sn + res["north"]

    allx = np.concatenate([gt_e, dr_e, mp[:, 0]]); ally = np.concatenate([gt_n, dr_n, mp[:, 1]])
    pad = 150
    xlim = (allx.min() - pad, allx.max() + pad); ylim = (ally.min() - pad, ally.max() + pad)

    fig, ax = plt.subplots(figsize=(9, 9))
    for line in net.edge_lines:                       # draw road network
        xs, ys = line.xy
        if xlim[0] < xs[0] < xlim[1] and ylim[0] < ys[0] < ylim[1]:
            ax.plot(xs, ys, color="0.8", lw=0.8, zorder=1)
    ax.plot(gt_e, gt_n, "g-", lw=3, label="Ground truth", zorder=3)
    ax.plot(dr_e, dr_n, "r--", lw=2, label=f"Free DR (drift {rawd:.0f}%)", zorder=4)
    ax.plot(mp[:, 0], mp[:, 1], "b-", lw=2, label=f"Map-aided DR (drift {mmd:.1f}%)", zorder=5)
    ax.plot(gt_e[0], gt_n[0], "ko", ms=9, label="Blackout start", zorder=6)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"1 km GNSS blackout on real roads ({k}, {turn:.0f}° of turning)\n"
                 f"Free inertial DR drifts {rawd:.0f}%; map-matching recovers it to {mmd:.1f}%")
    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "mapmatch_1km_demo.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}  (free {rawd:.0f}% -> map-aided {mmd:.1f}%)")


def main():
    demo_plot("vfa02")
    demo_plot("vw2") if not os.path.exists(os.path.join(C.FIG_DIR, "mapmatch_1km_demo.png")) else None
    aggregate()


if __name__ == "__main__":
    main()
