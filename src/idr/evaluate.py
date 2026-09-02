"""
Generate the proposal deliverables: dead-reckoning position plots over
simulated GNSS blackouts, plus a drift-vs-distance summary.

Produces PNGs in artifacts/figures/ and prints a drift table.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import dataset as D
from . import dead_reckoning as DR

CAL = 250  # 25 s pre-blackout calibration window (GNSS still available)


def _straightest_segment(df, dist_m, min_speed=3.0):
    """Find the start index of the segment covering ~dist_m with the least
    *total* heading variation (a tunnel/underpass-like straight run)."""
    sp = df["speed_ms"].to_numpy()
    cum = np.cumsum(sp) / C.FS
    course = np.unwrap(np.radians(df["course_deg"].to_numpy()))
    abs_turn = np.concatenate([[0], np.cumsum(np.abs(np.diff(course)))])
    n = len(df)
    best = None
    s = CAL
    while s < n - 5:
        e = int(np.searchsorted(cum, cum[s] + dist_m))
        if e >= n or e - s < 5 or sp[s:e].mean() < min_speed or sp[s - CAL:s].min() < 1.0:
            s += int(C.FS * 3)
            continue
        turn = abs_turn[e - 1] - abs_turn[s]          # total absolute turning
        if best is None or turn < best[2]:
            best = (s, e, turn)
        s += int(C.FS * 2)
    return best


def plot_blackout(df, s, e, title, fname, speed_mode="const", heading_mode="gyro"):
    cal = DR.calibrate(df, slice(s - CAL, s))
    res = DR.dead_reckon(df, cal, slice(s, e), speed_mode=speed_mode,
                         heading_mode=heading_mode)
    st = DR.drift_stats(res)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(res["gt_e"], res["gt_n"], "g-", lw=2.5, label="Ground truth (GNSS)")
    ax[0].plot(res["east"], res["north"], "r--", lw=2, label="Dead reckoning (IMU only)")
    ax[0].plot(res["gt_e"][0], res["gt_n"][0], "ko", ms=8, label="Blackout start")
    ax[0].plot(res["gt_e"][-1], res["gt_n"][-1], "g^", ms=9)
    ax[0].plot(res["east"][-1], res["north"][-1], "r^", ms=9)
    ax[0].plot([res["gt_e"][-1], res["east"][-1]],
               [res["gt_n"][-1], res["north"][-1]], "b:", lw=1)
    ax[0].set_aspect("equal", "datalim")
    ax[0].set_xlabel("East (m)"); ax[0].set_ylabel("North (m)")
    ax[0].legend(loc="best", fontsize=9)
    ax[0].set_title(f"{title}\nDistance {st['distance_m']:.0f} m | "
                    f"final drift {st['final_error_m']:.1f} m "
                    f"({st['drift_pct']:.1f}%)")
    ax[0].grid(alpha=0.3)

    t = np.arange(len(res["v"])) / C.FS
    ax[1].plot(t, res["v_true"] * 3.6, "g-", label="True speed")
    ax[1].plot(t, res["v"] * 3.6, "r--", label="DR speed")
    ax[1].set_xlabel("Time into blackout (s)"); ax[1].set_ylabel("Speed (km/h)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    ax[1].set_title("Speed during blackout")

    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, fname)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}  ->  drift {st['drift_pct']:.1f}% "
          f"({st['final_error_m']:.1f} m over {st['distance_m']:.0f} m)")
    return st


def main():
    print("Generating position plots ...")
    # 1) Short blackout (benchmark: <5 m over 50 m) on a motorway drive
    df = D.load_pair("vfa02", *D.list_pairs()["vfa02"])
    seg = _straightest_segment(df, 50)
    plot_blackout(df, seg[0], seg[1], "50 m GNSS blackout (motorway)",
                  "blackout_50m.png", "const", "gyro")

    # 2) ~1 km straight 'tunnel-like' blackout on a low-turn drive
    dfw = D.load_pair("vw2", *D.list_pairs()["vw2"])
    seg = _straightest_segment(dfw, 1000)
    plot_blackout(dfw, seg[0], seg[1],
                  "1 km GNSS blackout (tunnel-like straight run)",
                  "blackout_1km_tunnel.png", "const", "hold")

    # 3) Drift-vs-distance curve across all test drives (the honest full picture)
    drift_curve()
    print("Done.")


def drift_curve():
    """Median drift% vs blackout distance over all test drives -> shows where
    inertial-only DR meets <10% and where map-matching becomes necessary."""
    drives = {k: D.load_pair(k, *D.list_pairs()[k]) for k in C.TEST_DRIVES}
    dists = [50, 100, 200, 350, 500, 750, 1000]
    med, frac = [], []
    for dist_m in dists:
        ds = []
        for df in drives.values():
            sp = df["speed_ms"].to_numpy(); cum = np.cumsum(sp) / C.FS; n = len(df)
            s = CAL
            while s < n - 5:
                e = int(np.searchsorted(cum, cum[s] + dist_m))
                if e >= n or e - s < 5 or sp[s:e].mean() < 3 or sp[s - CAL:s].min() < 1:
                    s += int(C.FS * 5); continue
                cal = DR.calibrate(df, slice(s - CAL, s))
                res = DR.dead_reckon(df, cal, slice(s, e), "const", heading_mode="gyro")
                ds.append(DR.drift_stats(res)["drift_pct"]); s = e
        ds = np.array(ds)
        med.append(np.median(ds)); frac.append(100 * (ds < 10).mean())
        print(f"  {dist_m:4d} m: median drift {np.median(ds):4.1f}%  "
              f"({frac[-1]:.0f}% of blackouts <10%)")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(dists, med, "o-", color="tab:red", label="Median drift %")
    ax1.axhline(10, ls="--", color="k", lw=1, label="10% benchmark")
    ax1.set_xlabel("GNSS-blackout distance (m)")
    ax1.set_ylabel("Median drift (% of distance)", color="tab:red")
    ax1.set_ylim(0, max(med) * 1.2); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(dists, frac, "s--", color="tab:blue", alpha=0.6)
    ax2.set_ylabel("% blackouts under 10%", color="tab:blue"); ax2.set_ylim(0, 100)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Inertial-only dead-reckoning drift vs blackout distance\n"
                  "(smartphone IMU, const-speed + gyro heading, test drives)")
    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "drift_vs_distance.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
