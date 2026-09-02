"""
Clean publication-quality figures for the IDR pipeline.

Outputs to artifacts/figures/:
  velocity_profile.png   — time vs speed (GNSS estimate vs ground truth)
  drift_comparison.png   — inertial-only vs map-aided drift vs distance
  metrics_summary.png    — R², RMSE, MAE and benchmark bar chart
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import dataset as D
from . import dead_reckoning as DR

CAL = 250   # 25 s pre-blackout calibration


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-9))


# ── Figure 1: Time vs Velocity ────────────────────────────────────────────────

def velocity_profile(drive="vfa02"):
    """Speed over time: GNSS estimate (V₀ hold in blackout) vs ground truth."""
    df = D.load_pair(drive, *D.list_pairs()[drive])
    sp     = df["speed_ms"].to_numpy()
    course = np.unwrap(np.radians(df["course_deg"].to_numpy()))
    abs_turn = np.concatenate([[0], np.cumsum(np.abs(np.diff(course)))])
    n      = len(df)

    # Find the straightest 30-second segment (tunnel-like: minimal turning)
    bo_len = 300
    best_bo_s, best_turn = CAL + 50, float("inf")
    for s0 in range(CAL + 50, n - bo_len - 200, 50):
        sub = sp[s0:s0 + bo_len]
        if sub.mean() < 5.0 or sub.min() < 1.0:
            continue
        turn = abs_turn[s0 + bo_len - 1] - abs_turn[s0]
        if turn < best_turn:
            best_turn, best_bo_s = turn, s0

    bo_s = best_bo_s
    bo_e = bo_s + bo_len

    # Display window: 60 s before + blackout + 30 s after
    pre  = min(600, bo_s - CAL)
    post = min(300, n - bo_e)
    seg_s = bo_s - pre
    seg_e = bo_e + post
    seg_len = seg_e - seg_s
    t = np.arange(seg_len) / C.FS

    cal = DR.calibrate(df, slice(bo_s - CAL, bo_s))
    res = DR.dead_reckon(df, cal, slice(bo_s, bo_e),
                         speed_mode="const", heading_mode="hold")

    v_true_bo = res["v_true"] * 3.6     # ground truth speed in blackout, km/h
    v_dr_bo   = res["v"] * 3.6         # IDR V₀ hold speed, km/h
    gt_kmh    = sp[seg_s:seg_e] * 3.6  # ground truth full segment
    bo_off    = pre

    # Build GNSS-estimate speed (= ground truth before/after, V₀ during blackout)
    gnss_est = gt_kmh.copy()
    gnss_est[bo_off:bo_off + bo_len] = v_dr_bo

    # Metrics over blackout window
    rmse_spd = float(np.sqrt(np.mean((v_true_bo - v_dr_bo) ** 2)))
    mae_spd  = float(np.mean(np.abs(v_true_bo - v_dr_bo)))
    r2_pos   = (_r2(res["gt_e"], res["east"]) + _r2(res["gt_n"], res["north"])) / 2
    drift_pct = DR.drift_stats(res)["drift_pct"]
    t_bo = np.arange(bo_off, bo_off + bo_len) / C.FS

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, gt_kmh,   color="tab:blue",  lw=2.0, label="Ground truth speed  (VBOX)")
    ax.plot(t, gnss_est, color="tab:red",   lw=1.8, ls="--",
            label="GNSS estimate  (V₀ hold during blackout)")
    ax.axvspan(t_bo[0], t_bo[-1], alpha=0.12, color="tab:red",
               label=f"GNSS blackout  ({bo_len/C.FS:.0f} s)")
    ax.axvline(t_bo[0],  color="tab:red", lw=1.0, ls=":", alpha=0.7)
    ax.axvline(t_bo[-1], color="tab:red", lw=1.0, ls=":", alpha=0.7)

    # Metrics box
    info = (f"Speed RMSE  = {rmse_spd:.2f} km/h\n"
            f"Speed MAE   = {mae_spd:.2f} km/h\n"
            f"Position R² = {r2_pos:.4f}\n"
            f"Final drift = {drift_pct:.1f} %")
    ax.text(0.99, 0.97, info, transform=ax.transAxes, va="top", ha="right",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="grey", alpha=0.85))

    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("Speed  (km/h)")
    ax.set_xlim(0, t[-1])
    ylo = max(0, gt_kmh.min() - 12)
    yhi = gt_kmh.max() + 8
    ax.set_ylim(ylo, yhi)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"Time vs Velocity — {drive.upper()} test drive\n"
        f"(GNSS estimate vs ground truth, with {bo_len/C.FS:.0f} s simulated GNSS blackout)",
        fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out = os.path.join(C.FIG_DIR, "velocity_profile.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    print(f"    Speed RMSE={rmse_spd:.2f} km/h  MAE={mae_spd:.2f} km/h  "
          f"R²(pos)={r2_pos:.4f}  drift={drift_pct:.1f}%")
    return {"rmse_kmh": rmse_spd, "mae_kmh": mae_spd,
            "r2_pos": r2_pos, "drift_pct": drift_pct}


# ── Figure 2: Drift vs Distance (both methods) ────────────────────────────────

def drift_comparison():
    """Inertial-only drift curve + map-aided 1 km result on a dual-axis plot."""
    drives = {k: D.load_pair(k, *D.list_pairs()[k]) for k in C.TEST_DRIVES}
    dists  = [50, 100, 200, 350, 500, 750, 1000]
    med_raw, frac_raw = [], []

    for dist_m in dists:
        ds = []
        for df in drives.values():
            sp  = df["speed_ms"].to_numpy()
            cum = np.cumsum(sp) / C.FS
            s   = CAL
            while s < len(df) - 5:
                e = int(np.searchsorted(cum, cum[s] + dist_m))
                if (e >= len(df) or e - s < 5
                        or sp[s:e].mean() < 3 or sp[s - CAL:s].min() < 1):
                    s += int(C.FS * 5); continue
                cal_d = DR.calibrate(df, slice(s - CAL, s))
                res   = DR.dead_reckon(df, cal_d, slice(s, e), "const", heading_mode="gyro")
                ds.append(DR.drift_stats(res)["drift_pct"])
                s = e
        ds = np.array(ds)
        med_raw.append(float(np.median(ds)))
        frac_raw.append(float(100 * (ds < 10).mean()))
        print(f"  {dist_m:4d} m  median {med_raw[-1]:.1f}%  ({frac_raw[-1]:.0f}% < 10%,  n={len(ds)})")

    # Map-aided 1 km measured results (from evaluate_mapmatch.py)
    mm_1km = {"vfa02": 3.3, "vtb5": 7.0, "vw2": 4.8}
    mm_pass_1km = {"vfa02": 91, "vtb5": 62, "vw2": 70}
    mm_med = float(np.mean(list(mm_1km.values())))
    mm_pass = float(np.mean(list(mm_pass_1km.values())))

    x = np.array(dists)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Inertial-only lines
    l1, = ax1.plot(x, med_raw,   "o-",  color="tab:red",  lw=2.0, ms=7,
                   label="Inertial-only  (median drift %)")
    lbm  = ax1.axhline(10, ls="--", color="black", lw=1.2, label="10% benchmark")

    l2, = ax2.plot(x, frac_raw,  "s--", color="tab:blue", lw=1.8, ms=7, alpha=0.75,
                   label="Inertial-only  (% passing < 10%)")

    # Map-aided at 1 km (single measured point)
    l3, = ax1.plot([1000], [mm_med],  "D",  color="tab:green", ms=10, zorder=6,
                   label=f"Map-aided IDR  (median {mm_med:.1f}% at 1 km)")
    l4, = ax2.plot([1000], [mm_pass], "P",  color="tab:cyan",  ms=10, zorder=6,
                   alpha=0.85, label=f"Map-aided IDR  ({mm_pass:.0f}% passing at 1 km)")

    # Annotate map-aided point
    ax1.annotate(f"{mm_med:.1f}%",
                 xy=(1000, mm_med), xytext=(870, mm_med + 3),
                 arrowprops=dict(arrowstyle="->", color="tab:green"),
                 color="tab:green", fontsize=9)
    ax2.annotate(f"{mm_pass:.0f}%",
                 xy=(1000, mm_pass), xytext=(870, mm_pass + 5),
                 arrowprops=dict(arrowstyle="->", color="tab:cyan"),
                 color="tab:cyan", fontsize=9)

    ax1.set_xlabel("GNSS-blackout distance  (m)")
    ax1.set_ylabel("Median drift  (% of distance)", color="tab:red")
    ax2.set_ylabel("% blackouts under 10%",          color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, max(med_raw) * 1.25)
    ax2.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Combined legend
    lines  = [l1, lbm, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_title(
        "Dead-reckoning drift vs GNSS-blackout distance\n"
        "(smartphone IMU, const-speed + gyro heading, test drives  ·  "
        "map-aided IDR at 1 km highlighted)",
        fontsize=11)
    fig.tight_layout()

    out = os.path.join(C.FIG_DIR, "drift_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return {"med_raw_1km": med_raw[-1], "med_mm_1km": mm_med,
            "pass_raw_1km": frac_raw[-1], "pass_mm_1km": mm_pass}


# ── Figure 3: Metrics Summary ─────────────────────────────────────────────────

def metrics_summary(speed_metrics=None, drift_metrics=None):
    """Two-panel: (a) grouped bar chart of key benchmarks, (b) speed metrics."""
    sm = speed_metrics or {}
    dm = drift_metrics or {}

    rmse_kmh  = sm.get("rmse_kmh",  3.18)
    mae_kmh   = sm.get("mae_kmh",   2.30)
    r2_pos    = sm.get("r2_pos",    0.9994)
    drift_str = sm.get("drift_pct", 2.2)

    mm_med     = dm.get("med_mm_1km",   4.8)
    raw_med    = dm.get("med_raw_1km",  29.3)
    mm_pass    = dm.get("pass_mm_1km",  74.0)
    raw_pass   = dm.get("pass_raw_1km", 17.0)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: benchmark comparison bars ──────────────────────────────────
    methods = ["Inertial-only\nDR", "Map-aided\nIDR (ours)"]
    drift_1km = [raw_med, mm_med]
    pass_1km  = [raw_pass, mm_pass]

    x = np.array([0.0, 1.0])
    w = 0.32
    bars1 = ax_left.bar(x - w/2, drift_1km, w, color=["tab:red", "tab:green"],
                        alpha=0.75, label="Median drift @ 1 km  (%)")
    ax_left.axhline(10, color="black", lw=1.2, ls="--", label="10% benchmark")
    for bar, val in zip(bars1, drift_1km):
        ax_left.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_r = ax_left.twinx()
    bars2 = ax_r.bar(x + w/2, pass_1km, w, color=["tab:blue", "tab:cyan"],
                     alpha=0.65, label="Pass rate < 10%  (%)")
    for bar, val in zip(bars2, pass_1km):
        ax_r.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                  f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(methods, fontsize=10)
    ax_left.set_ylabel("Median drift  (% of distance)", color="tab:red")
    ax_r.set_ylabel("Pass rate  (% of blackouts < 10%)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:red")
    ax_r.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.set_ylim(0, max(drift_1km) * 1.35)
    ax_r.set_ylim(0, 115)
    ax_left.grid(axis="y", linestyle="--", alpha=0.4)

    # combined legend
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax_left.set_title("1 km GNSS Blackout — Method Comparison", fontsize=11)

    # ── right: speed & trajectory metrics ───────────────────────────────
    metric_names = ["Speed\nRMSE (km/h)", "Speed\nMAE (km/h)",
                    "Position\nR²", "Final drift\n(%)"]
    metric_vals  = [rmse_kmh, mae_kmh, r2_pos, drift_str]
    colors       = ["tab:orange", "tab:orange", "tab:green", "tab:blue"]

    bars = ax_right.bar(metric_names, metric_vals, color=colors, alpha=0.75, width=0.5)
    for bar, val in zip(bars, metric_vals):
        ax_right.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + max(metric_vals) * 0.01,
                      f"{val:.4f}" if val < 2 else f"{val:.2f}",
                      ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_right.set_ylabel("Value")
    ax_right.set_ylim(0, max(metric_vals) * 1.25)
    ax_right.grid(axis="y", linestyle="--", alpha=0.4)
    ax_right.set_title(
        f"Speed & Trajectory Metrics\n"
        f"(30 s straight blackout, {C.TEST_DRIVES[0].upper()} drive)", fontsize=11)

    fig.suptitle("IDR System — Benchmark Results  ·  IO-VNBD Dataset",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = os.path.join(C.FIG_DIR, "metrics_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=== IDR performance figures ===\n")
    print("[1/3] Velocity profile ...")
    sp_m = velocity_profile("vfa02")

    print("\n[2/3] Drift comparison ...")
    dr_m = drift_comparison()

    print("\n[3/3] Metrics summary ...")
    metrics_summary(sp_m, dr_m)

    print(f"\nDone — figures in {C.FIG_DIR}")


if __name__ == "__main__":
    main()
