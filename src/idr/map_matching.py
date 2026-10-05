"""
HMM map-matching (Newson & Krumm, 2009) to snap a drifting dead-reckoning
trajectory onto the OSM road network.

Idea: the vehicle must be on a road. Each observation (a DR position) has several
candidate road segments. The most likely sequence of roads is found with Viterbi,
where
  * emission  ~ how close the observation is to the candidate road (Gaussian), and
  * transition ~ how well the straight-line step between consecutive observations
                 matches the on-road (routing) distance between candidates.
The route-distance transition enforces road connectivity, so a slowly-rotating DR
drift is pulled back onto the correct connected route -- correcting the heading
error that inertial sensors cannot.
"""
import numpy as np

from . import dead_reckoning as _DR

NEG = -1e9


def _bearing(xa, xb):
    """Compass bearing (rad, from North, clockwise) of segment xa->xb in ENU."""
    return np.arctan2(xb[0] - xa[0], xb[1] - xa[1])


def _ang_diff(a, b):
    return abs(np.angle(np.exp(1j * (a - b))))


def map_aided_dr(net, df, cal, s, e, speed_mode="const", speed_series=None):
    """
    Route-constrained (map-aided) dead reckoning.

    Odometry distance (accurate) advances the vehicle along the road graph; the
    gyro supplies only the short-term relative heading used to pick the correct
    branch at each junction. Heading is reset to the road bearing on every edge,
    so the inertial heading drift that wrecks free DR never accumulates.
    Returns matched positions [K,2] in the network ENU frame.
    """
    from . import osm
    idx = np.arange(s, e)
    dt = 1.0 / _DR.C.FS
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()

    # speed + yaw from the same calibrated inertial stream as free DR
    sp_true = df["speed_ms"].to_numpy()
    acc = df[["ax", "ay", "az"]].to_numpy()[idx]
    gyr = df[["gx", "gy", "gz"]].to_numpy()[idx]
    grav = df[["grx", "gry", "grz"]].to_numpy()[idx]
    ghat = grav / np.clip(np.linalg.norm(grav, axis=1, keepdims=True), 1e-6, None)
    yaw = cal["yaw_sign"] * np.sum(gyr * ghat, axis=1) - cal["yaw_bias"]
    if speed_mode == "oracle":
        v = sp_true[idx]
    elif speed_mode == "series" and speed_series is not None:
        v = np.clip(speed_series, 0, None)
    else:  # const: hold last GNSS speed
        v = np.full(len(idx), sp_true[s])

    # --- anchor onto the road at the blackout start ---
    e0, n0 = osm.latlon_to_enu(lat[s], lon[s], net.lat0, net.lon0)
    H = np.radians(df["course_deg"].to_numpy()[s])
    cands = net.candidates(e0, n0, radius=60)
    if not cands:
        return None
    # choose the candidate edge whose orientation best matches travel heading
    best = None
    for ei, proj, d in cands:
        a, b = net.edges[ei]
        xa, xb = net.node_xy[a], net.node_xy[b]
        for frm, to in ((a, b), (b, a)):
            br = _bearing(net.node_xy[frm], net.node_xy[to])
            score = _ang_diff(br, H) + 0.02 * d
            if best is None or score < best[0]:
                best = (score, frm, to, ei)
    _, frm, to, _ = best
    # distance already travelled along current edge from 'frm'
    xf = np.array(net.node_xy[frm]); xt = np.array(net.node_xy[to])
    seg = xt - xf; seglen = np.hypot(*seg); segdir = seg / max(seglen, 1e-9)
    along = np.clip(np.dot(np.array([e0, n0]) - xf, segdir), 0, seglen)
    heading = _bearing(xf, xt)

    out = np.zeros((len(idx), 2))
    prev = frm
    for t in range(len(idx)):
        heading = heading + yaw[t] * dt          # gyro relative between resets
        remaining = v[t] * dt
        # advance along the graph
        guard = 0
        while remaining > 1e-6 and guard < 50:
            guard += 1
            if along + remaining <= seglen:
                along += remaining; remaining = 0
            else:
                remaining -= (seglen - along)
                # reached node 'to' -> pick next edge
                nbrs = [nb for nb in net.G[to] if nb != prev]
                if not nbrs:
                    nbrs = list(net.G[to])       # dead end: allow U-turn
                if not nbrs:
                    along = seglen; remaining = 0; break
                # choose neighbour whose bearing best matches current heading
                nxt = min(nbrs, key=lambda nb: _ang_diff(_bearing(net.node_xy[to], net.node_xy[nb]), heading))
                prev, frm2, to2 = to, to, nxt
                xf = np.array(net.node_xy[frm2]); xt = np.array(net.node_xy[to2])
                seglen = np.hypot(*(xt - xf))
                to = to2
                along = 0.0
                heading = _bearing(xf, xt)        # RESET heading to road bearing
        pos = xf + (xt - xf) * (along / max(seglen, 1e-9))
        out[t] = pos
    return out


def match(net, obs, sigma_z=25.0, beta=12.0, radius=60.0, max_radius=200.0):
    """
    net : RoadNetwork (ENU metres)
    obs : [K,2] observation points in the same ENU frame
    Returns dict with matched points [K,2], matched edge idx, and a 'matched'
    boolean mask (False where no road candidate was found).
    """
    K = len(obs)
    cand = []
    for e, n in obs:
        c = net.candidates(e, n, radius=radius, k=8)
        r = radius
        while not c and r < max_radius:
            r *= 2
            c = net.candidates(e, n, radius=r, k=8)
        cand.append(c)

    # Viterbi in log-space
    V = [None] * K          # best log-prob to each candidate
    back = [None] * K       # backpointer
    inv2s2 = 1.0 / (2 * sigma_z * sigma_z)

    def emis(c):            # c = (edge_idx, (pe,pn), dist)
        return -c[2] * c[2] * inv2s2

    # init
    first = next((t for t in range(K) if cand[t]), None)
    if first is None:
        return {"points": obs.copy(), "edges": [None] * K,
                "matched": np.zeros(K, bool)}
    V[first] = np.array([emis(c) for c in cand[first]])
    back[first] = [-1] * len(cand[first])

    prev_t = first
    for t in range(first + 1, K):
        if not cand[t]:
            continue
        step = np.hypot(obs[t][0] - obs[prev_t][0], obs[t][1] - obs[prev_t][1])
        Vt = np.full(len(cand[t]), NEG)
        bt = [-1] * len(cand[t])
        for j, cj in enumerate(cand[t]):
            best = NEG; bi = -1
            for i, ci in enumerate(cand[prev_t]):
                rd = net.route_distance(ci[0], ci[1], cj[0], cj[1])
                trans = -abs(step - rd) / beta
                val = V[prev_t][i] + trans
                if val > best:
                    best, bi = val, i
            Vt[j] = best + emis(cj)
            bt[j] = bi
        V[t] = Vt; back[t] = bt
        prev_t = t

    # backtrack
    points = obs.copy()
    edges = [None] * K
    matched = np.zeros(K, bool)
    j = int(np.argmax(V[prev_t]))
    for t in range(prev_t, first - 1, -1):
        if V[t] is None or not cand[t]:
            continue
        c = cand[t][j]
        points[t] = c[1]; edges[t] = c[0]; matched[t] = True
        j = back[t][j]
        if j < 0:
            break
    # fill unmatched ends with raw obs (already copied)
    return {"points": points, "edges": edges, "matched": matched}
