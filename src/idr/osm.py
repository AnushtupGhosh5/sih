"""
OpenStreetMap road network: fetch (Overpass), cache, and build a routable graph
in a local ENU (metres) frame for map-matching.

The vehicle drives on roads, so the road network is used as a geometric prior to
correct inertial dead-reckoning drift. Everything is projected to a planar ENU
frame (equirectangular around a drive-level reference) so shapely/networkx work
in metres.
"""
import os
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from . import config as C

OSM_DIR = os.path.join(C.ROOT, "data", "osm")
os.makedirs(OSM_DIR, exist_ok=True)

R_EARTH = 6371000.0
DRIVABLE = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
            "living_street|service|road|motorway_link|trunk_link|primary_link|"
            "secondary_link|tertiary_link")
OVERPASS = "https://overpass-api.de/api/interpreter"


def latlon_to_enu(lat, lon, lat0, lon0):
    lat0r = np.radians(lat0)
    e = np.radians(np.asarray(lon) - lon0) * R_EARTH * np.cos(lat0r)
    n = np.radians(np.asarray(lat) - lat0) * R_EARTH
    return e, n


def fetch_osm(bbox, cache_key):
    """bbox = (south, west, north, east). Returns Overpass JSON (cached)."""
    path = os.path.join(OSM_DIR, f"{cache_key}.json")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        with open(path) as f:
            return json.load(f)
    s, w, n, e = bbox
    query = (f"[out:json][timeout:90];way({s},{w},{n},{e})"
             f'["highway"~"{DRIVABLE}"];out geom;')
    data = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(OVERPASS, data=data,
                                         headers={"User-Agent": "sih-idr"})
            with urllib.request.urlopen(req, timeout=120) as r:
                js = json.loads(r.read().decode())
            with open(path, "w") as f:
                json.dump(js, f)
            return js
        except Exception as ex:  # noqa
            print(f"  Overpass retry {attempt+1}: {ex}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("Overpass fetch failed")


class RoadNetwork:
    """Routable road graph in ENU metres + spatial index over road segments."""

    def __init__(self, osm_json, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.G = nx.Graph()
        node_xy = {}
        for el in osm_json["elements"]:
            if el["type"] != "way" or "geometry" not in el:
                continue
            ids = el.get("nodes")
            geom = el["geometry"]
            if not ids or len(ids) != len(geom):
                wid = el.get("id", id(el))
                ids = [f"w{wid}_{i}" for i in range(len(geom))]  # fallback (rare)
            for nid, pt in zip(ids, geom):
                if nid not in node_xy:
                    e, n = latlon_to_enu(pt["lat"], pt["lon"], lat0, lon0)
                    node_xy[nid] = (float(e), float(n))
            for a, b in zip(ids[:-1], ids[1:]):
                xa, xb = node_xy[a], node_xy[b]
                d = float(np.hypot(xa[0] - xb[0], xa[1] - xb[1]))
                if d > 0:
                    self.G.add_edge(a, b, length=d)

        for nid, xy in node_xy.items():
            if nid in self.G:
                self.G.nodes[nid]["xy"] = xy
        self.node_xy = node_xy

        # spatial index over edge segments
        self.edges = list(self.G.edges())
        self.edge_lines = [LineString([node_xy[a], node_xy[b]]) for a, b in self.edges]
        self.tree = STRtree(self.edge_lines)

    def candidates(self, e, n, radius=40.0, k=8):
        """Return [(edge_idx, (pe,pn), dist)] for road segments near (e,n)."""
        p = Point(e, n)
        idxs = self.tree.query(p.buffer(radius))
        out = []
        for i in np.atleast_1d(idxs):
            line = self.edge_lines[int(i)]
            d = line.distance(p)
            if d <= radius:
                proj = line.interpolate(line.project(p))
                out.append((int(i), (proj.x, proj.y), float(d)))
        out.sort(key=lambda t: t[2])
        return out[:k]

    def _heur(self, u, v):
        xu, xv = self.node_xy[u], self.node_xy[v]
        return np.hypot(xu[0] - xv[0], xu[1] - xv[1])

    def route_distance(self, ei, pi, ej, pj, cap=2000.0):
        """Approx on-road distance between point pi on edge ei and pj on edge ej.
        Uses A* with a Euclidean heuristic so exploration stays local/fast."""
        if ei == ej:
            return float(np.hypot(pi[0] - pj[0], pi[1] - pj[1]))
        a1, b1 = self.edges[ei]
        a2, b2 = self.edges[ej]
        best = cap
        for na in (a1, b1):
            da = np.hypot(pi[0] - self.node_xy[na][0], pi[1] - self.node_xy[na][1])
            for nb in (a2, b2):
                db = np.hypot(pj[0] - self.node_xy[nb][0], pj[1] - self.node_xy[nb][1])
                try:
                    dnode = nx.astar_path_length(self.G, na, nb,
                                                 heuristic=self._heur, weight="length")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                best = min(best, da + dnode + db)
        return float(best)


def network_for_drive(df, cache_key, pad_m=300.0):
    """Build a RoadNetwork covering a drive's extent (ENU origin = drive start)."""
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    lat0, lon0 = lat[0], lon[0]
    dlat = pad_m / R_EARTH * 180 / np.pi
    dlon = dlat / max(np.cos(np.radians(lat0)), 1e-6)
    bbox = (lat.min() - dlat, lon.min() - dlon, lat.max() + dlat, lon.max() + dlon)
    js = fetch_osm(bbox, cache_key)
    return RoadNetwork(js, lat0, lon0)
