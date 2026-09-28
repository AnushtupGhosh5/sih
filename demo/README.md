# IDR Live Replay Demo

A navigation-app-style replay of a real IO-VNBD drive going through a 1 km GNSS
blackout, showing three tracks live on a map:

- **green** — ground truth (where the vehicle actually was)
- **red (dashed)** — plain smartphone inertial dead reckoning (drifts off the road)
- **cyan** — our IDR engine + map-matching (stays locked to the road)

The stats panel shows live drift for both methods; the banner flips to
"GNSS LOST · TUNNEL · DEAD RECKONING" during the blackout.

## Run it
Just open `index.html` in any browser (double-click). No install, no server.

For the smoothest experience (and live map tiles) be online. It also works
offline: Leaflet is vendored locally and the road network is drawn from cached
OSM data, so the tracks are always visible even without background tiles.

Optionally serve it (avoids any browser file:// quirks):
```
cd demo && python -m http.server 8000     # then open http://localhost:8000
```

## Regenerate the scenario
```
python -m src.idr.demo_export              # rewrites replay_data.js / .json
```
Edit `drive=` in `src/idr/demo_export.py` (e.g. "vfa02", "vtb5", "vw2") to pick a
different drive; the exporter auto-selects the most illustrative 1 km blackout.
