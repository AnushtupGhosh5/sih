# Y1 data audit

Generated from the raw Y1 files by `scripts/audit_y1.py`.

## Inventory

- Smartphone samples: 70,285
- Vehicle samples: 70,285
- Smartphone columns: 24
- Vehicle columns: 29

## Timing

- Smartphone `TIME SINCE START` span: 6690.761 s
- Smartphone elapsed-time resets: 3
- Smartphone wall-clock span: 7391.318 s
- Smartphone wall-clock large gaps: 3
- Vehicle clock span: 7666.800 s
- Vehicle clock large gaps: 2

## Row-alignment diagnostic

- Mean row-matched GPS separation: 657.1 m
- Median row-matched GPS separation: 651.6 m
- 95th percentile separation: 1242.4 m
- Maximum separation: 1533.0 m
- Smartphone GPS speed vs vehicle GNSS velocity correlation: 0.080
- Smartphone yaw-rate vs vehicle yaw-rate correlation: -0.002

Equal row counts therefore do **not** establish valid sample alignment. Timestamp
repair and segment-level synchronization are required before supervised training.

## Data-quality notes

- Smartphone GPS latitude and longitude contain no missing values.
- Smartphone GPS speed has 4 invalid values.
- Smartphone GPS orientation has 94 invalid values.
- Vehicle clutch position is constant across the recording.
- The vehicle GNSS velocity and indicated speed correlation is 0.876.
- Y1 is one driver/session, so it supports a proof of concept but not a driver-independent test.
