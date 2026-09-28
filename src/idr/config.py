"""Central configuration for the IDR training/eval pipeline."""
import os

# ---- Paths ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
V_DIR = os.path.join(RAW_DIR, "V-Dataset")
S_DIR = os.path.join(RAW_DIR, "S-Dataset")
ART_DIR = os.path.join(ROOT, "artifacts")          # models, scalers
FIG_DIR = os.path.join(ROOT, "artifacts", "figures")
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ENCODING = "latin-1"          # headers contain deg / superscript-2 / micro symbols
FS = 10.0                     # sampling rate (Hz)
KMH_TO_MS = 1.0 / 3.6

# ---- Windowing ----
WINDOW = 20                   # 2.0 s window at 10 Hz
STRIDE_TRAIN = 4              # dense sampling for training windows
NUM_FEATURES = 6              # mounting-invariant channels (see dataset.compute_features)

# ---- Vehicle / driver scope ----
# IO-VNBD was recorded by several drivers using DIFFERENT cars + phones. The
# vibration signature that encodes speed is vehicle-specific, so we train and
# evaluate WITHIN one vehicle (Driver E, the data-rich car: vf*/vt*/vw* = 64
# drives). This matches the deployment scenario (one user's phone in their car).
INCLUDE_PREFIXES = ("vf", "vt", "vw")

# ---- Drive-level split (held out entire drives to avoid leakage) ----
# Test drives chosen for the dead-reckoning / drift evaluation: a motorway run
# (vfa02) for the 1 km @ high-speed benchmark plus diverse routes.
TEST_DRIVES = ["vfa02", "vtb5", "vw2"]
VAL_DRIVES = ["vw4", "vta25", "vtb9"]
# Pure-stationary drives (no forward motion) - excluded from speed regression.
EXCLUDE_DRIVES = ["vw1", "vw15", "vtb3"]

# ---- Training ----
BATCH_SIZE = 512
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
