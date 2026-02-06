import os, zipfile, textwrap, shutil, base64, io, math, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

base_zip = "/mnt/data/vxp_vxp_sim_bo105_optionB_run2_legacy.zip"
workdir = Path("/mnt/data/vxp_final_build")
if workdir.exists():
    shutil.rmtree(workdir)
workdir.mkdir(parents=True, exist_ok=True)

# Extract
with zipfile.ZipFile(base_zip, "r") as z:
    z.extractall(workdir)

proj = workdir / "vxp_vxp_sim"
assets = proj / "assets" / "icons"
assets.mkdir(parents=True, exist_ok=True)

# Create small XP-like monochrome icons
def make_icon(name, draw_fn, size=22):
    im = Image.new("RGBA", (size, size), (0,0,0,0))
    d = ImageDraw.Draw(im)
    draw_fn(d, size)
    im.save(assets / f"{name}.png")

def icon_disconnect(d, s):
    # X
    d.line((5,5,s-5,s-5), fill=(0,0,0,255), width=2)
    d.line((s-5,5,5,s-5), fill=(0,0,0,255), width=2)

def icon_upload(d, s):
    # up arrow
    d.polygon([(s/2,4),(s-6,12),(s-12,12),(s-12,s-4),(12,s-4),(12,12),(6,12)], fill=(0,0,0,255))

def icon_download(d, s):
    # down arrow
    d.polygon([(6,s-12),(12,s-12),(12,6),(s-12,6),(s-12,s-12),(s-6,s-12),(s/2,s-4)], fill=(0,0,0,255))

def icon_viewlog(d, s):
    # page
    d.rectangle((6,4,s-6,s-4), outline=(0,0,0,255), width=2)
    d.line((8,8,s-8,8), fill=(0,0,0,255), width=1)
    d.line((8,11,s-8,11), fill=(0,0,0,255), width=1)
    d.line((8,14,s-8,14), fill=(0,0,0,255), width=1)

def icon_print(d, s):
    # printer
    d.rectangle((6,5,s-6,11), outline=(0,0,0,255), width=2)
    d.rectangle((7,11,s-7,s-7), outline=(0,0,0,255), width=2)
    d.rectangle((8,3,s-8,6), outline=(0,0,0,255), width=2)

def icon_help(d, s):
    # question mark
    d.arc((6,4,s-6,s-6), start=200, end=20, fill=(0,0,0,255), width=2)
    d.line((s/2, s/2, s/2, s-8), fill=(0,0,0,255), width=2)
    d.ellipse((s/2-1, s-6, s/2+1, s-4), fill=(0,0,0,255))

def icon_exit(d, s):
    # curved arrow left
    d.arc((5,5,s-5,s-5), start=90, end=220, fill=(0,0,0,255), width=2)
    d.polygon([(6, s/2), (11, s/2-4), (11, s/2+4)], fill=(0,0,0,255))

make_icon("disconnect", icon_disconnect)
make_icon("upload", icon_upload)
make_icon("download", icon_download)
make_icon("viewlog", icon_viewlog)
make_icon("print", icon_print)
make_icon("help", icon_help)
make_icon("exit", icon_exit)

# Helper to b64 encode images for inline HTML
def b64_png(path):
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")

icons_b64 = {p.stem: b64_png(p) for p in assets.glob("*.png")}

# Build new app.py implementing requested changes
app_py = proj / "app.py"
new_code = r"""
from __future__ import annotations

import base64
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ==========================================================
# Chadwick-Helmuth VXP (training simulator) – BO105 Main Rotor
# Tracking & Balance – Option B
#
# Requested final behaviors:
# - Centered (not wide). More square 4:3 feel.
# - XP/VXP-like bold fonts + button styling.
# - Left toolbar with icon images (VXP-style).
# - Only "Main Rotor Track & Balance Run 1" on main menu.
# - RUN 2 / RUN 3 created via NEXT RUN inside main-rotor menu.
# - Guided 3-run scenario:
#     RUN 1: tracking out -> corrected with Pitch Link
#     RUN 2: fwd-flight tracking out -> corrected with Trim Tabs
#     RUN 3: 1/rev vib out -> corrected with Weight -> after 3rd measurement OK
# - Edit Solution screen updates the applied changes.
# - Graphical results: show track for ONE selected measurement.
# - Full result includes MORE (computed) measurement lines.
# - View results for different RUNs (selector inside Main Rotor screens).
#
# Training/simulation only. Not for real aircraft work.
# ==========================================================

# ---- Rotor config (BO105: 4 blades) ----
BLADES: List[str] = ["BLU", "GRN", "YEL", "RED"]
BLADE_FULL = {"BLU": "Blue", "GRN": "Green", "YEL": "Yellow (Ref)", "RED": "Red"}

# 0° = 12 o'clock, clockwise
BLADE_CLOCK_DEG = {"YEL": 0.0, "RED": 90.0, "BLU": 180.0, "GRN": 270.0}

# Collect menu requested (3 tests)
REGIMES: List[str] = ["GROUND", "HOVER", "HORIZONTAL"]
REGIME_LABEL = {"GROUND": "100% Ground", "HOVER": "Hover Flight", "HORIZONTAL": "Horizontal Flight"}

# Extra "display-only" lines to make the output look like VXP
DISPLAY_POINTS: List[Tuple[str, str]] = [
    ("100% Ground", "GROUND"),
    ("Hover Flight", "HOVER"),
    ("Hover IGE (est)", "HOVER"),
    ("40 KIAS (est)", "HORIZONTAL"),
    ("80 KIAS (est)", "HORIZONTAL"),
    ("Horizontal Flight", "HORIZONTAL"),
]

# Option B cue (visual)
TRACKING_OPTION = "B"
STROBEX_MODE_SWITCH = "B"

# BO105 RPM cue (displayed)
BO105_DISPLAY_RPM = 424.0

# ----------------------------------------------------------
# Data structures
# ----------------------------------------------------------
@dataclass
class BalanceReading:
    amp_ips: float
    phase_deg: float
    rpm: float

@dataclass
class Measurement:
    regime: str
    balance: BalanceReading
    track_mm: Dict[str, float]  # per blade, relative to YEL

# ----------------------------------------------------------
# XP / VXP look
# ----------------------------------------------------------
def _load_icon_b64(name: str) -> str:
    return st.session_state.vxp_icons_b64.get(name, "")

XP_CSS = r"""
<style>
/* Hide Streamlit chrome */
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu {
  display: none !important;

# Ensure icons present in session_state
if 'vxp_icons_b64' not in st.session_state:
    st.session_state.vxp_icons_b64 = _VXP_ICONS_B64.copy()

/* Page background */
html, body, [data-testid="stAppViewContainer"] {
  background: #bfbfbf;
  font-family: Tahoma, "MS Sans Serif", Verdana, Arial, sans-serif;
  font-size: 12px;
}

/* Make centered / squarer */
.block-container {
  padding-top: 0.25rem;
  padding-bottom: 0.75rem;
  max-width: 1080px;
  margin: 0 auto;
}

/* App frame */
.vxp-frame {
  background: #c0c0c0;
  border: 2px solid #404040;
  box-shadow: 2px 2px 0px #808080;
  border-radius: 2px;
  max-width: 1024px;
  margin: 0 auto;
}

.vxp-titlebar {
  background: linear-gradient(90deg, #0a246a 0%, #3a6ea5 100%);
  color: white;
  padding: 6px 10px;
  height: 30px;
  font-weight: 900;
  letter-spacing: 0.2px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.vxp-menubar {
  background: #d4d0c8;
  border-bottom: 1px solid #808080;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
}

.vxp-content {
  padding: 10px;
  background: #c0c0c0;
}

/* VXP-style buttons */
.stButton > button {
  background: #d4d0c8 !important;
  color: #000 !important;
  border-top: 2px solid #ffffff !important;
  border-left: 2px solid #ffffff !important;
  border-right: 2px solid #404040 !important;
  border-bottom: 2px solid #404040 !important;
  border-radius: 0px !important;
  font-weight: 900 !important;
  font-size: 13px !important;
  padding: 10px 12px !important;
  letter-spacing: 0.2px;
}

.stButton > button:active {
  border-top: 2px solid #404040 !important;
  border-left: 2px solid #404040 !important;
  border-right: 2px solid #ffffff !important;
  border-bottom: 2px solid #ffffff !important;
}

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
  border-radius: 0px !important;
  border: 2px solid #404040 !important;
  background: #ffffff !important;
  font-weight: 700 !important;
}

.vxp-label {
  font-weight: 900;
  font-size: 13px;
}

.vxp-mono {
  font-family: "Courier New", Courier, monospace;
  font-size: 12px;
  white-space: pre;
  background: #efefef;
  border: 2px solid #808080;
  padding: 10px;
}

/* Left toolbar (HTML buttons) */
.vxp-toolbar {
  background: #d4d0c8;
  border: 2px solid #808080;
  box-shadow: inset 1px 1px 0px #ffffff;
  padding: 8px;
  border-radius: 2px;
}

.vxp-sidebtn {
  display: block;
  background: #d4d0c8;
  border-top: 2px solid #ffffff;
  border-left: 2px solid #ffffff;
  border-right: 2px solid #404040;
  border-bottom: 2px solid #404040;
  text-decoration: none;
  color: #000;
  font-weight: 900;
  font-size: 13px;
  padding: 12px 10px;
  margin-bottom: 10px;
}

.vxp-sidebtn:active {
  border-top: 2px solid #404040;
  border-left: 2px solid #404040;
  border-right: 2px solid #ffffff;
  border-bottom: 2px solid #ffffff;
}

.vxp-sidebtn img {
  width: 18px;
  height: 18px;
  vertical-align: middle;
  margin-right: 10px;
}

/* Header strip inside screens */
.vxp-strip {
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin: 4px 0 10px 0;
  font-weight: 900;
}
</style>
"""

# ----------------------------------------------------------
# Math helpers
# ----------------------------------------------------------
def _vec_from_clock_deg(theta_deg: float) -> np.ndarray:
    phi = math.radians(90.0 - theta_deg)
    return np.array([math.cos(phi), math.sin(phi)], dtype=float)

def _clock_deg_from_vec(v: np.ndarray) -> float:
    x, y = float(v[0]), float(v[1])
    phi = math.degrees(math.atan2(y, x))
    return (90.0 - phi) % 360.0

def _clock_label(theta_deg: float) -> str:
    hour = int(round(theta_deg / 30.0)) % 12
    hour = 12 if hour == 0 else hour
    minute = 0 if abs((theta_deg / 30.0) - round(theta_deg / 30.0)) < 0.25 else 30
    return f"{hour:02d}:{minute:02d}"

def _round_quarter(x: float) -> float:
    return round(x * 4.0) / 4.0

# ----------------------------------------------------------
# Simulation model (guided 3-run script)
# ----------------------------------------------------------
PITCHLINK_MM_PER_TURN = 10.0
TRIMTAB_MMTRACK_PER_MM = 15.0
BOLT_IPS_PER_GRAM = 0.0020

# Guided base states per run (training values)
RUN_BASE_TRACK = {
    1: {  # tracking problem
        "GROUND": {"BLU": +18.0, "GRN": -8.0, "YEL": 0.0, "RED": -12.0},
        "HOVER": {"BLU": +14.0, "GRN": -6.0, "YEL": 0.0, "RED": -10.0},
        "HORIZONTAL": {"BLU": +10.0, "GRN": -4.0, "YEL": 0.0, "RED": -8.0},
    },
    2: {  # fwd-flight track still off (trim tabs)
        "GROUND": {"BLU": +4.0, "GRN": -3.0, "YEL": 0.0, "RED": -2.0},
        "HOVER": {"BLU": +3.0, "GRN": -2.0, "YEL": 0.0, "RED": -2.0},
        "HORIZONTAL": {"BLU": +14.0, "GRN": -6.0, "YEL": 0.0, "RED": -9.0},
    },
    3: {  # track OK, balance/vib to be corrected by weight before last measurement
        "GROUND": {"BLU": +2.0, "GRN": -2.0, "YEL": 0.0, "RED": -1.0},
        "HOVER": {"BLU": +2.0, "GRN": -1.5, "YEL": 0.0, "RED": -1.0},
        "HORIZONTAL": {"BLU": +2.0, "GRN": -2.0, "YEL": 0.0, "RED": -1.0},
    },
}

RUN_BASE_BAL = {
    1: {"GROUND": (0.30, 125.0), "HOVER": (0.12, 110.0), "HORIZONTAL": (0.09, 95.0)},
    2: {"GROUND": (0.22, 140.0), "HOVER": (0.09, 120.0), "HORIZONTAL": (0.07, 105.0)},
    3: {"GROUND": (0.18, 160.0), "HOVER": (0.08, 135.0), "HORIZONTAL": (0.06, 120.0)},
}

def _default_adjustments() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        r: {
            "pitch_turns": {b: 0.0 for b in BLADES},
            "trim_mm": {b: 0.0 for b in BLADES},
            "bolt_g": {b: 0.0 for b in BLADES},
        }
        for r in REGIMES
    }

def _simulate_measurement(run: int, regime: str) -> Measurement:
    adj = st.session_state.vxp_adjustments[regime]

    # Base guided script
    base_track = RUN_BASE_TRACK.get(run, RUN_BASE_TRACK[3])[regime].copy()
    base_amp, base_phase = RUN_BASE_BAL.get(run, RUN_BASE_BAL[3])[regime]

    # Apply pitch link (affects all regimes)
    track: Dict[str, float] = {}
    for b in BLADES:
        pitch_effect = PITCHLINK_MM_PER_TURN * float(adj["pitch_turns"][b])
        trim_effect = 0.0
        if regime == "HORIZONTAL":
            trim_effect = TRIMTAB_MMTRACK_PER_MM * float(adj["trim_mm"][b])
        noise = random.gauss(0.0, 0.45)
        track[b] = float(base_track[b] + pitch_effect + trim_effect + noise)

    # Relative to YEL
    yel0 = float(track["YEL"])
    for b in BLADES:
        track[b] = float(track[b] - yel0)
    track["YEL"] = 0.0

    # Balance vector
    v = _vec_from_clock_deg(base_phase) * float(base_amp)
    for b in BLADES:
        grams = float(adj["bolt_g"][b])
        v += (-BOLT_IPS_PER_GRAM * grams) * _vec_from_clock_deg(BLADE_CLOCK_DEG[b])
    v += np.array([random.gauss(0.0, 0.003), random.gauss(0.0, 0.003)], dtype=float)

    amp = float(np.linalg.norm(v))
    phase = float(_clock_deg_from_vec(v)) if amp > 1e-6 else 0.0

    return Measurement(regime=regime, balance=BalanceReading(amp, phase, BO105_DISPLAY_RPM), track_mm=track)

# ----------------------------------------------------------
# Guided “Edit Solution” mapping (VXP names)
# ----------------------------------------------------------
# VXP names (kept for UI realism)
EDIT_ITEMS = [
    ("Pitch Link (flats)", "pitch"),
    ("Tab Sta 5 (deg)", "tab5"),
    ("Tab Sta 6 (deg)", "tab6"),
    ("Weight (plqts)", "weight"),
]

def _flats_to_turns(flats: float) -> float:
    # training mapping: 6 flats = 1 turn
    return flats / 6.0

def _deg_to_trim_mm(deg: float) -> float:
    # training mapping: 1 deg ~ 1 mm equivalent
    return deg

def _plqts_to_grams(plqts: float) -> float:
    # training mapping: 1 plqt ~ 10 g
    return plqts * 10.0

# ----------------------------------------------------------
# Solution logic / limits (training)
# ----------------------------------------------------------
def _track_limit(regime: str) -> float:
    # training tolerances aligned with typical excerpts
    if regime == "HOVER":
        return 5.0
    if regime == "HORIZONTAL":
        return 5.0
    return 10.0

def _balance_limit(regime: str) -> float:
    return 0.40 if regime == "GROUND" else 0.05

def _track_spread(m: Measurement) -> float:
    vals = [m.track_mm[b] for b in BLADES]
    return float(max(vals) - min(vals))

def _suggest_pitchlink(meas: Dict[str, Measurement]) -> Dict[str, float]:
    # Use ground + hover average per blade
    used = [r for r in ("GROUND", "HOVER") if r in meas]
    if not used:
        return {b: 0.0 for b in BLADES}
    out = {}
    for b in BLADES:
        avg = sum(meas[r].track_mm[b] for r in used) / len(used)
        out[b] = _round_quarter((-avg) / PITCHLINK_MM_PER_TURN)
    return out

def _suggest_trimtabs(meas: Dict[str, Measurement]) -> Dict[str, float]:
    if "HORIZONTAL" not in meas:
        return {b: 0.0 for b in BLADES}
    out = {}
    for b in BLADES:
        dev = meas["HORIZONTAL"].track_mm[b]
        out[b] = max(-5.0, min(5.0, _round_quarter((-dev) / TRIMTAB_MMTRACK_PER_MM)))
    return out

def _suggest_weight(meas: Dict[str, Measurement]) -> Tuple[str, float]:
    # choose the worst amplitude regime among measured
    if not meas:
        return ("YEL", 0.0)
    worst_r = max(meas.keys(), key=lambda r: meas[r].balance.amp_ips)
    m = meas[worst_r]
    amp = m.balance.amp_ips
    phase = m.balance.phase_deg
    target = (phase + 180.0) % 360.0

    def dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    blade = min(BLADES, key=lambda bb: dist(target, BLADE_CLOCK_DEG[bb]))
    grams = max(5.0, min(120.0, round(amp / BOLT_IPS_PER_GRAM / 5.0) * 5.0))
    return (blade, grams)

def _all_ok_for_run(run: int, meas: Dict[str, Measurement]) -> bool:
    # OK when all collected regimes within both limits
    for r in REGIMES:
        if r not in meas:
            return False
        if _track_spread(meas[r]) > _track_limit(r):
            return False
        if meas[r].balance.amp_ips > _balance_limit(r):
            return False
    # Additionally: guided end of scenario after Run 3
    return True

# ----------------------------------------------------------
# UI helpers
# ----------------------------------------------------------
def _go(screen: str, **kwargs) -> None:
    st.session_state.vxp_screen = screen
    for k, v in kwargs.items():
        st.session_state[k] = v

def _frame_start(title: str) -> None:
    st.markdown(XP_CSS, unsafe_allow_html=True)
    st.markdown("<div class='vxp-frame'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='vxp-titlebar'><div>{title}</div><div style='font-weight:900;'>✕</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='vxp-menubar'>File&nbsp;&nbsp;View&nbsp;&nbsp;Log&nbsp;&nbsp;Test AU&nbsp;&nbsp;Settings&nbsp;&nbsp;Help</div>", unsafe_allow_html=True)
    st.markdown("<div class='vxp-content'>", unsafe_allow_html=True)

def _frame_end() -> None:
    st.markdown("</div></div>", unsafe_allow_html=True)

def _toolbar() -> None:
    # HTML toolbar with inline b64 icons. Clicking uses query param "nav".
    def sidebtn(icon: str, label: str, nav: str) -> str:
        b64 = _load_icon_b64(icon)
        img = f"<img src='data:image/png;base64,{b64}'/>" if b64 else ""
        return f\"\"\"<a class='vxp-sidebtn' href='?nav={nav}'>{img}{label}</a>\"\"\"

    st.markdown("<div class='vxp-toolbar'>", unsafe_allow_html=True)
    st.markdown(sidebtn("disconnect", "Disconnect", "disconnect"), unsafe_allow_html=True)
    st.markdown(sidebtn("upload", "Upload", "upload"), unsafe_allow_html=True)
    st.markdown(sidebtn("download", "Download", "download"), unsafe_allow_html=True)
    st.markdown(sidebtn("viewlog", "View Log", "viewlog"), unsafe_allow_html=True)
    st.markdown(sidebtn("print", "Print AU", "print"), unsafe_allow_html=True)
    st.markdown(sidebtn("help", "Help", "help"), unsafe_allow_html=True)
    st.markdown(sidebtn("exit", "Exit", "exit"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def _apply_nav() -> None:
    qp = st.experimental_get_query_params()
    nav = (qp.get("nav", [""])[0] or "").lower()
    if not nav:
        return
    # clear params immediately
    st.experimental_set_query_params()
    if nav in ("disconnect", "exit"):
        _go("home")
    # others are UI-only
    st.rerun()

def _run_selector_inline() -> int:
    runs = sorted(st.session_state.vxp_runs.keys())
    current = int(st.session_state.vxp_view_run)
    if current not in runs:
        current = runs[0]
        st.session_state.vxp_view_run = current
    idx = runs.index(current)
    run = st.selectbox("Run", runs, index=idx, key="run_selector")
    st.session_state.vxp_view_run = int(run)
    return int(run)

# ----------------------------------------------------------
# Plotting (VXP-like)
# ----------------------------------------------------------
def _plot_track_marker(meas: Measurement) -> plt.Figure:
    fig = plt.figure(figsize=(6.3, 2.4), dpi=120)
    fig.patch.set_facecolor("#c0c0c0")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    # Vertical scale like VXP (±32.5)
    ax.set_ylim(-32.5, 32.5)
    ax.set_xlim(0.5, len(BLADES) + 0.5)

    ax.set_yticks([-32.5, 0.0, 32.5])
    ax.set_ylabel("mm", fontsize=9)
    ax.set_xticks(range(1, len(BLADES) + 1))
    ax.set_xticklabels(BLADES, fontsize=9, fontweight="bold")

    for i in range(1, len(BLADES) + 1):
        ax.axvline(i, color="black", linewidth=0.6, linestyle=":")

    # Plot squares at each blade track height
    xs = list(range(1, len(BLADES) + 1))
    ys = [meas.track_mm[b] for b in BLADES]
    ax.scatter(xs, ys, marker="s", s=28)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Track Height (mm) — {REGIME_LABEL[meas.regime]}", fontsize=10, fontweight="bold")

    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(1.0)

    fig.tight_layout(pad=0.8)
    return fig

def _plot_track_graph(meas_by_regime: Dict[str, Measurement]) -> plt.Figure:
    xs = [REGIME_LABEL[r] for r in REGIMES if r in meas_by_regime]
    fig = plt.figure(figsize=(6.3, 2.6), dpi=120)
    fig.patch.set_facecolor("#c0c0c0")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    for b in BLADES:
        ys = [meas_by_regime[r].track_mm[b] for r in REGIMES if r in meas_by_regime]
        ax.plot(xs, ys, marker="s", linewidth=1.2, markersize=4, label=b)
    ax.set_ylim(-32.5, 32.5)
    ax.set_ylabel("mm", fontsize=9)
    ax.set_title("Track Height (relative to YEL)", fontsize=10, fontweight="bold")
    ax.axhline(0.0, linewidth=0.8)
    ax.grid(True, linestyle=":", linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(1.0)
    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(pad=0.9)
    return fig

def _plot_polar(meas: Measurement) -> plt.Figure:
    fig = plt.figure(figsize=(5.4, 5.4), dpi=120)
    fig.patch.set_facecolor("#c0c0c0")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ticks = [math.radians(t) for t in range(0, 360, 30)]
    labels = ["12","1","2","3","4","5","6","7","8","9","10","11"]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_rmax(max(0.35, meas.balance.amp_ips*1.4))
    ax.grid(True, linestyle=":", linewidth=0.6)
    theta = math.radians(meas.balance.phase_deg)
    ax.plot([theta],[meas.balance.amp_ips], marker="o", markersize=7)
    ax.text(theta, meas.balance.amp_ips+0.01, f"{meas.balance.amp_ips:.2f}", fontsize=9, ha="center")
    ax.set_title("1/rev Balance (IPS vs Phase)", fontsize=10, fontweight="bold", pad=12)
    fig.tight_layout(pad=0.8)
    return fig

# ----------------------------------------------------------
# Text output blocks (VXP-like)
# ----------------------------------------------------------
def _legacy_results_text(run: int, meas_by_regime: Dict[str, Measurement]) -> str:
    lines: List[str] = []
    lines.append("BO105   MAIN ROTOR  TRACK & BALANCE")
    lines.append(f"OPTION: {TRACKING_OPTION}   STROBEX MODE: {STROBEX_MODE_SWITCH}")
    lines.append(f"RUN: {run}   ID: TRAINING")
    lines.append("")
    lines.append("----- Balance Measurements -----")
    for name, src in DISPLAY_POINTS:
        if src not in meas_by_regime:
            continue
        m = meas_by_regime[src]
        # use small deltas for "est" lines
        amp = m.balance.amp_ips * (1.05 if "(est)" in name else 1.0)
        ph = (m.balance.phase_deg + (5 if "(est)" in name else 0)) % 360
        lines.append(f"{name:<18}  1P {amp:0.2f} IPS  {_clock_label(ph):>5}  RPM:{m.balance.rpm:0.0f}")

    lines.append("")
    lines.append("----- Track Height (mm rel. YEL) -----")
    for name, src in DISPLAY_POINTS:
        if src not in meas_by_regime:
            continue
        m = meas_by_regime[src]
        # for "est" just nudge slightly
        def nud(x): return x + (0.6 if "(est)" in name else 0.0)
        row = "  ".join([f"{b}:{nud(m.track_mm[b]):+5.1f}" for b in BLADES])
        lines.append(f"{name:<18}  {row}")
    lines.append("")
    return "\n".join(lines)

def _applied_changes_text(run: int) -> str:
    applied = st.session_state.vxp_applied_changes.get(run, [])
    if not applied:
        return "NONE"
    return "\n".join([f"  - {x}" for x in applied])

# ----------------------------------------------------------
# Screens
# ----------------------------------------------------------
def screen_home() -> None:
    _frame_start("Chadwick-Helmuth VXP  —  BO105 (Training)")
    st.markdown("<div class='vxp-label'>Select Procedure:</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    st.button("Aircraft Info", use_container_width=True, on_click=_go, args=("aircraft_info",))
    st.button("Main Rotor Track & Balance Run 1", use_container_width=True, on_click=_go, args=("mr_menu",))

    st.button("Vibration Signatures", use_container_width=True, on_click=_go, args=("not_impl",))
    st.button("Measurements Only", use_container_width=True, on_click=_go, args=("not_impl",))
    st.button("Setup / Utilities", use_container_width=True, on_click=_go, args=("not_impl",))

    _frame_end()

def screen_not_impl() -> None:
    _frame_start("VXP  —  Not Implemented")
    st.write("Solo se implementa **Main Rotor – Tracking & Balance (Option B)** para el BO105.")
    st.button("Close", on_click=_go, args=("home",))
    _frame_end()

def screen_aircraft_info() -> None:
    _frame_start("AIRCRAFT INFO")
    info = st.session_state.vxp_aircraft

    c1, c2 = st.columns([0.35, 0.65], gap="large")
    with c1:
        st.markdown("<div class='vxp-label'>WEIGHT:</div>", unsafe_allow_html=True)
        st.markdown("<div class='vxp-label'>C.G.:</div>", unsafe_allow_html=True)
        st.markdown("<div class='vxp-label'>HOURS:</div>", unsafe_allow_html=True)
        st.markdown("<div class='vxp-label'>INITIALS:</div>", unsafe_allow_html=True)
    with c2:
        info["weight"] = st.number_input("", value=float(info.get("weight", 0.0)), key="weight_in")
        info["cg"] = st.number_input("", value=float(info.get("cg", 0.0)), key="cg_in")
        info["hours"] = st.number_input("", value=float(info.get("hours", 0.0)), key="hrs_in")
        info["initials"] = st.text_input("", value=str(info.get("initials", "")), key="init_in")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.button("Note Codes", use_container_width=True, on_click=_go, args=("note_codes",))
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.button("Close", on_click=_go, args=("home",))
    _frame_end()

def screen_note_codes() -> None:
    _frame_start("NOTE CODES")
    codes = [
        (0, "Scheduled Insp"),
        (1, "Balance"),
        (2, "Troubleshooting"),
        (3, "Low Freq Vib"),
        (4, "Med Freq Vib"),
        (5, "High Freq Vib"),
        (6, "Component Change"),
    ]
    selected = st.session_state.vxp_note_codes
    for code, name in codes:
        label = f"{code:02d} {name}"
        checked = "✓" if code in selected else ""
        cols = st.columns([0.85, 0.15])
        with cols[0]:
            if st.button(label, use_container_width=True, key=f"nc_{code}"):
                if code in selected: selected.remove(code)
                else: selected.add(code)
                st.rerun()
        with cols[1]:
            st.markdown(f"<div style='font-size:20px; font-weight:900; padding-top:10px;'>{checked}</div>", unsafe_allow_html=True)
    st.button("Close", on_click=_go, args=("aircraft_info",))
    _frame_end()

def screen_mr_menu() -> None:
    _frame_start(f"Main Rotor Balance Run {st.session_state.vxp_run}")
    st.markdown(f"<div class='vxp-strip'><div>Tracking &amp; Balance – Option {TRACKING_OPTION}</div><div>Run {st.session_state.vxp_run}</div></div>", unsafe_allow_html=True)

    def btn(label: str, screen: str):
        st.button(label, use_container_width=True, on_click=_go, args=(screen,))

    btn("COLLECT", "collect")
    btn("MEASUREMENTS LIST", "meas_list")
    btn("MEASUREMENTS GRAPH", "meas_graph")
    btn("SETTINGS", "settings")
    btn("SOLUTION", "solution")
    btn("NEXT RUN", "next_run_prompt")

    st.button("Close", on_click=_go, args=("home",))
    _frame_end()

def _current_run_data(run: int) -> Dict[str, Measurement]:
    return st.session_state.vxp_runs.setdefault(run, {})

def _completed_set(run: int) -> set:
    return st.session_state.vxp_completed_by_run.setdefault(run, set())

def screen_collect() -> None:
    _frame_start(f"Main Rotor: Run {st.session_state.vxp_run} — Day Mode")
    run = int(st.session_state.vxp_run)
    st.markdown(f"<div class='vxp-strip'><div>RPM {BO105_DISPLAY_RPM:.1f}</div><div>Run {run}</div></div>", unsafe_allow_html=True)

    completed = _completed_set(run)
    for r in REGIMES:
        cols = st.columns([0.86, 0.14])
        with cols[0]:
            if st.button(REGIME_LABEL[r], use_container_width=True, key=f"reg_{run}_{r}"):
                st.session_state.vxp_pending_regime = r
                _go("acquire")
                st.rerun()
        with cols[1]:
            mark = "✓" if r in completed else ""
            st.markdown(f"<div style='font-size:22px; font-weight:900; padding-top:10px;'>{mark}</div>", unsafe_allow_html=True)

    if run == 3 and len(completed) == 3:
        # Guided endpoint cue
        st.markdown("<div class='vxp-label' style='margin-top:10px;'>✓ RUN 3 COMPLETE — PARAMETERS OK</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()

def screen_acquire() -> None:
    _frame_start("ACQUIRING …")
    run = int(st.session_state.vxp_run)
    r = st.session_state.get("vxp_pending_regime")
    if not r:
        st.button("Close", on_click=_go, args=("collect",))
        _frame_end()
        return

    st.markdown(f"<div class='vxp-label'>{REGIME_LABEL[r]}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='vxp-label'>RPM {BO105_DISPLAY_RPM:.1f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='vxp-label'>Set Strobex: Mode {STROBEX_MODE_SWITCH}</div>", unsafe_allow_html=True)
    st.markdown("<div class='vxp-mono'>M/R LAT\t\tACQUIRING\n\nM/R OBT\t\tACQUIRING</div>", unsafe_allow_html=True)

    if not st.session_state.get("vxp_acq_in_progress", False):
        st.session_state.vxp_acq_in_progress = True
        p = st.progress(0)
        for i in range(80):
            time.sleep(0.01)
            p.progress(i + 1)

        meas = _simulate_measurement(run, r)
        _current_run_data(run)[r] = meas
        _completed_set(run).add(r)

        st.session_state.vxp_pending_regime = None
        st.session_state.vxp_acq_in_progress = False

        # Guided logic: in Run 3, if operator already applied weights before last measurement,
        # the third measurement (typically HORIZONTAL) will show "OK" naturally.
        _go("collect")
        st.rerun()

    st.button("Close", on_click=_go, args=("collect",))
    _frame_end()

def screen_meas_list() -> None:
    _frame_start("MEASUREMENTS LIST")
    view_run = _run_selector_inline()
    data = _current_run_data(view_run)

    if not data:
        st.write("No measurements for this run yet. Go to COLLECT.")
        st.button("Close", on_click=_go, args=("mr_menu",))
        _frame_end()
        return

    st.markdown(f"<div class='vxp-mono' style='height:520px; overflow:auto;'>{_legacy_results_text(view_run, data)}</div>", unsafe_allow_html=True)
    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()

def screen_meas_graph() -> None:
    _frame_start("MEASUREMENTS GRAPH")
    view_run = _run_selector_inline()
    data = _current_run_data(view_run)

    if not data:
        st.write("No measurements for this run yet. Go to COLLECT.")
        st.button("Close", on_click=_go, args=("mr_menu",))
        _frame_end()
        return

    # Select a measurement (one of the collected regimes)
    available = [r for r in REGIMES if r in data]
    sel = st.selectbox("Select Measurement", available, format_func=lambda rr: REGIME_LABEL[rr])
    m = data[sel]

    left, right = st.columns([0.50, 0.50], gap="medium")
    with left:
        st.markdown(f"<div class='vxp-mono' style='height:520px; overflow:auto;'>{_legacy_results_text(view_run, data)}</div>", unsafe_allow_html=True)
    with right:
        st.pyplot(_plot_track_marker(m), clear_figure=True)
        st.pyplot(_plot_polar(m), clear_figure=True)

    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()

def screen_settings() -> None:
    _frame_start("SETTINGS")
    st.write("Ajustes (simulación) por condición. Para el BO105 se permiten:")
    st.write("• **Pitch links**, **Trim tabs** y **Weights**.")
    view_run = _run_selector_inline()
    st.caption(f"Viewing adjustments used for RUN {view_run} (edits affect the current run settings).")

    regime = st.selectbox("Regime", options=REGIMES, format_func=lambda r: REGIME_LABEL[r])
    adj = st.session_state.vxp_adjustments[regime]

    hdr = st.columns([0.20, 0.27, 0.27, 0.26])
    hdr[0].markdown("**Blade**")
    hdr[1].markdown("**Pitch link (turns)**")
    hdr[2].markdown("**Trim tab (mm)**")
    hdr[3].markdown("**Bolt weight (g)**")

    for b in BLADES:
        row = st.columns([0.20, 0.27, 0.27, 0.26])
        row[0].markdown(f"{b}")
        adj["pitch_turns"][b] = float(row[1].number_input("", value=float(adj["pitch_turns"][b]), step=0.25, key=f"pl_{regime}_{b}"))
        adj["trim_mm"][b] = float(row[2].number_input("", value=float(adj["trim_mm"][b]), step=0.5, key=f"tt_{regime}_{b}"))
        adj["bolt_g"][b] = float(row[3].number_input("", value=float(adj["bolt_g"][b]), step=5.0, key=f"wt_{regime}_{b}"))

    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()

def screen_solution() -> None:
    _frame_start("SOLUTION")
    view_run = _run_selector_inline()
    data = _current_run_data(view_run)

    if not data:
        st.write("No measurements for this run yet. Go to COLLECT.")
        st.button("Close", on_click=_go, args=("mr_menu",))
        _frame_end()
        return

    st.selectbox("", options=["BALANCE ONLY", "TRACK ONLY", "TRACK + BALANCE"], index=2, key="sol_type")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.button("GRAPHICAL SOLUTION", use_container_width=True, on_click=_go, args=("solution_graph",))
    st.button("SHOW SOLUTION", use_container_width=True, on_click=_go, args=("solution_text",))
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.button("EDIT SOLUTION", use_container_width=True, on_click=_go, args=("edit_solution",))
    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()

def screen_solution_graph() -> None:
    _frame_start("RESULTS")
    view_run = _run_selector_inline()
    data = _current_run_data(view_run)
    if not data:
        st.write("No measurements for this run yet.")
        st.button("Close", on_click=_go, args=("solution",))
        _frame_end()
        return

    # Header controls (decorative like VXP)
    top = st.columns([0.22,0.22,0.22,0.34])
    top[0].selectbox("Maximize", ["Normal","Maximize"], index=0)
    top[1].selectbox("Run", [str(r) for r in sorted(st.session_state.vxp_runs.keys())], index=sorted(st.session_state.vxp_runs.keys()).index(view_run))
    top[2].selectbox("Blade Ref1", ["YEL"], index=0)
    top[3].markdown("<div style='text-align:right; font-weight:900;'> </div>", unsafe_allow_html=True)

    # Choose measurement for visual track/polar
    available = [r for r in REGIMES if r in data]
    sel = st.selectbox("Select Bal Meas", available, format_func=lambda rr: REGIME_LABEL[rr])
    m = data[sel]

    left, right = st.columns([0.50, 0.50], gap="medium")
    with left:
        st.markdown(f"<div class='vxp-mono' style='height:590px; overflow:auto;'>{_legacy_results_text(view_run, data)}</div>", unsafe_allow_html=True)

        st.markdown("<div class='vxp-label' style='margin-top:10px;'>APPLIED CHANGES (THIS RUN)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='vxp-mono' style='height:120px; overflow:auto;'>{_applied_changes_text(view_run)}</div>", unsafe_allow_html=True)

    with right:
        st.pyplot(_plot_track_marker(m), clear_figure=True)
        st.pyplot(_plot_track_graph(data), clear_figure=True)
        st.pyplot(_plot_polar(m), clear_figure=True)

    st.button("Close", on_click=_go, args=("solution",))
    _frame_end()

def screen_solution_text() -> None:
    _frame_start("SHOW SOLUTION")
    view_run = _run_selector_inline()
    data = _current_run_data(view_run)

    if not data:
        st.write("No measurements for this run yet.")
        st.button("Close", on_click=_go, args=("solution",))
        _frame_end()
        return

    # Build a guided instruction block (what to adjust next)
    lines: List[str] = []
    lines.append(_legacy_results_text(view_run, data))
    lines.append("----- Suggested Next Action (Training) -----")

    if view_run == 1:
        sug = _suggest_pitchlink(data)
        lines.append("RUN 1: Correct TRACKING using Pitch Link (flats).")
        for b in BLADES:
            flats = sug[b]*6.0
            if abs(flats) >= 0.5:
                lines.append(f"  {b}: {flats:+.1f} flats (≈ {sug[b]:+.2f} turns)")
    elif view_run == 2:
        sug = _suggest_trimtabs(data)
        lines.append("RUN 2: Correct FORWARD FLIGHT TRACK using Trim Tabs (Tab Sta 5/6).")
        for b in BLADES:
            if abs(sug[b]) >= 0.25:
                lines.append(f"  {b}: {sug[b]:+.2f} mm (equiv)")
    else:
        blade, grams = _suggest_weight(data)
        lines.append("RUN 3: Correct 1/REV VIBRATION using Weight (plqts).")
        lines.append(f"  Add ~{grams:.0f} g at {blade} bolt (≈ {grams/10:.1f} plqts)")
        if len(_completed_set(3)) == 3 and _all_ok_for_run(3, data):
            lines.append("")
            lines.append("✓ PARAMETERS OK — TRAINING COMPLETE")

    report = "\n".join(lines)
    st.markdown(f"<div class='vxp-mono' style='height:680px; overflow:auto;'>{report}</div>", unsafe_allow_html=True)

    st.button("Close", on_click=_go, args=("solution",))
    _frame_end()

def screen_edit_solution() -> None:
    _frame_start("EDIT SOLUTION")
    run = int(st.session_state.vxp_run)
    view_run = _run_selector_inline()
    st.caption("Select the appropriate adjustment.")

    # Buttons like VXP
    for label, key in EDIT_ITEMS:
        if st.button(label, use_container_width=True, key=f"edit_{key}"):
            st.session_state.vxp_edit_item = key
            _go("edit_solution_item")
            st.rerun()

    st.button("Close", on_click=_go, args=("solution",))
    _frame_end()

def screen_edit_solution_item() -> None:
    # Input screen that resembles the VXP sub-screen (BLU/GRN/YEL/RED with a Close button)
    run = int(st.session_state.vxp_run)
    item = st.session_state.get("vxp_edit_item", "pitch")
    label = dict(EDIT_ITEMS).get(item, "EDIT")

    _frame_start(label)

    # Determine suggested defaults based on current run data
    data = _current_run_data(run)

    if item == "pitch":
        sug = _suggest_pitchlink(data)
        st.markdown("<div class='vxp-label'>INPUT THE CURRENT ADJUSTMENT MADE (FLATS)</div>", unsafe_allow_html=True)
        for b in BLADES:
            k = f"pitch_flats_{b}"
            st.session_state.vxp_edit_values.setdefault(k, sug[b]*6.0)
    elif item in ("tab5","tab6"):
        sug = _suggest_trimtabs(data)
        st.markdown("<div class='vxp-label'>INPUT THE CURRENT ADJUSTMENT MADE (DEG)</div>", unsafe_allow_html=True)
        for b in BLADES:
            k = f"{item}_deg_{b}"
            st.session_state.vxp_edit_values.setdefault(k, sug[b])
    else:
        blade, grams = _suggest_weight(data)
        st.markdown("<div class='vxp-label'>INPUT THE CURRENT ADJUSTMENT MADE (PLQTS)</div>", unsafe_allow_html=True)
        for b in BLADES:
            k = f"wt_plqts_{b}"
            st.session_state.vxp_edit_values.setdefault(k, (grams/10.0) if b == blade else 0.0)

    # Layout like original: Blade labels left, input boxes center, units right
    for b in BLADES:
        row = st.columns([0.18, 0.55, 0.27])
        row[0].markdown(f"<div class='vxp-label'>{b}</div>", unsafe_allow_html=True)

        if item == "pitch":
            v = row[1].number_input("", value=float(st.session_state.vxp_edit_values[f"pitch_flats_{b}"]), step=0.5, key=f"in_pitch_{b}")
            st.session_state.vxp_edit_values[f"pitch_flats_{b}"] = float(v)
            row[2].markdown("<div class='vxp-label'>flats</div>", unsafe_allow_html=True)

        elif item in ("tab5","tab6"):
            v = row[1].number_input("", value=float(st.session_state.vxp_edit_values[f"{item}_deg_{b}"]), step=0.5, key=f"in_{item}_{b}")
            st.session_state.vxp_edit_values[f"{item}_deg_{b}"] = float(v)
            row[2].markdown("<div class='vxp-label'>deg</div>", unsafe_allow_html=True)

        else:
            v = row[1].number_input("", value=float(st.session_state.vxp_edit_values[f"wt_plqts_{b}"]), step=0.5, key=f"in_wt_{b}")
            st.session_state.vxp_edit_values[f"wt_plqts_{b}"] = float(v)
            row[2].markdown("<div class='vxp-label'>plqts</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Apply: update settings immediately + record applied changes for this RUN
    def apply_changes():
        applied_lines = st.session_state.vxp_applied_changes.setdefault(run, [])

        if item == "pitch":
            for b in BLADES:
                flats = float(st.session_state.vxp_edit_values[f"pitch_flats_{b}"])
                turns = _flats_to_turns(flats)
                for r in REGIMES:
                    st.session_state.vxp_adjustments[r]["pitch_turns"][b] = turns
            applied_lines.append("Pitch Link (flats): " + ", ".join([f"{b} {st.session_state.vxp_edit_values[f'pitch_flats_{b}']:+.1f}" for b in BLADES]))

        elif item in ("tab5","tab6"):
            # apply to HORIZONTAL only (as in our sim)
            for b in BLADES:
                deg = float(st.session_state.vxp_edit_values[f"{item}_deg_{b}"])
                st.session_state.vxp_adjustments["HORIZONTAL"]["trim_mm"][b] = _deg_to_trim_mm(deg)
            applied_lines.append(f"{label}: " + ", ".join([f"{b} {st.session_state.vxp_edit_values[f'{item}_deg_{b}']:+.1f}" for b in BLADES]))

        else:
            # apply to all regimes
            for b in BLADES:
                plqts = float(st.session_state.vxp_edit_values[f"wt_plqts_{b}"])
                grams = _plqts_to_grams(plqts)
                for r in REGIMES:
                    st.session_state.vxp_adjustments[r]["bolt_g"][b] = grams
            applied_lines.append("Weight (plqts): " + ", ".join([f"{b} {st.session_state.vxp_edit_values[f'wt_plqts_{b}']:+.1f}" for b in BLADES]))

        _go("edit_solution")
        st.rerun()

    st.button("Close", on_click=_go, args=("edit_solution",))
    st.button("Apply", on_click=apply_changes)

    _frame_end()

def screen_next_run_prompt() -> None:
    # VXP-like prompt shown from inside Main Rotor menu
    _frame_start("NEXT RUN")

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    def start_next(apply: bool):
        # only allow up to RUN 3 in this training script
        cur = int(st.session_state.vxp_run)
        if cur >= 3:
            _go("mr_menu")
            return
        nxt = cur + 1
        st.session_state.vxp_runs.setdefault(nxt, {})
        st.session_state.vxp_completed_by_run.setdefault(nxt, set())
        st.session_state.vxp_run = nxt
        st.session_state.vxp_view_run = nxt
        _go("mr_menu")
        st.rerun()

    st.button("UPDATE SETTINGS - START NEXT RUN", use_container_width=True, on_click=start_next, kwargs={"apply": True})
    st.button("NO CHANGES MADE - START NEXT RUN", use_container_width=True, on_click=start_next, kwargs={"apply": False})
    st.button("CANCEL - STAY ON RUN", use_container_width=True, on_click=_go, args=("mr_menu",))

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()

# ----------------------------------------------------------
# Session init
# ----------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("vxp_screen", "home")
    st.session_state.setdefault("vxp_run", 1)

    # Only Run 1 exists at start (Run2/3 via Next Run)
    st.session_state.setdefault("vxp_runs", {1: {}})
    st.session_state.setdefault("vxp_completed_by_run", {1: set()})

    # Viewing run selector
    st.session_state.setdefault("vxp_view_run", 1)

    st.session_state.setdefault("vxp_aircraft", {"weight": 0.0, "cg": 0.0, "hours": 0.0, "initials": ""})
    st.session_state.setdefault("vxp_note_codes", {1})  # start with Balance

    st.session_state.setdefault("vxp_adjustments", _default_adjustments())

    st.session_state.setdefault("vxp_pending_regime", None)
    st.session_state.setdefault("vxp_acq_in_progress", False)

    # Edit solution state
    st.session_state.setdefault("vxp_edit_item", "pitch")
    st.session_state.setdefault("vxp_edit_values", {})
    st.session_state.setdefault("vxp_applied_changes", {})  # run -> list[str]

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="VXP Simulator – BO105", layout="centered")
    _init_state()
    _apply_nav()

    # Layout: toolbar + main window
    tcol, maincol = st.columns([0.20, 0.80], gap="small")
    with tcol:
        _toolbar()

    with maincol:
        scr = st.session_state.vxp_screen
        if scr == "home":
            screen_home()
        elif scr == "not_impl":
            screen_not_impl()
        elif scr == "aircraft_info":
            screen_aircraft_info()
        elif scr == "note_codes":
            screen_note_codes()
        elif scr == "mr_menu":
            screen_mr_menu()
        elif scr == "collect":
            screen_collect()
        elif scr == "acquire":
            screen_acquire()
        elif scr == "meas_list":
            screen_meas_list()
        elif scr == "meas_graph":
            screen_meas_graph()
        elif scr == "settings":
            screen_settings()
        elif scr == "solution":
            screen_solution()
        elif scr == "solution_graph":
            screen_solution_graph()
        elif scr == "solution_text":
            screen_solution_text()
        elif scr == "edit_solution":
            screen_edit_solution()
        elif scr == "edit_solution_item":
            screen_edit_solution_item()
        elif scr == "next_run_prompt":
            screen_next_run_prompt()
        else:
            st.session_state.vxp_screen = "home"
            st.rerun()

if __name__ == "__main__":
    main()
"""
# Inject b64 icons into session_state at init by adding a small block at top of file
# We'll place icons dict into st.session_state on first run (after imports)
icons_dict_literal = "{\n" + ",\n".join([f'    "{k}": "{v}"' for k,v in icons_b64.items()]) + "\n}\n"

# Patch: after streamlit import, insert icons dict init
marker = "import streamlit as st\n"
parts = new_code.split(marker)
assert len(parts) == 2
new_code = parts[0] + marker + f"\n\n# Embedded toolbar icons (base64 PNG)\n_VXP_ICONS_B64 = {icons_dict_literal}\n\n# Ensure icons present in session_state\nif 'vxp_icons_b64' not in st.session_state:\n    st.session_state.vxp_icons_b64 = _VXP_ICONS_B64\n\n" + parts[1]

app_py.write_text(new_code, encoding="utf-8")

# Update requirements to include pillow (already installed but for completeness)
req = proj / "requirements.txt"
req.write_text("streamlit\nmatplotlib\nnumpy\nPillow\n", encoding="utf-8")

# Create README quick
(proj/"README.md").write_text("""# VXP Simulator (BO105) – Training Prototype

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
