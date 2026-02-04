from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ==========================================================
# Vibrex VXP (training simulator) – BO105 Main Rotor
# ----------------------------------------------------------
# This is a UI/logic prototype to mimic the legacy VXP flow
# shown in the screenshots.
#
# Added for your request:
# - "Tracking & Balance – Option B" (Strobex mode B cue)
# - Run handling with persistent Run 1 / Run 2 data (and beyond)
#
# IMPORTANT: Training/simulation only. Not for real aircraft work.
# ==========================================================

# ---- Rotor config (BO105: 4 blades) ----
BLADES: List[str] = ["BLU", "GRN", "YEL", "RED"]
BLADE_FULL = {"BLU": "Blue", "GRN": "Green", "YEL": "Yellow (Ref)", "RED": "Red"}

# Clock/azimuth model used by this simulator:
# 0° = 12 o'clock (nose/forward), increasing clockwise.
BLADE_CLOCK_DEG = {
    "YEL": 0.0,
    "RED": 90.0,
    "BLU": 180.0,
    "GRN": 270.0,
}

REGIMES: List[str] = ["GROUND", "HOVER", "HORIZONTAL"]
REGIME_LABEL = {
    "GROUND": "100% Ground",
    "HOVER": "Hover Flight",
    "HORIZONTAL": "Horizontal Flight",
}

# ----------------------------
# Tracking & Balance option
# ----------------------------
# "Option B" (training) cue: Strobex tracker Mode switch = B.
TRACKING_OPTION = "B"
STROBEX_MODE_SWITCH = "B"
# Dial value is shown in the BK117 training excerpt; for BO105 we keep it as a
# UI hint only (simulator-only, not a real setup value).
STROBEX_RPM_DIAL = 614

# BO105 setup cue from AMM excerpt: Balancer/Phazor R.P.M. Tune = 424.
# For the simulator we use this as the displayed rotor RPM during COLLECT.
BO105_DISPLAY_RPM = 424.0


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class BalanceReading:
    amp_ips: float
    phase_deg: float  # 0..360 (clock deg)
    rpm: float


@dataclass
class Measurement:
    regime: str
    balance: BalanceReading
    track_mm: Dict[str, float]  # per blade, relative to YEL reference


# ----------------------------
# Legacy / XP look
# ----------------------------

XP_CSS = """
<style>
/* Hide Streamlit chrome */
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu {
  display: none !important;
}

/* Page background */
html, body, [data-testid="stAppViewContainer"] {
  background: #bfbfbf;
  font-family: Tahoma, "MS Sans Serif", Verdana, Arial, sans-serif;
  font-size: 11px;
}

/* Remove Streamlit default paddings a bit */
.block-container {
  padding-top: 0.15rem;
  padding-bottom: 0.75rem;
  max-width: 1120px;
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
  padding: 4px 8px;
  height: 26px;
  font-weight: 700;
  letter-spacing: 0.2px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.vxp-menubar {
  background: #d4d0c8;
  border-bottom: 1px solid #808080;
  padding: 3px 8px;
  font-size: 12px;
}

.vxp-content {
  padding: 12px;
  background: #c0c0c0;
}

/* Toolbar */
.vxp-toolbar {
  background: #d4d0c8;
  border: 2px solid #808080;
  box-shadow: inset 1px 1px 0px #ffffff;
  padding: 8px;
  border-radius: 2px;
}

/* Buttons – try to mimic XP */
.stButton > button {
  background: #d4d0c8 !important;
  color: #000 !important;
  border-top: 2px solid #ffffff !important;
  border-left: 2px solid #ffffff !important;
  border-right: 2px solid #404040 !important;
  border-bottom: 2px solid #404040 !important;
  border-radius: 0px !important;
  font-weight: 400;
  font-size: 12px;
  padding: 8px 10px !important;
}

/* Left toolbar buttons (smaller, icon-ish) */
.vxp-toolbar .stButton > button {
  font-size: 12px !important;
  padding: 6px 8px !important;
  text-align: left !important;
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
}

/* Monospace output */
.vxp-mono {
  font-family: "Courier New", Courier, monospace;
  font-size: 12px;
  white-space: pre;
  background: #efefef;
  border: 2px solid #808080;
  padding: 10px;
}

/* Small status line */
.vxp-status {
  font-size: 12px;
  color: #111;
  margin-top: 8px;
}

/* Hide Streamlit footer */
footer {visibility: hidden;}
</style>
"""


# ----------------------------
# Math helpers
# ----------------------------

def _vec_from_clock_deg(theta_deg: float) -> np.ndarray:
    """Convert clock degrees (0=north, cw positive) to xy unit vector."""
    phi = math.radians(90.0 - theta_deg)  # standard math angle (0=x, ccw)
    return np.array([math.cos(phi), math.sin(phi)], dtype=float)


def _clock_deg_from_vec(v: np.ndarray) -> float:
    x, y = float(v[0]), float(v[1])
    phi = math.degrees(math.atan2(y, x))  # -180..180 from +x axis
    theta = (90.0 - phi) % 360.0
    return theta


def _clock_label(theta_deg: float) -> str:
    # 12 positions (hours)
    hour = int(round(theta_deg / 30.0)) % 12
    hour = 12 if hour == 0 else hour
    minute = 0 if abs((theta_deg / 30.0) - round(theta_deg / 30.0)) < 0.25 else 30
    if minute == 0:
        return f"{hour:02d}:00"
    return f"{hour:02d}:30"


def _round_quarter_turn(x: float) -> float:
    return round(x * 4.0) / 4.0


# ----------------------------
# Simulation model (simple)
# ----------------------------

# Base track (mm, relative to YEL)
BASE_TRACK: Dict[str, Dict[str, float]] = {
    "GROUND": {"BLU": 12.0, "GRN": -5.0, "YEL": 0.0, "RED": -10.0},
    "HOVER": {"BLU": 7.0, "GRN": -3.0, "YEL": 0.0, "RED": -6.0},
    "HORIZONTAL": {"BLU": 18.0, "GRN": -10.0, "YEL": 0.0, "RED": -5.0},
}

# Base 1/rev imbalance in IPS + clock phase (training values)
BASE_BALANCE: Dict[str, Tuple[float, float]] = {
    "GROUND": (0.32, 125.0),
    "HOVER": (0.11, 110.0),
    "HORIZONTAL": (0.08, 95.0),
}

# Sensitivities (training-only).
PITCHLINK_MM_PER_TURN = 10.0  # 1 full turn ~ 10 mm at tip (AMM statement)
TRIMTAB_MMTRACK_PER_MM = 15.0  # 1 mm tab bend ~ 15 mm track change at tip
BOLT_IPS_PER_GRAM = 0.0020  # 50 g ~ 0.10 ips (approx training scaling)


def _default_adjustments() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-regime adjustable items. (In reality some are not regime-specific, but this matches the requested UI.)"""
    return {
        r: {
            "pitch_turns": {b: 0.0 for b in BLADES},
            "trim_mm": {b: 0.0 for b in BLADES},
            "bolt_g": {b: 0.0 for b in BLADES},
        }
        for r in REGIMES
    }


def _simulate_measurement(regime: str) -> Measurement:
    adj = st.session_state.vxp_adjustments[regime]

    # Track
    track: Dict[str, float] = {}
    for b in BLADES:
        base = BASE_TRACK[regime][b]
        pitch_effect = PITCHLINK_MM_PER_TURN * float(adj["pitch_turns"][b])
        trim_effect = 0.0
        if regime == "HORIZONTAL":
            trim_effect = TRIMTAB_MMTRACK_PER_MM * float(adj["trim_mm"][b])
        noise = random.gauss(0.0, 0.6)
        track[b] = float(base + pitch_effect + trim_effect + noise)

    # Force YEL to be the reference (0.0) to mimic the legacy "relative to YEL" display.
    yel0 = float(track.get("YEL", 0.0))
    for b in BLADES:
        track[b] = float(track[b] - yel0)
    track["YEL"] = 0.0

    # Balance (vector)
    base_amp, base_phase = BASE_BALANCE[regime]
    v = _vec_from_clock_deg(base_phase) * float(base_amp)

    for b in BLADES:
        grams = float(adj["bolt_g"][b])
        v += (-BOLT_IPS_PER_GRAM * grams) * _vec_from_clock_deg(BLADE_CLOCK_DEG[b])

    # small noise
    v += np.array([random.gauss(0.0, 0.004), random.gauss(0.0, 0.004)], dtype=float)

    amp = float(np.linalg.norm(v))
    phase = float(_clock_deg_from_vec(v)) if amp > 1e-6 else 0.0

    return Measurement(
        regime=regime,
        balance=BalanceReading(amp_ips=amp, phase_deg=phase, rpm=BO105_DISPLAY_RPM),
        track_mm=track,
    )


# ----------------------------
# Solution logic (training)
# ----------------------------

def _suggest_track_corrections(meas: Measurement) -> List[str]:
    out: List[str] = []
    regime = meas.regime

    # Tolerances (from AMM excerpts)
    if regime == "GROUND":
        tol = 10.0
        corr = "pitch link turns"
        scale = PITCHLINK_MM_PER_TURN
    elif regime == "HOVER":
        tol = 5.0
        corr = "pitch link turns"
        scale = PITCHLINK_MM_PER_TURN
    else:
        tol = 5.0
        corr = "trim tab (mm)"
        scale = TRIMTAB_MMTRACK_PER_MM

    vals = {b: meas.track_mm[b] for b in BLADES}
    spread = max(vals.values()) - min(vals.values())

    out.append(f"TRACK ({REGIME_LABEL[regime]}): spread {spread:+.1f} mm (limit ~{tol:.0f} mm).")

    if spread <= tol:
        out.append("  ✓ Track within limits for this regime.")
        return out

    # Suggest corrections for the worst blade(s)
    worst_high = max(vals, key=lambda k: vals[k])
    worst_low = min(vals, key=lambda k: vals[k])

    # do not touch reference blade unless it's the only option
    protected = {"YEL"}

    candidates = [worst_high, worst_low]
    for b in candidates:
        if b in protected and any(x not in protected for x in candidates):
            continue
        dev = vals[b]
        # We want to drive dev toward 0
        raw = -dev / scale
        rec = _round_quarter_turn(raw)
        if regime in ("GROUND", "HOVER"):
            direction = "CW (lowers tip)" if rec < 0 else "CCW (raises tip)"
            out.append(f"  • {b} ({BLADE_FULL[b]}): {corr} {rec:+.2f} turns ({direction})")
        else:
            # trim tab mm
            rec = max(-5.0, min(5.0, rec))
            direction = "down (lowers tip)" if rec < 0 else "up (raises tip)"
            out.append(f"  • {b} ({BLADE_FULL[b]}): {corr} {rec:+.2f} mm ({direction})")

    return out


def _suggest_balance_corrections(meas: Measurement) -> List[str]:
    out: List[str] = []
    amp = meas.balance.amp_ips
    phase = meas.balance.phase_deg

    # Limits from AMM excerpts (training reference)
    if meas.regime == "GROUND":
        limit = 0.40
    else:
        limit = 0.05

    out.append(
        f"BALANCE ({REGIME_LABEL[meas.regime]}): {amp:.3f} ips @ {phase:.1f}° ({_clock_label(phase)}). Limit ~{limit:.2f} ips."
    )

    if amp <= limit:
        out.append("  ✓ Balance within limits for this regime.")
        return out

    # Recommend adding weight opposite the imbalance vector
    target = (phase + 180.0) % 360.0

    # Choose nearest blade bolt location
    def dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    blade = min(BLADES, key=lambda bb: dist(target, BLADE_CLOCK_DEG[bb]))

    # Training scaling: grams proportional to ips
    grams = int(round(amp / BOLT_IPS_PER_GRAM))
    grams = max(5, min(120, grams))

    out.append(
        f"  • Add ~{grams} g to blade bolt of {blade} ({BLADE_FULL[blade]}) near {target:.0f}° ({_clock_label(target)})."
    )
    out.append("  • Re-acquire after the weight change.")

    return out


def _solution_report(meas_by_regime: Dict[str, Measurement]) -> str:
    lines: List[str] = []
    lines.append("VIBREX VXP – MAIN ROTOR (BO105) – TRAINING OUTPUT")
    lines.append(f"PROCEDURE: Tracking & Balance – Option {TRACKING_OPTION} (Strobex mode {STROBEX_MODE_SWITCH})")
    lines.append(f"RUN: {st.session_state.vxp_run}")
    lines.append("")

    for r in REGIMES:
        if r not in meas_by_regime:
            continue
        m = meas_by_regime[r]
        lines.append("=" * 62)
        lines.append(f"REGIME: {REGIME_LABEL[r]}")
        lines.append("")
        # Track table
        lines.append("TRACK (mm, relative to YEL):")
        for b in BLADES:
            lines.append(f"  {b}: {m.track_mm[b]:+6.1f}")
        lines.append("")
        # Balance
        lines.append(
            f"BALANCE: {m.balance.amp_ips:.3f} ips @ {_clock_label(m.balance.phase_deg)} ({m.balance.phase_deg:.1f}°)"
        )
        lines.append("")
        # Suggested actions
        lines.append("SUGGESTED CORRECTIONS:")
        for s in _suggest_track_corrections(m) + _suggest_balance_corrections(m):
            lines.append(f"{s}")
        lines.append("")

    return "\n".join(lines)


def _legacy_solution_screen_text(meas_by_regime: Dict[str, Measurement]) -> str:
    """Compact text block intended to resemble the VXP result screen."""
    run = int(st.session_state.vxp_run)
    lines: List[str] = []
    lines.append(f"BO105   MAIN ROTOR TRACK & BALANCE (OPT {TRACKING_OPTION})")
    lines.append(f"RUN: {run}    ID: training")
    lines.append("")
    lines.append("----- Balance Measurements -----")
    for r in REGIMES:
        if r not in meas_by_regime:
            continue
        m = meas_by_regime[r]
        lines.append(
            f"{REGIME_LABEL[r]:<18}  1P {m.balance.amp_ips:0.2f} IPS  {_clock_label(m.balance.phase_deg):>5}  RPM:{m.balance.rpm:0.0f}"
        )
    lines.append("")
    lines.append("----- Track Height (mm rel. YEL) -----")
    for r in REGIMES:
        if r not in meas_by_regime:
            continue
        m = meas_by_regime[r]
        row = "  ".join([f"{b}:{m.track_mm[b]:+5.1f}" for b in BLADES])
        lines.append(f"{REGIME_LABEL[r]:<18}  {row}")

    lines.append("")
    lines.append("----- Solution (training) -----")
    for r in REGIMES:
        if r not in meas_by_regime:
            continue
        m = meas_by_regime[r]
        lines.append(f"{REGIME_LABEL[r]}:")
        for s in _suggest_track_corrections(m):
            lines.append(f"  {s}")
        for s in _suggest_balance_corrections(m):
            lines.append(f"  {s}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------
# UI helpers
# ----------------------------

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
    st.markdown("<div class='vxp-toolbar'>", unsafe_allow_html=True)
    # Small, left-side legacy toolbar (icon-ish labels)
    st.button("✖  Disconnect", use_container_width=True, on_click=_go, args=("home",))
    st.button("⬆  Upload", use_container_width=True)
    st.button("⬇  Download", use_container_width=True)
    st.button("📄  View Log", use_container_width=True)
    st.button("🖨  Print AU", use_container_width=True)
    st.button("?  Help", use_container_width=True)
    st.button("↩  Exit", use_container_width=True, on_click=_go, args=("home",))
    st.markdown("</div>", unsafe_allow_html=True)


def _header_line(left: str, right: str = "") -> None:
    cols = st.columns([0.7, 0.3])
    cols[0].markdown(f"<div style='font-weight:400;'>{left}</div>", unsafe_allow_html=True)
    cols[1].markdown(
        f"<div style='text-align:right; font-weight:400;'>{right}</div>",
        unsafe_allow_html=True,
    )


# ----------------------------
# Screens
# ----------------------------

def screen_home() -> None:
    _frame_start("Chadwick-Helmuth VXP  —  BO105 (Training)")

    _header_line("Select Procedure:")
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    def btn(label: str, screen: str):
        st.button(label, use_container_width=True, on_click=_go, args=(screen,))

    btn("Aircraft Info", "aircraft_info")

    # Main Rotor Tracking & Balance (Option B) – Run 1 / Run 2
    st.button(
        "Main Rotor Track & Balance Run 1",
        use_container_width=True,
        on_click=_go,
        args=("mr_menu",),
        kwargs={"vxp_run": 1},
    )

    has_run2 = 2 in st.session_state.vxp_runs
    st.button(
        "Main Rotor Track & Balance Run 2",
        use_container_width=True,
        disabled=not has_run2,
        on_click=_go,
        args=("mr_menu",),
        kwargs={"vxp_run": 2},
    )
    btn("Tail Rotor Balance Run 1", "not_impl")
    btn("T/R Driveshaft Balance Run 1", "not_impl")
    btn("Vibration Signatures", "not_impl")
    btn("Measurements Only", "not_impl")
    btn("Setup / Utilities", "not_impl")

    st.markdown("<div class='vxp-status'>Training prototype. BO105 procedure flow only.</div>", unsafe_allow_html=True)
    _frame_end()


def screen_not_impl() -> None:
    _frame_start("VXP  —  Not Implemented")
    st.write("Solo se implementa **Main Rotor – Tracking & Balance (Option B)** para el BO105 (RUN 1 / RUN 2).")
    st.button("Close", on_click=_go, args=("home",))
    _frame_end()


def screen_aircraft_info() -> None:
    _frame_start("AIRCRAFT INFO")

    info = st.session_state.vxp_aircraft

    c1, c2 = st.columns([0.35, 0.65], gap="large")
    with c1:
        st.write("WEIGHT:")
        st.write("C.G.:")
        st.write("HOURS:")
        st.write("INITIALS:")
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
                if code in selected:
                    selected.remove(code)
                else:
                    selected.add(code)
                st.rerun()
        with cols[1]:
            st.markdown(f"<div style='font-size:20px; font-weight:900; padding-top:10px;'>{checked}</div>", unsafe_allow_html=True)

    st.button("Close", on_click=_go, args=("aircraft_info",))
    _frame_end()


def screen_mr_menu() -> None:
    _frame_start(f"Main Rotor Balance Run {st.session_state.vxp_run}")

    st.caption(f"Tracking & Balance – Option {TRACKING_OPTION} (Strobex mode {STROBEX_MODE_SWITCH}).")

    def btn(label: str, screen: str):
        st.button(label, use_container_width=True, on_click=_go, args=(screen,))

    btn("COLLECT", "collect")
    btn("MEASUREMENTS LIST", "meas_list")
    btn("MEASUREMENTS GRAPH", "meas_graph")
    btn("SETTINGS", "settings")
    btn("SOLUTION", "solution")
    btn("NEXT RUN", "next_run")
    btn("TEST SENSORS", "not_impl")
    btn("FASTRAK OPTIONS", "not_impl")

    st.button("Close", on_click=_go, args=("home",))
    _frame_end()


def screen_next_run() -> None:
    """Create/select the next run (RUN 2, RUN 3, ...).

    In VXP the operator usually applies corrections and then presses NEXT RUN
    to re-acquire. We keep the per-regime adjustments, but start with empty
    measurements for the new run.
    """

    runs: Dict[int, Dict[str, Measurement]] = st.session_state.vxp_runs
    next_run = (max(runs.keys()) + 1) if runs else 1
    runs[next_run] = {}

    completed_by_run: Dict[int, set] = st.session_state.vxp_completed_by_run
    completed_by_run[next_run] = set()

    st.session_state.vxp_run = next_run
    st.session_state.vxp_pending_regime = None
    st.session_state.vxp_acq_in_progress = False

    _go("mr_menu")
    st.rerun()


def _current_run_data() -> Dict[str, Measurement]:
    return st.session_state.vxp_runs.setdefault(int(st.session_state.vxp_run), {})


def _current_completed() -> set:
    completed_by_run: Dict[int, set] = st.session_state.vxp_completed_by_run
    return completed_by_run.setdefault(int(st.session_state.vxp_run), set())


def _current_completed() -> set:
    by_run: Dict[int, set] = st.session_state.vxp_completed_by_run
    return by_run.setdefault(int(st.session_state.vxp_run), set())


def screen_collect() -> None:
    _frame_start(f"Main Rotor: Run {st.session_state.vxp_run}   —   Day Mode")

    run_data = _current_run_data()

    _header_line(f"RPM   {BO105_DISPLAY_RPM:.1f}", f"Run {st.session_state.vxp_run}")

    # show regime buttons with checkmarks
    completed = _current_completed()
    for r in REGIMES:
        cols = st.columns([0.86, 0.14])
        with cols[0]:
            if st.button(REGIME_LABEL[r], use_container_width=True, key=f"reg_{r}"):
                st.session_state.vxp_pending_regime = r
                _go("acquire")
                st.rerun()
        with cols[1]:
            mark = "✓" if r in completed else ""
            st.markdown(f"<div style='font-size:22px; font-weight:900; padding-top:10px;'>{mark}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.button("Close", on_click=_go, args=("mr_menu",))

    _frame_end()


def screen_acquire() -> None:
    _frame_start("ACQUIRING …")
    r = st.session_state.get("vxp_pending_regime")
    if not r:
        st.write("No regime selected.")
        st.button("Close", on_click=_go, args=("collect",))
        _frame_end()
        return

    st.markdown(f"**{REGIME_LABEL[r]}**")
    st.markdown(f"RPM {BO105_DISPLAY_RPM:.1f}")
    st.markdown(f"Set Strobex: Mode switch: {STROBEX_MODE_SWITCH}   |   R.P.M. Dial: {STROBEX_RPM_DIAL}")

    # mimic two channels acquiring
    st.markdown("<div class='vxp-mono'>M/R LAT\t\tACQUIRING\n\nM/R OBT\t\tACQUIRING</div>", unsafe_allow_html=True)

    # Only run the progress simulation once per enter
    if not st.session_state.get("vxp_acq_in_progress", False):
        st.session_state.vxp_acq_in_progress = True
        p = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            p.progress(i + 1)

        meas = _simulate_measurement(r)
        _current_run_data()[r] = meas
        _current_completed().add(r)
        st.session_state.vxp_pending_regime = None
        st.session_state.vxp_acq_in_progress = False
        _go("collect")
        st.rerun()

    st.button("Close", on_click=_go, args=("collect",))
    _frame_end()


def screen_meas_list() -> None:
    _frame_start("MEASUREMENTS LIST")

    data = _current_run_data()

    if not data:
        st.write("No measurements yet. Go to COLLECT.")
        st.button("Close", on_click=_go, args=("mr_menu",))
        _frame_end()
        return

    # Build a legacy-looking text block
    lines: List[str] = []
    lines.append(f"BO105  MAIN ROTOR  —  RUN {st.session_state.vxp_run}")
    lines.append("")

    for r in REGIMES:
        if r not in data:
            continue
        m = data[r]
        lines.append(f"---- {REGIME_LABEL[r]} ----")
        lines.append(f"1P  AMP {m.balance.amp_ips:.3f} IPS   PHASE {_clock_label(m.balance.phase_deg)}")
        lines.append("TRACK HEIGHT (mm) RELATIVE TO YEL")
        for b in BLADES:
            lines.append(f"  {b}: {m.track_mm[b]:+6.1f}")
        lines.append("")

    block = "\n".join(lines)
    st.markdown(f"<div class='vxp-mono'>{block}</div>", unsafe_allow_html=True)

    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()


def _plot_track_graph(data: Dict[str, Measurement]) -> plt.Figure:
    """Legacy-style track graph (closer to VXP look)."""
    xs = [REGIME_LABEL[r] for r in REGIMES if r in data]
    fig = plt.figure(figsize=(6.6, 3.4), dpi=120)
    fig.patch.set_facecolor("#c0c0c0")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    if not xs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    for b in BLADES:
        ys = [data[r].track_mm[b] for r in REGIMES if r in data]
        ax.plot(xs, ys, marker="s", linewidth=1.5, markersize=4, label=b)

    # Match the older-ish scaling often seen (±32.5 mm)
    ax.set_ylim(-32.5, 32.5)
    ax.set_ylabel("mm")
    ax.set_title("Track Height (relative to YEL)", fontsize=10)
    ax.axhline(0.0, linewidth=1)

    # Simple, old-style grid
    ax.grid(True, linestyle=":", linewidth=0.7)

    # Frame/spines like a boxed control
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(1.0)

    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(pad=1.2)
    return fig


def _plot_polar(data: Dict[str, Measurement]) -> plt.Figure:
    """Legacy-style polar (clock) plot."""
    fig = plt.figure(figsize=(5.2, 5.2), dpi=120)
    fig.patch.set_facecolor("#c0c0c0")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("white")

    # 0 at 12 o'clock, clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Clock labels (12,1,2,...)
    ticks = [math.radians(t) for t in range(0, 360, 30)]
    labels = ["12", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)

    # Radial scale
    ax.set_rmax(max([data[r].balance.amp_ips for r in data] + [0.35]) * 1.2)
    ax.set_rlabel_position(135)
    ax.grid(True, linestyle=":", linewidth=0.7)

    for r in REGIMES:
        if r not in data:
            continue
        m = data[r]
        theta = math.radians(m.balance.phase_deg)
        ax.plot([theta], [m.balance.amp_ips], marker="o", markersize=6, label=REGIME_LABEL[r])
        ax.text(theta, m.balance.amp_ips + 0.01, f"{m.balance.amp_ips:.2f}", fontsize=8, ha="center")

    ax.set_title("1/rev Balance (IPS vs Phase)", fontsize=10, pad=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.14), fontsize=8, frameon=False)
    fig.tight_layout(pad=1.0)
    return fig


def _plot_bar(data: Dict[str, Measurement]) -> plt.Figure:
    xs = [REGIME_LABEL[r] for r in REGIMES if r in data]
    ys = [data[r].balance.amp_ips for r in REGIMES if r in data]
    fig = plt.figure(figsize=(4.0, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.bar(xs, ys)
    ax.set_ylabel("IPS")
    ax.set_title("1/rev Amplitude")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def screen_meas_graph() -> None:
    _frame_start("MEASUREMENTS GRAPH")

    data = _current_run_data()
    if not data:
        st.write("No measurements yet. Go to COLLECT.")
        st.button("Close", on_click=_go, args=("mr_menu",))
        _frame_end()
        return

    left, right = st.columns([0.52, 0.48], gap="medium")

    # Left: legacy list
    with left:
        lines: List[str] = []
        lines.append(f"BO105  MAIN ROTOR  —  RUN {st.session_state.vxp_run}")
        lines.append("")
        for r in REGIMES:
            if r not in data:
                continue
            m = data[r]
            lines.append(f"{REGIME_LABEL[r]}  1P {m.balance.amp_ips:.3f} IPS  {_clock_label(m.balance.phase_deg)}")
        lines.append("")
        lines.append("TRACK (mm rel. YEL)")
        for r in REGIMES:
            if r not in data:
                continue
            m = data[r]
            row = "  ".join([f"{b}:{m.track_mm[b]:+5.1f}" for b in BLADES])
            lines.append(f"{REGIME_LABEL[r]:<18} {row}")

        block = "\n".join(lines)
        st.markdown(
            f"<div class='vxp-mono' style='height:380px; overflow:auto;'>{block}</div>",
            unsafe_allow_html=True,
        )

    # Right: legacy-looking plots (closer to VXP screen)
    with right:
        st.pyplot(_plot_track_graph(data), clear_figure=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.button("Select Bal Meas", key="sel_bal_meas", use_container_width=False)
        st.pyplot(_plot_polar(data), clear_figure=True)

    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()


def screen_settings() -> None:
    _frame_start("SETTINGS")

    st.write("Ajustes (simulación) por condición. Para el BO105 se permiten:")
    st.write("• **Pitch links (turns)**, **Bolt weights (g)** y **Trim tabs (mm)**.")

    regime = st.selectbox("Regime", options=REGIMES, format_func=lambda r: REGIME_LABEL[r])
    adj = st.session_state.vxp_adjustments[regime]

    st.markdown("---")
    st.write("Introduce valores por pala (BLU/GRN/YEL/RED):")

    hdr = st.columns([0.18, 0.27, 0.27, 0.28])
    hdr[0].markdown("**Blade**")
    hdr[1].markdown("**Pitch link (turns)**")
    hdr[2].markdown("**Trim tab (mm)**")
    hdr[3].markdown("**Bolt weight (g)**")

    for b in BLADES:
        row = st.columns([0.18, 0.27, 0.27, 0.28])
        row[0].markdown(f"{b}  —  {BLADE_FULL[b]}")

        adj["pitch_turns"][b] = float(
            row[1].number_input(
                "",
                value=float(adj["pitch_turns"][b]),
                step=0.25,
                key=f"pl_{regime}_{b}",
            )
        )
        adj["trim_mm"][b] = float(
            row[2].number_input(
                "",
                value=float(adj["trim_mm"][b]),
                step=0.5,
                key=f"tt_{regime}_{b}",
            )
        )
        adj["bolt_g"][b] = float(
            row[3].number_input(
                "",
                value=float(adj["bolt_g"][b]),
                step=5.0,
                key=f"wt_{regime}_{b}",
            )
        )

    st.markdown("---")
    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()


def screen_solution() -> None:
    _frame_start("SOLUTION")

    data = _current_run_data()
    if not data:
        st.write("No measurements yet. Go to COLLECT.")
        st.button("Close", on_click=_go, args=("mr_menu",))
        _frame_end()
        return

    st.selectbox("", options=["BALANCE ONLY", "TRACK ONLY", "TRACK + BALANCE"], index=2, key="sol_type")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Two menus as requested
    # Two menus as requested (legacy wording)
    st.button("MEASUREMENTS GRAPH", use_container_width=True, on_click=_go, args=("solution_graph",))
    st.button("SHOW SOLUTION", use_container_width=True, on_click=_go, args=("solution_text",))

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.button("PREDICTIONS", use_container_width=True)
    st.button("EDIT SOLUTION", use_container_width=True)
    st.button("RESTORE SOLUTION", use_container_width=True)

    st.button("Close", on_click=_go, args=("mr_menu",))
    _frame_end()


def screen_solution_graph() -> None:
    # This screen aims to resemble the classic VXP "results" view.
    _frame_start("RESULTS")

    data = _current_run_data()
    if not data:
        st.write("No measurements yet. Go to COLLECT.")
        st.button("Close", on_click=_go, args=("solution",))
        _frame_end()
        return

    left, right = st.columns([0.52, 0.48], gap="medium")

    with left:
        block = _legacy_solution_screen_text(data)
        st.markdown(
            f"<div class='vxp-mono' style='height:560px; overflow:auto;'>{block}</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.pyplot(_plot_track_graph(data), clear_figure=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        st.button("Select Bal Meas", key="sel_bal_meas_sol", use_container_width=False)
        st.pyplot(_plot_polar(data), clear_figure=True)

    st.button("Close", on_click=_go, args=("solution",))
    _frame_end()


def screen_solution_text() -> None:
    _frame_start("SHOW SOLUTION")

    data = _current_run_data()
    report = _solution_report(data)
    st.markdown(f"<div class='vxp-mono' style='height:520px; overflow:auto;'>{report}</div>", unsafe_allow_html=True)

    st.button("Close", on_click=_go, args=("solution",))
    _frame_end()


# ----------------------------
# Session init
# ----------------------------

def _init_state() -> None:
    if "vxp_screen" not in st.session_state:
        st.session_state.vxp_screen = "home"

    # Default with RUN 1 + RUN 2 available (as in your request)
    st.session_state.setdefault("vxp_run", 1)
    st.session_state.setdefault("vxp_runs", {1: {}, 2: {}})
    st.session_state.setdefault("vxp_completed_by_run", {1: set(), 2: set()})

    st.session_state.setdefault("vxp_aircraft", {"weight": 0.0, "cg": 0.0, "hours": 0.0, "initials": ""})
    st.session_state.setdefault("vxp_note_codes", {1})  # start with Balance

    st.session_state.setdefault("vxp_adjustments", _default_adjustments())

    # Per-run completion tracking (regimes collected)
    st.session_state.setdefault("vxp_pending_regime", None)
    st.session_state.setdefault("vxp_acq_in_progress", False)


# ----------------------------
# Main app
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Vibrex VXP Simulator – BO105", layout="wide")
    _init_state()

    # Layout: toolbar + main window
    # Keep the UI closer to the legacy 4:3 look (more "square")
    tcol, maincol = st.columns([0.14, 0.86], gap="small")

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
        elif scr == "next_run":
            screen_next_run()
        else:
            st.session_state.vxp_screen = "home"
            st.rerun()


if __name__ == "__main__":
    main()
