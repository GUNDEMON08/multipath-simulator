"""Interactive Wireless Multipath Signal Propagation and Degradation Simulator.

Run locally:   streamlit run app.py
All physics lives in the other modules; this file only builds the interface.

State model: every control is a Streamlit widget whose value lives in
``st.session_state`` under a fixed key.  Presets simply write those keys (inside
callbacks, before the widgets are created), and ``get_tx / get_model / get_scene /
get_paths`` rebuild the dataclasses from the keys on every run.
"""
from __future__ import annotations

import html
import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import experiments as ex
import propagation_model as pm
import report_export as rep
import signal_analysis as sa
import signal_generator as sg
import visualizations as viz

st.set_page_config(page_title="Multipath Signal Propagation Simulator", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- compatibility helpers
def _wide(fn) -> dict:
    """Full-width keyword that works on both older and newer Streamlit releases."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params and "stretch" in str(params["width"].annotation):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


PLOT_W, DF_W, BTN_W, DL_W = _wide(st.plotly_chart), _wide(st.dataframe), _wide(st.button), _wide(st.download_button)


def show_plot(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "responsive": True}, **PLOT_W)


# --------------------------------------------------------------------------- labels
LBL_SIGNAL = {"sine": "Sine wave (CW)", "bpsk": "Digital (BPSK data)"}
LBL_REF = {"transmitted": "Transmitted signal", "los": "Direct path only (LOS)"}
LBL_MODE = {"simplified": "Simplified (educational)", "detailed": "Detailed (physical)"}
LBL_SRC = {"manual": "Manual length", "direct": "Scene: direct (LOS)", "ground": "Scene: ground bounce",
           "ceiling": "Scene: ceiling bounce", "wall": "Scene: wall bounce"}
LBL_KIND = {"direct": "Direct", "reflected": "Reflected"}
LBL_PAT = {"omni": "Omnidirectional", "directional": "Directional"}
LEVELS = ["Beginner", "Advanced"]
GROUPS = ["Transmit & paths", "Received signal", "Position fading", "Phase sweep",
          "Digital distortion", "Channel response"]
FOCUS_GROUP = {"signals": "Received signal", "position": "Position fading", "phase": "Phase sweep"}
inv = lambda d: {v: k for k, v in d.items()}   # noqa: E731

HELP = {
    "type": "Sine wave: one steady carrier. Digital: BPSK bits (the carrier flips sign for a 0), used to show inter-symbol interference.",
    "amp": "Peak amplitude A of the transmitted carrier in volts. It sets the reference amplitude.",
    "freq": "Carrier frequency f in MHz. Wavelength = c / f (300 MHz -> about 1 m). Phase differences between paths depend on f x delay.",
    "phase": "Initial phase of the transmitted carrier. It rotates every path equally, so it does not change the strength.",
    "dur": "Length of the simulated window in ns. It is rounded to whole carrier cycles (sine) or whole bit patterns (digital).",
    "spc": "Samples per carrier cycle. Higher = smoother curves but slower.",
    "noise": "Noise standard deviation as a fraction of the transmitted amplitude A (0 = off). Sets the SNR and the bit error rate.",
    "bits": "Repeating bit pattern for BPSK (only 0 and 1).",
    "cpb": "Carrier cycles per bit. Bit rate = f / cycles per bit.",
    "ref": "Reference for amplitude change, power ratio and dB: the transmitted signal, or only the direct (LOS) path.",
    "mode": "Simplified: the phase slider alone sets each path's carrier phase (delay only shifts the bit envelope). "
            "Detailed: the phase caused by the delay, -2 pi f tau, is applied automatically, plus the slider and reflection phase.",
    "spread": "Detailed option: scale each path's amplitude by (shortest path length / its length), a 1/distance spreading loss.",
    "rxpat": "Omnidirectional receives every direction equally; directional weights each path by its angle of arrival.",
    "rxang": "Direction the receive antenna points to (degrees from the +x axis).",
    "rxexp": "Directivity exponent n in gain = ((1 + cos(angle error))/2)^n. Larger n = narrower beam.",
    "en": "Switch the path on or off. A disabled path contributes nothing.",
    "src": "Manual: you type the length. Scene options: length and arrival angle come from the 2-D geometry (image method).",
    "kind": "Direct: line of sight. Reflected: the reflection coefficient applies.",
    "pamp": "Relative strength of the path (1 = same amplitude as transmitted).",
    "att": "Extra loss in dB. Amplitude factor = 10^(-dB/20): 6 dB halves the amplitude, 20 dB leaves 10 %.",
    "pph": "Phase shift you add to the path, 0-360 degrees. In simplified mode this is the path's whole carrier phase.",
    "len": "Path length in metres. Delay = length / c (about 3.34 ns per metre).",
    "xd": "Extra delay in ns on top of length / c (for example processing or a longer cable).",
    "ang": "Angle of arrival at the receiver (degrees from +x). It matters when the receive antenna is directional.",
    "gam": "Reflection coefficient. The magnitude is the reflection loss; a negative value means a 180-degree phase inversion.",
}


# --------------------------------------------------------------------------- state: write a configuration into widget keys
PATH_FIELDS = ("name", "en", "src", "kind", "amp", "att", "ph", "len", "xd", "ang", "gam")


def _set_path_keys(uid: int, p: pm.Path) -> None:
    ss = st.session_state
    ss[f"p{uid}_name"] = str(p.name)
    ss[f"p{uid}_en"] = bool(p.enabled)
    ss[f"p{uid}_src"] = LBL_SRC[p.source]
    ss[f"p{uid}_kind"] = LBL_KIND[p.kind]
    ss[f"p{uid}_amp"] = float(np.clip(p.amplitude, 0.0, 2.0))
    ss[f"p{uid}_att"] = float(np.clip(p.attenuation_db, 0.0, 60.0))
    ss[f"p{uid}_ph"] = float(p.phase_deg) % 360.0
    ss[f"p{uid}_len"] = float(np.clip(p.length_m, 0.01, 5000.0))
    ss[f"p{uid}_xd"] = float(np.clip(p.extra_delay_ns, 0.0, 200.0))
    ss[f"p{uid}_ang"] = float(np.clip(p.angle_deg, -180.0, 180.0))
    ss[f"p{uid}_gam"] = float(np.clip(p.reflection_coeff, -1.0, 1.0))


def apply_config(tx: sg.TxConfig, model: pm.ModelConfig, scene: pm.Scene, paths: list,
                 sweep_path: int = 1, sweep_x=(4.0, 18.0), focus: str = "signals") -> None:
    """Write a full configuration into session_state (call only from callbacks / before widgets exist)."""
    ss = st.session_state
    ss["tx_type"] = LBL_SIGNAL[tx.signal_type]
    ss["tx_amp"] = float(np.clip(tx.amplitude, 0.1, 5.0))
    ss["tx_freq"] = float(np.clip(tx.freq_mhz, 10.0, 3000.0))
    ss["tx_phase"] = float(tx.phase_deg) % 360.0
    ss["tx_dur"] = float(np.clip(tx.duration_ns, 5.0, 5000.0))
    ss["tx_spc"] = int(tx.samples_per_cycle)
    ss["tx_noise"] = float(np.clip(tx.noise_level, 0.0, 1.0))
    ss["tx_bits"] = str(tx.bits)
    ss["tx_cpb"] = int(np.clip(tx.cycles_per_bit, 2, 20))
    ss["tx_ref"] = LBL_REF[tx.reference_mode]
    ss["m_mode"] = LBL_MODE[model.mode]
    ss["m_spread"] = bool(model.spreading_loss)
    ss["m_rxpat"] = LBL_PAT[model.rx_pattern]
    ss["m_rxang"] = float(np.clip(model.rx_boresight_deg, -180.0, 180.0))
    ss["m_rxexp"] = float(np.clip(model.rx_pattern_exp, 0.5, 8.0))
    ss["sc_w"] = float(np.clip(scene.wall_x, 5.0, 40.0))
    ss["sc_h"] = float(np.clip(scene.ceiling_y, 3.0, 20.0))
    ss["sc_txx"], ss["sc_txy"] = float(scene.tx_x), float(scene.tx_y)
    ss["sc_rxx"], ss["sc_rxy"] = float(scene.rx_x), float(scene.rx_y)
    ss["sc_ground"], ss["sc_ceiling"], ss["sc_wall"] = bool(scene.ground), bool(scene.ceiling), bool(scene.wall)

    for old in list(ss.get("path_ids", [])):
        for f in PATH_FIELDS:
            ss.pop(f"p{old}_{f}", None)
    ids = []
    for p in paths[:pm.MAX_PATHS]:
        ss["uid_counter"] = ss.get("uid_counter", 0) + 1
        uid = ss["uid_counter"]
        _set_path_keys(uid, p)
        ids.append(uid)
    ss["path_ids"] = ids
    ss["sweep_uid"] = ids[min(max(sweep_path, 0), len(ids) - 1)]
    ss["sweep_range"] = (float(sweep_x[0]), float(sweep_x[1]))
    ss["sig_group"] = FOCUS_GROUP.get(focus, "Received signal")


def apply_experiment(key: str) -> None:
    e = ex.get_experiment(key)
    apply_config(e.tx, e.model, e.scene, e.paths, e.sweep_path, e.sweep_x, e.focus)


def on_preset_change() -> None:
    apply_experiment(st.session_state["exp_key"])


def on_reset_experiment() -> None:
    apply_experiment(st.session_state["exp_key"])


def on_restore_defaults() -> None:
    st.session_state["exp_key"] = "H"
    st.session_state["ui_level"] = "Beginner"
    apply_experiment("H")


def on_add_path() -> None:
    ss = st.session_state
    ids = ss["path_ids"]
    if len(ids) >= pm.MAX_PATHS:
        return
    ss["uid_counter"] += 1
    uid = ss["uid_counter"]
    last_len = ss[f"p{ids[-1]}_len"] if ids else 5.0
    _set_path_keys(uid, pm.Path(name=f"Path {len(ids) + 1}", kind="reflected", amplitude=0.5, attenuation_db=3.0,
                                length_m=float(last_len) + 1.3, reflection_coeff=0.8, angle_deg=30.0))
    ids.append(uid)


def on_remove_path(uid: int) -> None:
    ss = st.session_state
    if len(ss["path_ids"]) <= 1:
        return
    ss["path_ids"].remove(uid)
    for f in PATH_FIELDS:
        ss.pop(f"p{uid}_{f}", None)
    if ss.get("sweep_uid") == uid:
        ss["sweep_uid"] = ss["path_ids"][-1]


def init_state() -> None:
    if "path_ids" not in st.session_state:
        st.session_state["exp_key"] = "H"
        st.session_state["ui_level"] = "Beginner"
        st.session_state["snapshots"] = []
        apply_experiment("H")


# --------------------------------------------------------------------------- state: read widgets -> dataclasses
def get_tx() -> sg.TxConfig:
    ss = st.session_state
    return sg.TxConfig(signal_type=inv(LBL_SIGNAL)[ss["tx_type"]], amplitude=ss["tx_amp"], freq_mhz=ss["tx_freq"],
                       phase_deg=ss["tx_phase"], duration_ns=ss["tx_dur"], samples_per_cycle=int(ss["tx_spc"]),
                       noise_level=ss["tx_noise"], bits=ss["tx_bits"], cycles_per_bit=int(ss["tx_cpb"]),
                       reference_mode=inv(LBL_REF)[ss["tx_ref"]])


def get_model() -> pm.ModelConfig:
    ss = st.session_state
    return pm.ModelConfig(mode=inv(LBL_MODE)[ss["m_mode"]], spreading_loss=ss["m_spread"],
                          rx_pattern=inv(LBL_PAT)[ss["m_rxpat"]], rx_boresight_deg=ss["m_rxang"],
                          rx_pattern_exp=ss["m_rxexp"])


def get_scene() -> pm.Scene:
    ss = st.session_state
    return pm.validate_scene(pm.Scene(wall_x=ss["sc_w"], ceiling_y=ss["sc_h"], tx_x=ss["sc_txx"], tx_y=ss["sc_txy"],
                                      rx_x=ss["sc_rxx"], rx_y=ss["sc_rxy"], ground=ss["sc_ground"],
                                      ceiling=ss["sc_ceiling"], wall=ss["sc_wall"]))


def get_paths() -> list:
    ss = st.session_state
    out = []
    for uid in ss["path_ids"]:
        out.append(pm.Path(name=ss[f"p{uid}_name"].strip() or f"Path {uid}", enabled=ss[f"p{uid}_en"],
                           kind=inv(LBL_KIND)[ss[f"p{uid}_kind"]], source=inv(LBL_SRC)[ss[f"p{uid}_src"]],
                           amplitude=ss[f"p{uid}_amp"], attenuation_db=ss[f"p{uid}_att"], phase_deg=ss[f"p{uid}_ph"],
                           length_m=ss[f"p{uid}_len"], extra_delay_ns=ss[f"p{uid}_xd"], angle_deg=ss[f"p{uid}_ang"],
                           reflection_coeff=ss[f"p{uid}_gam"]))
    return out


def config_dict(tx, model, scene, paths) -> dict:
    return {"tx": tx.to_dict(), "model": model.to_dict(), "scene": scene.to_dict(),
            "paths": [p.to_dict() for p in paths]}


def config_objects(cfg: dict):
    return (sg.TxConfig(**cfg["tx"]), pm.ModelConfig(**cfg["model"]), pm.Scene(**cfg["scene"]),
            [pm.Path(**d) for d in cfg["paths"]])


def run_config(tx, model, scene, paths):
    sim = pm.simulate(tx, paths, model, scene)
    return sim, sa.compute_metrics(sim, sg.validate_tx(replace(tx)), model)


def on_save_snapshot() -> None:
    ss = st.session_state
    tx, model, scene, paths = get_tx(), get_model(), get_scene(), get_paths()
    sim, m = run_config(tx, model, scene, paths)
    name = (ss.get("snap_name") or "").strip() or f"Snapshot {len(ss['snapshots']) + 1}"
    row = rep.observation_row(f"{ex.get_experiment(ss['exp_key']).title} | {name}", sim, tx, model, m)
    ss["snapshots"].append({"name": name, "cfg": config_dict(tx, model, scene, paths), "row": row})
    ss["snap_name"] = ""


def on_load_snapshot() -> None:
    ss = st.session_state
    pick = ss.get("snap_pick")
    for s in ss["snapshots"]:
        if s["name"] == pick:
            tx, model, scene, paths = config_objects(s["cfg"])
            apply_config(tx, model, scene, paths)
            return


def on_clear_snapshots() -> None:
    st.session_state["snapshots"] = []


# --------------------------------------------------------------------------- styling
CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#0f2748 0%,#14365f 55%,#1b4d6b 100%); border:1px solid rgba(120,160,220,.25);
       border-radius: 16px; padding: 18px 22px; margin-bottom: 12px;}
.hero h1 {font-size: 1.55rem; margin: 0 0 4px 0; line-height: 1.2;}
.hero p {margin: 0; color: #c8d6ee; font-size: .95rem;}
.hero .badges span {display:inline-block; background: rgba(77,163,255,.15); border:1px solid rgba(77,163,255,.35);
       color:#bfe0ff; border-radius: 999px; padding: 1px 10px; margin: 8px 6px 0 0; font-size: .75rem;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 10px 0;}
.card {background: #121b2e; border: 1px solid rgba(120,150,200,.22); border-radius: 12px; padding: 10px 12px;}
.card .lbl {font-size: .72rem; color: #9fb1cf; text-transform: uppercase; letter-spacing: .04em;}
.card .val {font-size: 1.45rem; font-weight: 650; line-height: 1.25;}
.card .sub {font-size: .74rem; color: #8fa2c4;}
.card.good .val {color: #5ee6a8;} .card.warn .val {color: #ffcf66;} .card.bad .val {color: #ff7a7a;}
.interp {border-radius: 12px; padding: 12px 16px; border: 1px solid rgba(120,150,200,.25); background: #101a2c; margin-bottom: 10px;}
.interp.good {border-left: 5px solid #5ee6a8;} .interp.warn {border-left: 5px solid #ffcf66;}
.interp.bad {border-left: 5px solid #ff7a7a;} .interp.neutral {border-left: 5px solid #4da3ff;}
.interp h4 {margin: 0 0 6px 0; font-size: 1.05rem;} .interp ul {margin: 0; padding-left: 1.15rem;}
.interp li {font-size: .9rem; margin: 2px 0;}
.txline {background:#0e1729; border:1px dashed rgba(120,150,200,.35); border-radius:10px; padding:8px 12px;
         font-size:.86rem; color:#cfe0ff; margin: 6px 0 10px 0;}
.note {font-size:.8rem; color:#93a6c8;}
.stButton > button, .stDownloadButton > button {min-height: 44px; border-radius: 10px;}
div[data-testid="stTabs"] button[role="tab"] {padding: 10px 14px; font-size: .95rem;}
@media (max-width: 640px) {
  .hero h1 {font-size: 1.2rem;} .hero p {font-size: .85rem;}
  .card .val {font-size: 1.2rem;} .block-container {padding-left: .7rem; padding-right: .7rem;}
}
</style>
"""


def card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="card {tone}"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>')


def fnum(x, spec: str, none: str = "n/a") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return none
    if isinstance(x, float) and np.isinf(x):
        return "no noise" if x > 0 else "-inf"
    return format(x, spec)


def db_text(x: float) -> str:
    if np.isnan(x):
        return "n/a"
    return "<= -120 dB" if x <= sa.DB_FLOOR else f"{x:+.2f} dB"


# --------------------------------------------------------------------------- sections
def tx_summary_line(tx: sg.TxConfig, scene_text: str = "") -> str:
    lam = pm.C / tx.freq_hz
    kind = (f"BPSK {tx.bits} ({tx.cycles_per_bit} cycles/bit, {tx.freq_mhz / tx.cycles_per_bit:.1f} Mbit/s)"
            if tx.is_digital else "Sine wave")
    noise = f"noise sigma = {tx.noise_level:g} A" if tx.noise_level > 0 else "noise off"
    return (f"<b>Current transmitter:</b> {kind} &middot; A = {tx.amplitude:g} V &middot; f = {tx.freq_mhz:g} MHz "
            f"(&lambda; = {lam:.3f} m, T = {1e3 / tx.freq_mhz:.3f} ns) &middot; &phi;<sub>0</sub> = {tx.phase_deg:g}&deg; "
            f"&middot; window {tx.duration_ns:g} ns &middot; {tx.samples_per_cycle} samples/cycle &middot; {noise} "
            f"&middot; reference: {LBL_REF[tx.reference_mode].lower()}")


def tab_experiment() -> None:
    ss = st.session_state
    titles = ex.experiment_titles()
    c1, c2, c3 = st.columns([3, 1.2, 1.2])
    with c1:
        st.selectbox("Experiment preset", list(titles), format_func=lambda k: titles[k], key="exp_key",
                     on_change=on_preset_change,
                     help="Each preset configures the transmitter, paths and scene of the same mathematical model.")
    with c2:
        st.write("")
        st.button("↺ Reset experiment", on_click=on_reset_experiment, key="btn_reset",
                  help="Reload the selected preset's settings.", **BTN_W)
    with c3:
        st.write("")
        st.button("⟲ Restore defaults", on_click=on_restore_defaults, key="btn_defaults",
                  help="Return the whole app to its opening state.", **BTN_W)

    e = ex.get_experiment(ss["exp_key"])
    st.markdown(f"**Goal:** {e.goal}")
    st.markdown(f"**Try this:** {e.try_this}")
    if ss["ui_level"] == "Beginner":
        st.info(f"**Why it works:** {e.why}")
    st.caption(f"After a preset loads, open the **📈 Signals** tab - it jumps to the most relevant graph "
               f"(\"{FOCUS_GROUP.get(e.focus, 'Received signal')}\").")

    st.radio("Explanation level", LEVELS, horizontal=True, key="ui_level",
             help="Beginner: plain-language explanations. Advanced: equations, extra metrics and phase breakdowns.")

    st.subheader("Transmitter & simulation")
    a, b, c = st.columns(3)
    with a:
        st.radio("Signal type", list(LBL_SIGNAL.values()), key="tx_type", help=HELP["type"])
        st.slider("Amplitude A (V)", 0.1, 5.0, step=0.1, key="tx_amp", help=HELP["amp"])
        st.number_input("Frequency (MHz)", min_value=10.0, max_value=3000.0, step=10.0, key="tx_freq", help=HELP["freq"])
    with b:
        st.slider("Initial phase (deg)", 0.0, 360.0, step=1.0, key="tx_phase", help=HELP["phase"])
        st.number_input("Duration (ns)", min_value=5.0, max_value=5000.0, step=5.0, key="tx_dur", help=HELP["dur"])
        st.select_slider("Sampling (samples per cycle)", options=[16, 32, 64, 128], key="tx_spc", help=HELP["spc"])
    with c:
        st.slider("Noise level (sigma / A)", 0.0, 1.0, step=0.01, key="tx_noise", help=HELP["noise"])
        st.radio("Reference signal", list(LBL_REF.values()), key="tx_ref", help=HELP["ref"])
    d1, d2 = st.columns([2, 1])
    with d1:
        st.text_input("BPSK bit pattern (digital only)", key="tx_bits", max_chars=32, help=HELP["bits"])
    with d2:
        st.number_input("Carrier cycles per bit (digital only)", min_value=2, max_value=20, step=1, key="tx_cpb",
                        help=HELP["cpb"])

    with st.expander("Model options (simplified / detailed, loss, receive antenna)"):
        st.radio("Phase model", list(LBL_MODE.values()), key="m_mode", horizontal=True, help=HELP["mode"])
        st.checkbox("Include 1/distance spreading loss", key="m_spread", help=HELP["spread"])
        m1, m2, m3 = st.columns(3)
        with m1:
            st.selectbox("Receive antenna", list(LBL_PAT.values()), key="m_rxpat", help=HELP["rxpat"])
        with m2:
            st.slider("Antenna direction (deg)", -180.0, 180.0, step=1.0, key="m_rxang", help=HELP["rxang"])
        with m3:
            st.slider("Directivity exponent n", 0.5, 8.0, step=0.5, key="m_rxexp", help=HELP["rxexp"])


def tab_scene_paths() -> None:
    ss = st.session_state
    st.markdown("Move the transmitter, the receiver or the room and every scene-linked path length, delay and phase "
                "is recalculated. Paths set to *Manual length* are not drawn.")
    with st.expander("📍 Scene geometry controls", expanded=True):
        g1, g2 = st.columns(2)
        with g1:
            st.slider("Receiver x (m)", 0.1, 40.0, step=0.05, key="sc_rxx", help="Move the receiver horizontally.")
            st.slider("Receiver y (m)", 0.1, 20.0, step=0.05, key="sc_rxy", help="Receiver height.")
        with g2:
            st.slider("Transmitter x (m)", 0.1, 40.0, step=0.05, key="sc_txx", help="Move the transmitter horizontally.")
            st.slider("Transmitter y (m)", 0.1, 20.0, step=0.05, key="sc_txy", help="Transmitter height.")
        g3, g4, g5 = st.columns(3)
        with g3:
            st.number_input("Room width / wall position (m)", min_value=5.0, max_value=40.0, step=1.0, key="sc_w")
        with g4:
            st.number_input("Room height / ceiling (m)", min_value=3.0, max_value=20.0, step=1.0, key="sc_h")
        with g5:
            st.checkbox("Ground reflector", key="sc_ground")
            st.checkbox("Ceiling reflector", key="sc_ceiling")
            st.checkbox("Right-wall reflector", key="sc_wall")
        st.caption("Positions are kept inside the room. Reflections use the image method (single bounce, flat mirror).")

    scene = get_scene()
    # scene-linked lengths are shown (and locked) in the path editor
    for uid in ss["path_ids"]:
        src = inv(LBL_SRC)[ss[f"p{uid}_src"]]
        if src != "manual":
            length, angle, _, _ = pm.resolve_geometry(src, scene)
            if length is not None:
                ss[f"p{uid}_len"] = float(np.clip(length, 0.01, 5000.0))
                ss[f"p{uid}_ang"] = float(np.clip(angle, -180.0, 180.0))
    scene_slot = st.container()

    st.subheader("Paths")
    tx = get_tx()
    lam = pm.C / tx.freq_hz
    for pos, uid in enumerate(list(ss["path_ids"])):
        label = f"P{pos + 1} · {ss[f'p{uid}_name']}" + ("" if ss[f"p{uid}_en"] else "  (disabled)")
        with st.expander(label, expanded=(pos < 2)):
            r1, r2, r3 = st.columns([1, 2, 2])
            with r1:
                st.checkbox("Enabled", key=f"p{uid}_en", help=HELP["en"])
            with r2:
                st.text_input("Name", key=f"p{uid}_name", max_chars=24)
            with r3:
                st.selectbox("Length source", list(LBL_SRC.values()), key=f"p{uid}_src", help=HELP["src"])
            linked = inv(LBL_SRC)[ss[f"p{uid}_src"]] != "manual"
            c1, c2 = st.columns(2)
            with c1:
                st.slider("Strength (relative)", 0.0, 2.0, step=0.01, key=f"p{uid}_amp", help=HELP["pamp"])
                st.slider("Attenuation (dB)", 0.0, 60.0, step=0.5, key=f"p{uid}_att", help=HELP["att"])
                st.slider("Phase shift (deg)", 0.0, 360.0, step=1.0, key=f"p{uid}_ph", help=HELP["pph"])
                st.slider("Reflection coefficient", -1.0, 1.0, step=0.05, key=f"p{uid}_gam", help=HELP["gam"],
                          disabled=(ss[f"p{uid}_kind"] == LBL_KIND["direct"] and not linked))
            with c2:
                st.number_input("Path length (m)", min_value=0.01, max_value=5000.0, step=0.05, format="%.3f",
                                key=f"p{uid}_len", help=HELP["len"], disabled=linked)
                st.number_input("Extra delay (ns)", min_value=0.0, max_value=200.0, step=0.5, key=f"p{uid}_xd",
                                help=HELP["xd"])
                st.slider("Angle of arrival (deg)", -180.0, 180.0, step=1.0, key=f"p{uid}_ang", help=HELP["ang"],
                          disabled=linked)
                st.selectbox("Path type", list(LBL_KIND.values()), key=f"p{uid}_kind", help=HELP["kind"], disabled=linked)
            delay_ns = ss[f"p{uid}_len"] / pm.C * 1e9 + ss[f"p{uid}_xd"]
            st.caption(f"Length {ss[f'p{uid}_len']:.3f} m = {ss[f'p{uid}_len'] / lam:.2f} wavelengths  ·  "
                       f"delay {delay_ns:.2f} ns = {delay_ns * 1e-9 * tx.freq_hz:.2f} carrier cycles")
            st.button("🗑 Remove this path", key=f"rm_{uid}", on_click=on_remove_path, args=(uid,),
                      disabled=len(ss["path_ids"]) <= 1, **BTN_W)

    st.button("➕ Add path", key="btn_add", on_click=on_add_path, disabled=len(ss["path_ids"]) >= pm.MAX_PATHS,
              help=f"Up to {pm.MAX_PATHS} paths.", **BTN_W)
    return scene_slot


def render_top(m: dict, sim: pm.SimResult, tx: sg.TxConfig, advanced: bool) -> None:
    tone_amp = "good" if m["amp_change_pct"] > 2 else ("bad" if m["amp_change_pct"] < -50 else ("warn" if m["amp_change_pct"] < -2 else ""))
    cards = [
        card("Received peak", f"{m['rx_amp']:.3f} V", f"reference {m['ref_amp']:.3f} V"),
        card("Amplitude change", f"{fnum(m['amp_change_pct'], '+.1f')} %", "vs reference", tone_amp if not np.isnan(m["amp_change_pct"]) else ""),
        card("Power change", db_text(m["power_db"]), f"ratio {fnum(m['power_ratio'], '.3f')}", tone_amp if not np.isnan(m["amp_change_pct"]) else ""),
        card("Received RMS", f"{m['rx_rms']:.3f} V", f"reference {m['ref_rms']:.3f} V"),
        card("Active paths", f"{m['n_active']} / {m['n_total']}", f"excess delay {m['delay']['excess_ns']:.1f} ns"),
        card("Distortion", f"{m['distortion'] * 100:.1f} %", "waveform shape error",
             "warn" if m["distortion"] > 0.05 else ""),
    ]
    if m["noise_on"]:
        cards.append(card("SNR", f"{fnum(m['snr_db'], '.1f')} dB", f"noise sigma {sim.sigma:.3f} V"))
    if m["is_digital"]:
        cards.append(card("Eye margin", fnum(m["margin"], ".2f"), "1.0 = ideal",
                          "bad" if (not np.isnan(m["margin"]) and m["margin"] <= 0) else ("warn" if m["margin"] < 0.5 else "good")))
        cards.append(card("BER (estimate)", fnum(m["ber_estimate"], ".2e"),
                          f"measured {m['bit_errors']}/{m['n_bits']} bits"))
    if advanced:
        cards.append(card("Reference power", f"{m['ref_power']:.4f} W", "V_rms² into 1 Ω"))
        cards.append(card("Received power", f"{m['rx_power']:.4f} W", "V_rms² into 1 Ω"))
    st.markdown('<div class="cards">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    items = "".join(f"<li>{html.escape(p)}</li>" for p in m["points"])
    st.markdown(f'<div class="interp {m["tone"]}"><h4>{html.escape(m["label"])}</h4><ul>{items}</ul></div>',
                unsafe_allow_html=True)
    for w in sim.warnings:
        st.warning(w)
    b1, b2 = st.columns([2, 3])
    with b1:
        st.text_input("Snapshot name", key="snap_name", placeholder="e.g. 2-path, 90° apart", label_visibility="collapsed")
    with b2:
        st.button("📸 Save snapshot", key="btn_snap", on_click=on_save_snapshot,
                  help="Store the current configuration and its results for the comparison and observation table.", **BTN_W)


def tab_signals(sim, tx, model, scene, paths, m, terms) -> None:
    ss = st.session_state
    st.radio("Plot group", GROUPS, horizontal=True, key="sig_group", label_visibility="collapsed")
    group = ss["sig_group"]
    if group == "Transmit & paths":
        show_plot(viz.fig_transmitted(sim, tx), "f1")
        show_plot(viz.fig_paths(sim, tx), "f2")
    elif group == "Received signal":
        show_plot(viz.fig_received(sim, tx), "f3")
        show_plot(viz.fig_tx_vs_rx(sim, tx, sim.ref_label), "f4")
        show_plot(viz.fig_envelope_power(sim, tx), "f5")
    elif group == "Position fading":
        lo_max = float(ss["sc_w"])
        lo, hi = ss["sweep_range"]
        st.slider("Receiver sweep range (m)", 0.1, 40.0, step=0.1, key="sweep_range",
                  help="Receiver x-positions to scan. Scene-linked paths change length with position.")
        lo, hi = ss["sweep_range"]
        hi = min(hi, scene.wall_x - 0.05)
        lo = min(lo, hi - 0.1)
        xs = np.linspace(lo, hi, 700)
        sweep = pm.sweep_receiver(paths, tx, model, scene, xs)
        cur = abs(pm.carrier_phasor(tx, terms)) / tx.amplitude
        show_plot(viz.fig_position(sweep, scene.rx_x, cur), "f6")
        linked = any(p.source != "manual" for p in paths)
        if not linked:
            st.info("No path is linked to the scene, so the strength does not depend on the receiver position. "
                    "Set a path's *Length source* to a scene option in the Scene & paths tab.")
        elif model.mode == "simplified":
            st.info("Simplified mode uses the phase sliders instead of the geometric phase, so movement changes only "
                    "amplitudes. Choose the Detailed model to see fading.")
        else:
            depth = 20 * np.log10(max(sweep["combined"].max(), 1e-9) / max(sweep["combined"].min(), 1e-9))
            st.caption(f"Over this range the multipath strength varies by {depth:.1f} dB "
                       f"(narrow-band carrier, steady state). One fade every few tenths of a wavelength "
                       f"(λ = {pm.C / tx.freq_hz:.2f} m).")
    elif group == "Phase sweep":
        opts = list(ss["path_ids"])
        if ss.get("sweep_uid") not in opts:
            ss["sweep_uid"] = opts[-1]
        st.selectbox("Path whose phase is swept", opts, key="sweep_uid",
                     format_func=lambda u: f"{ss[f'p{u}_name']}", help="The other paths stay as configured.")
        idx = opts.index(ss["sweep_uid"])
        phases = np.linspace(0, 360, 361)
        mags = pm.sweep_phase(paths, tx, model, scene, idx, phases)
        cur = abs(pm.carrier_phasor(tx, terms)) / tx.amplitude
        show_plot(viz.fig_phase(phases, mags, paths[idx].phase_deg, cur, paths[idx].name), "f7")
        st.caption(f"Range over a full phase turn: {mags.min():.3f} A (worst) to {mags.max():.3f} A (best). "
                   "Equal amplitudes would reach zero.")
    elif group == "Digital distortion":
        if not tx.is_digital:
            st.info("Select the **Digital (BPSK data)** signal type in the Experiment tab (or load Experiment E) to see "
                    "bit distortion and the eye diagram.")
        else:
            show_plot(viz.fig_digital(sim, tx), "f8a")
            show_plot(viz.fig_eye(sim, tx), "f8b")
            st.caption(f"Bit duration {m['bit_ns']:.2f} ns; delay spread {m['delay']['excess_ns']:.2f} ns "
                       f"({m['delay']['excess_ns'] / m['bit_ns']:.2f} bits). Eye margin {fnum(m['margin'], '.2f')} "
                       f"(1 = ideal, ≤ 0 = closed). BER measured {m['bit_errors']}/{m['n_bits']} bits, "
                       f"analytic estimate {fnum(m['ber_estimate'], '.2e')}.")
    else:
        show_plot(viz.fig_frequency(terms, tx, model), "f9")
        show_plot(viz.fig_impulse(terms), "f10")
        st.caption(f"Nulls in |H(f)| are spaced about 1/Δτ apart (Δτ = delay difference). RMS delay spread "
                   f"{m['delay']['rms_ns']:.2f} ns.")


def tab_analysis(sim, tx, model, m, advanced: bool) -> None:
    st.subheader("Path list and phase bookkeeping")
    df = pd.DataFrame(sa.path_rows(sim, tx, model))
    cols = ["Path", "Type", "Active", "Length (m)", "Delay (ns)", "Final amplitude a_i", "Phase applied (deg)",
            "Phase vs first path (deg)"]
    if advanced:
        cols = list(df.columns)
    st.dataframe(df[cols], hide_index=True, **DF_W)
    st.caption("Phase applied = user phase + reflection phase" + (" + propagation phase (detailed mode)."
               if model.mode == "detailed" else ". Propagation phase is not applied in simplified mode."))

    st.subheader("Why does this result occur?")
    for line in sa.explanations(sim, tx, model, m, advanced):
        st.markdown(f"- {line}")

    st.subheader("Numerical results")
    res = {
        "Reference": m["ref_label"],
        "Reference amplitude (V)": fnum(m["ref_amp"], ".5f"), "Received peak amplitude (V)": fnum(m["rx_amp"], ".5f"),
        "Received RMS amplitude (V)": fnum(m["rx_rms"], ".5f"), "Amplitude change (%)": fnum(m["amp_change_pct"], "+.3f"),
        "Reference power (V², 1 Ω)": fnum(m["ref_power"], ".5g"), "Received power (V², 1 Ω)": fnum(m["rx_power"], ".5g"),
        "Power ratio": fnum(m["power_ratio"], ".5g"), "Power change (dB)": db_text(m["power_db"]),
        "Active paths": f"{m['n_active']} of {m['n_total']}",
        "First arrival / last arrival (ns)": f"{m['delay']['first_ns']:.2f} / {m['delay']['last_ns']:.2f}",
        "Excess delay (ns)": f"{m['delay']['excess_ns']:.3f} ({m['excess_cycles']:.2f} carrier cycles)",
        "RMS delay spread (ns)": f"{m['delay']['rms_ns']:.3f}",
        "SNR (dB)": fnum(m["snr_db"], ".2f") if m["noise_on"] else "noise off",
        "Waveform distortion (shape error)": f"{m['distortion']:.4f}",
    }
    if m["is_digital"]:
        res["Eye margin (1 = ideal)"] = fnum(m["margin"], ".3f")
        res["BER measured"] = f"{m['bit_errors']} errors / {m['n_bits']} bits"
        res["BER analytic estimate"] = fnum(m["ber_estimate"], ".3e")
    st.dataframe(pd.DataFrame({"Quantity": list(res), "Value": [str(v) for v in res.values()]}),
                 hide_index=True, **DF_W)
    st.caption("Simulated results from the mathematical model - not laboratory measurements. "
               "Power is the normalised V_rms² (power into 1 Ω), a relative quantity.")


def tab_compare(exp_title, sim, tx, model, scene, paths, m) -> None:
    ss = st.session_state
    snaps = ss["snapshots"]
    live_row = rep.observation_row(f"{exp_title} | live", sim, tx, model, m)

    st.subheader("Observation table")
    st.caption("Every saved snapshot adds a row. Labelled as simulated observations, not physical measurements.")
    rows = [s["row"] for s in snaps] + [live_row]
    obs = pd.DataFrame(rows)
    st.dataframe(obs, hide_index=True, **DF_W)

    st.subheader("Snapshots and side-by-side comparison")
    if snaps:
        names = [s["name"] for s in snaps]
        cc1, cc2, cc3 = st.columns([3, 1.2, 1.2])
        with cc1:
            st.selectbox("Snapshot", names, key="snap_pick")
        with cc2:
            st.write("")
            st.button("↩ Load snapshot", on_click=on_load_snapshot, key="btn_loadsnap", **BTN_W)
        with cc3:
            st.write("")
            st.button("🗑 Clear all", on_click=on_clear_snapshots, key="btn_clearsnap", **BTN_W)
        options = ["Live (current settings)"] + names
        k1, k2 = st.columns(2)
        with k1:
            pick_a = st.selectbox("Compare A", options, index=0, key="cmp_a")
        with k2:
            pick_b = st.selectbox("Compare B", options, index=min(1, len(options) - 1), key="cmp_b")

        def build(name):
            if name == options[0]:
                return sim, m, name
            cfg = next(s["cfg"] for s in snaps if s["name"] == name)
            t_, mo_, sc_, pa_ = config_objects(cfg)
            s_, m_ = run_config(t_, mo_, sc_, pa_)
            return s_, m_, name

        sa_, ma_, na_ = build(pick_a)
        sb_, mb_, nb_ = build(pick_b)
        keys = [("Active paths", "n_active", "d"), ("Received peak (V)", "rx_amp", ".4f"),
                ("Received RMS (V)", "rx_rms", ".4f"), ("Amplitude change (%)", "amp_change_pct", "+.2f"),
                ("Power ratio", "power_ratio", ".4f"), ("Power change (dB)", "power_db", "+.3f"),
                ("Distortion", "distortion", ".4f"), ("Excess delay (ns)", None, ".2f")]
        table = []
        for label, key, spec in keys:
            if key is None:
                va, vb = ma_["delay"]["excess_ns"], mb_["delay"]["excess_ns"]
            else:
                va, vb = ma_[key], mb_[key]
            diff = (vb - va) if not (np.isnan(float(va)) or np.isnan(float(vb))) else float("nan")
            table.append({"Quantity": label, na_: fnum(va, spec), nb_: fnum(vb, spec),
                          "B - A": fnum(diff, "+.4g")})
        st.dataframe(pd.DataFrame(table), hide_index=True, **DF_W)
        show_plot(viz.fig_compare(sa_, sb_, na_, nb_), "fcmp")
        st.write(f"**A:** {ma_['label']}  \n**B:** {mb_['label']}")
    else:
        st.info("Save a snapshot with the 📸 button above the tabs. Change a parameter, save another, then compare them here.")

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇ Signal samples (CSV)", rep.samples_csv(sim, tx), file_name="multipath_samples.csv",
                           mime="text/csv", key="dl_samples", **DL_W)
        st.download_button("⬇ Observation table (CSV)", rep.observations_csv(rows), file_name="multipath_observations.csv",
                           mime="text/csv", key="dl_obs", **DL_W)
    with d2:
        st.download_button("⬇ Experiment summary (Markdown)", rep.summary_markdown(exp_title, sim, tx, model, m),
                           file_name="multipath_summary.md", mime="text/markdown", key="dl_sum", **DL_W)
        st.download_button("⬇ Configuration (JSON)", rep.config_json(tx, paths, model, scene, exp_title),
                           file_name="multipath_config.json", mime="application/json", key="dl_cfg", **DL_W)


GUIDE = r"""
### What is multipath?
A radio wave rarely travels in a straight line only. It also bounces off the ground, walls and ceilings, so the receiver
gets several **copies** of the signal. Each copy is **delayed** (longer path), **weaker** (loss) and **phase-shifted**.
The receiver sees their sum, which can be stronger, weaker or distorted.

### How to use the simulator
1. Pick an experiment (A-H) in **🧪 Experiment**, or start from the custom one (H).
2. Change phase, strength, attenuation, length, delay or the number of paths in **📡 Scene & paths**.
3. Watch the numbers above the tabs and the graphs in **📈 Signals**. Save snapshots and compare them in **🔁 Compare & export**.

### Signal model
For a sinusoidal carrier of amplitude $A$, frequency $f$ and initial phase $\varphi_0$:
"""

GUIDE2 = r"""
Meaning of the terms (each phase is applied **once**):
- $a_i = s_i\,10^{-L_{dB,i}/20}\,|\Gamma_i|\,g_{rx}(\theta_i)$ (optionally $\times L_{min}/L_i$ for spreading loss).
- $\varphi_{user,i}$ is the phase slider. $\varphi_{refl,i}=180^\circ$ if the reflection coefficient is negative.
- The **propagation phase** $-2\pi f\tau_i$ is not a separate term: it comes from evaluating the carrier at $t-\tau_i$ (detailed mode).
  In *simplified* mode the carrier is not delayed - the slider alone sets the phase - so nothing is counted twice.

Metrics:
"""

GUIDE3 = r"""
### Assumptions and limitations
- 2-D side view, perfectly flat reflecting surfaces, **single-bounce** reflections by the image method; the receiver and transmitter are points.
- Reflection coefficients are real numbers (no angle- or polarisation-dependent Fresnel behaviour); no diffraction, scattering, absorption by obstacles or Doppler shift; the channel is static.
- Power is the **normalised** $V_{rms}^2$ (into 1 Ω): a *relative* measure, not antenna power in watts.
- Position and phase sweeps use the steady-state carrier (phasor) result, which matches the time-domain waveform for a sine wave.
- BPSK detection assumes ideal carrier recovery and bit timing locked to the first-arriving path; BER from 8-32 bits is statistically rough, so an analytic estimate is shown too.
- This is an educational model, **not** a full electromagnetic solver. All outputs are *simulated observations*.

### Suggested 5-minute demo
A (reference) → B (add a path in phase: stronger) → C (opposite phase: cancellation) → D (move the receiver: fading) →
E (digital echo: eye closes) → G (phase sweep) → H (build your own and compare snapshots).

### Glossary
**Constructive / destructive interference** - waves adding or cancelling. **Fading** - signal strength varying with position or frequency.
**Delay spread** - time between the first and last arriving copy. **ISI** - overlap of neighbouring bits caused by delayed copies.
"""


def tab_guide() -> None:
    st.markdown(GUIDE)
    st.latex(r"y(t)=\sum_i A\,a_i\cos\!\big(2\pi f\,(t-\tau_i)+\varphi_0+\varphi_{user,i}+\varphi_{refl,i}\big),\qquad \tau_i=\frac{L_i}{c}")
    st.markdown(GUIDE2)
    st.latex(r"\Delta A(\%)=100\,\frac{A_{rx}-A_{ref}}{A_{ref}},\quad "
             r"\text{Power ratio}=\left(\frac{V_{rms,rx}}{V_{rms,ref}}\right)^{2},\quad "
             r"\Delta P(\mathrm{dB})=10\log_{10}(\text{Power ratio})")
    st.latex(r"\text{two paths: } A_{rx}=A\sqrt{a_1^2+a_2^2+2a_1a_2\cos\Delta\varphi},\qquad "
             r"\Delta\varphi=2\pi f\,(\tau_2-\tau_1)+\ldots")
    st.markdown(GUIDE3)


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>📡 Interactive Wireless Multipath Signal Propagation and Degradation Simulator</h1>'
        "<p>A wireless signal reaches the receiver along several paths. Change path length, delay, attenuation, phase and "
        "reflections and see when the received signal becomes stronger, weaker or distorted.</p>"
        '<div class="badges"><span>Signal model</span><span>2-D propagation scene</span><span>Live metrics</span>'
        "<span>Experiments A-H</span><span>Simulated observations</span></div></div>", unsafe_allow_html=True)

    top = st.container()
    t_exp, t_scene, t_sig, t_ana, t_cmp, t_guide = st.tabs(
        ["🧪 Experiment", "📡 Scene & paths", "📈 Signals", "📊 Analysis", "🔁 Compare & export", "📘 Guide & model"])

    with t_exp:
        tab_experiment()
    with t_scene:
        scene_slot = tab_scene_paths()

    tx, model, scene, paths = get_tx(), get_model(), get_scene(), get_paths()
    try:
        sim, m = run_config(tx, model, scene, paths)
    except Exception as err:   # never show a stack trace during a presentation
        st.error(f"The simulation could not be run with these settings: {err}")
        st.stop()
    tx_valid = sg.validate_tx(replace(tx))
    terms = sim.terms
    advanced = st.session_state["ui_level"] == "Advanced"
    exp_title = ex.get_experiment(st.session_state["exp_key"]).title

    with top:
        st.markdown(f'<div class="txline">{tx_summary_line(tx_valid)}</div>', unsafe_allow_html=True)
        render_top(m, sim, tx_valid, advanced)

    with scene_slot:
        show_plot(viz.fig_scene(scene, terms), "fscene")
        manual = [t.name for t in terms if t.active and t.source == "manual"]
        if manual:
            st.caption("Not drawn (manual length): " + ", ".join(manual))

    with t_sig:
        tab_signals(sim, tx_valid, model, scene, paths, m, terms)
    with t_ana:
        tab_analysis(sim, tx_valid, model, m, advanced)
    with t_cmp:
        tab_compare(exp_title, sim, tx_valid, model, scene, paths, m)
    with t_guide:
        tab_guide()

    st.markdown('<p class="note">Results are simulated observations from a simplified multipath model - '
                'not physical laboratory measurements.</p>', unsafe_allow_html=True)


main()
