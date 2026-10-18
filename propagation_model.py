"""Multipath propagation model.

Signal model (per path i, sinusoidal carrier)
---------------------------------------------
    y_i(t) = A * a_i * cos( 2*pi*f*(t - tau_i) + phi0 + phi_user_i + phi_refl_i )     [detailed mode]
    y(t)   = sum_i y_i(t)

    a_i     = rel_i * 10^(-att_i/20) * |Gamma_i| * spread_i * g_rx(theta_i)
    tau_i   = L_i / c + extra_delay_i
    phi_refl_i = 180 deg when the reflection coefficient Gamma_i < 0, else 0

Phase bookkeeping (no double counting)
--------------------------------------
Three different phase contributions exist and each is applied exactly once:

1. ``phase_user``  -- the phase-shift slider (0-360 deg).  Models any additional
   phase the presenter wants to dial in (antenna, filter, ...).
2. ``phase_prop``  -- phase accumulated by propagation, -2*pi*f*tau_i.  It is NOT
   a separate term in the code: it appears automatically because the carrier is
   evaluated at (t - tau_i).  It is reported separately for information only.
3. ``phase_refl``  -- 180 deg inversion for a negative reflection coefficient.

In **simplified (educational) mode** the carrier is evaluated at t instead of
(t - tau_i): the propagation phase is replaced by the phase slider so a student
can set "path 2 is 180 deg out of phase" directly.  The delay then only shifts
the digital bit envelope (inter-symbol interference).  In **detailed mode** the
propagation phase is applied from the delay, as in a real channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
import numpy as np

import signal_generator as sg

C = 299_792_458.0                       # speed of light [m/s]
MAX_PATHS = 10
SURFACE_SOURCES = ("ground", "ceiling", "wall")
SOURCES = ("manual", "direct") + SURFACE_SOURCES


# --------------------------------------------------------------------------- data classes
@dataclass
class Path:
    """One propagation path."""
    name: str = "Path 1"
    enabled: bool = True
    kind: str = "direct"             # "direct" | "reflected" (used for manual paths)
    source: str = "manual"           # "manual" or a scene-geometry source
    amplitude: float = 1.0           # relative strength (1 = same as transmitted)
    attenuation_db: float = 0.0      # extra loss [dB]
    phase_deg: float = 0.0           # user-selected phase shift [deg]
    length_m: float = 3.0            # path length [m] (manual paths)
    extra_delay_ns: float = 0.0      # additional non-geometric delay [ns]
    angle_deg: float = 0.0           # angle of arrival (manual paths) [deg]
    reflection_coeff: float = 1.0    # signed real coefficient Gamma in [-1, 1]

    def to_dict(self) -> dict:
        return asdict(self)


def validate_path(p: Path) -> Path:
    """Return a copy with every field clamped to a legal range."""
    p = replace(p)
    p.kind = p.kind if p.kind in ("direct", "reflected") else "direct"
    p.source = p.source if p.source in SOURCES else "manual"
    p.amplitude = float(np.clip(p.amplitude, 0.0, 10.0))
    p.attenuation_db = float(np.clip(p.attenuation_db, 0.0, 120.0))
    p.phase_deg = float(p.phase_deg) % 360.0
    p.length_m = float(np.clip(p.length_m, 0.01, 10_000.0))
    p.extra_delay_ns = float(np.clip(p.extra_delay_ns, 0.0, 10_000.0))
    p.angle_deg = ((float(p.angle_deg) + 180.0) % 360.0) - 180.0
    p.reflection_coeff = float(np.clip(p.reflection_coeff, -1.0, 1.0))
    return p


@dataclass
class ModelConfig:
    mode: str = "detailed"           # "simplified" | "detailed"
    spreading_loss: bool = False     # 1/L amplitude spreading relative to the shortest path
    rx_pattern: str = "omni"         # "omni" | "directional"
    rx_boresight_deg: float = 0.0    # receive-antenna pointing direction [deg]
    rx_pattern_exp: float = 2.0      # directivity exponent n in ((1+cos d)/2)^n

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scene:
    """Rectangular 2-D room with optional reflecting surfaces."""
    wall_x: float = 20.0             # right wall position / scene width [m]
    ceiling_y: float = 10.0          # ceiling height / scene height [m]
    tx_x: float = 2.0
    tx_y: float = 3.0
    rx_x: float = 12.0
    rx_y: float = 1.5
    ground: bool = True
    ceiling: bool = True
    wall: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def validate_scene(s: Scene) -> Scene:
    s = replace(s)
    s.wall_x = float(np.clip(s.wall_x, 2.0, 200.0))
    s.ceiling_y = float(np.clip(s.ceiling_y, 2.0, 100.0))
    s.tx_x = float(np.clip(s.tx_x, 0.05, s.wall_x - 0.05))
    s.rx_x = float(np.clip(s.rx_x, 0.05, s.wall_x - 0.05))
    s.tx_y = float(np.clip(s.tx_y, 0.05, s.ceiling_y - 0.05))
    s.rx_y = float(np.clip(s.rx_y, 0.05, s.ceiling_y - 0.05))
    return s


@dataclass
class PathTerm:
    """Fully evaluated path: every quantity the signal synthesis needs."""
    idx: int
    name: str
    kind: str
    source: str
    active: bool                 # enabled AND geometrically valid
    note: str                    # e.g. "surface missing"
    length_m: float
    angle_deg: float
    delay_s: float
    amp_rel: float               # user relative strength
    att_db: float
    refl_mag: float
    spread: float
    rx_gain: float
    amp_total: float             # final relative amplitude a_i
    phase_user_deg: float
    phase_refl_deg: float
    phase_prop_deg: float        # informational: -360*f*tau (wrapped to +-180)
    phase_total_deg: float       # carrier phase actually applied (0..360)
    polyline: list = field(default_factory=list)   # [(x, y), ...] for the scene plot

    @property
    def delay_ns(self) -> float:
        return self.delay_s * 1e9


# --------------------------------------------------------------------------- geometry
def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def resolve_geometry(source: str, scene: Scene):
    """Image-method geometry for a path source.

    Returns ``(length, angle_deg, polyline, note)``.  ``length`` is None if the
    path cannot exist (surface switched off, or Tx/Rx on the wrong side).
    The angle is the direction (from the receiver, measured from +x) the
    signal arrives from.
    """
    tx = (scene.tx_x, scene.tx_y)
    rx = (scene.rx_x, scene.rx_y)
    W, H = scene.wall_x, scene.ceiling_y

    def angle_to(pt):
        return float(np.degrees(np.arctan2(pt[1] - rx[1], pt[0] - rx[0])))

    if source == "direct":
        return _dist(tx, rx), angle_to(tx), [tx, rx], ""

    if source == "ground":
        if not scene.ground:
            return None, 0.0, [], "ground surface is off"
        img = (tx[0], -tx[1])
        s = (0.0 - img[1]) / (rx[1] - img[1])
        pt = (img[0] + s * (rx[0] - img[0]), 0.0)
    elif source == "ceiling":
        if not scene.ceiling:
            return None, 0.0, [], "ceiling surface is off"
        img = (tx[0], 2 * H - tx[1])
        s = (H - img[1]) / (rx[1] - img[1])
        pt = (img[0] + s * (rx[0] - img[0]), H)
    elif source == "wall":
        if not scene.wall:
            return None, 0.0, [], "wall is off"
        img = (2 * W - tx[0], tx[1])
        s = (W - img[0]) / (rx[0] - img[0])
        pt = (W, img[1] + s * (rx[1] - img[1]))
    else:
        return None, 0.0, [], "unknown source"

    if not (0.0 <= pt[0] <= W and 0.0 <= pt[1] <= H):
        return None, 0.0, [], "reflection point outside the room"
    return _dist(img, rx), angle_to(pt), [tx, pt, rx], ""


def rx_antenna_gain(angle_deg: float, model: ModelConfig) -> float:
    """Receive-antenna amplitude gain for an arrival direction (1 for omni)."""
    if model.rx_pattern != "directional":
        return 1.0
    delta = np.deg2rad(angle_deg - model.rx_boresight_deg)
    return float(((1.0 + np.cos(delta)) / 2.0) ** max(model.rx_pattern_exp, 0.0))


def _wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


# --------------------------------------------------------------------------- path terms
def compute_terms(paths: list[Path], tx: sg.TxConfig, model: ModelConfig,
                  scene: Scene) -> list[PathTerm]:
    """Evaluate every path (geometry, loss, delay, phases)."""
    scene = validate_scene(scene)
    f = tx.freq_hz
    raw = []
    for i, p in enumerate(paths):
        p = validate_path(p)
        note = ""
        valid = True
        polyline: list = []
        if p.source == "manual":
            length, angle, kind = p.length_m, p.angle_deg, p.kind
        else:
            length, angle, polyline, note = resolve_geometry(p.source, scene)
            kind = "direct" if p.source == "direct" else "reflected"
            if length is None:
                valid, length, angle = False, p.length_m, p.angle_deg
        raw.append((i, p, length, angle, kind, valid, note, polyline))

    live = [r[2] for r in raw if r[1].enabled and r[5]]
    l_min = max(min(live), 1e-3) if live else 1.0

    terms = []
    for i, p, length, angle, kind, valid, note, polyline in raw:
        active = bool(p.enabled and valid)
        refl_mag = abs(p.reflection_coeff) if kind == "reflected" else 1.0
        spread = (l_min / max(length, 1e-3)) if model.spreading_loss else 1.0
        gain = rx_antenna_gain(angle, model)
        amp = p.amplitude * 10 ** (-p.attenuation_db / 20.0) * refl_mag * spread * gain
        delay = length / C + p.extra_delay_ns * 1e-9
        phi_refl = 180.0 if (kind == "reflected" and p.reflection_coeff < 0) else 0.0
        phi_prop = _wrap180(-360.0 * f * delay)
        total = p.phase_deg + phi_refl + (phi_prop if model.mode == "detailed" else 0.0)
        terms.append(PathTerm(
            idx=i, name=p.name, kind=kind, source=p.source, active=active, note=note,
            length_m=float(length), angle_deg=float(angle), delay_s=float(delay),
            amp_rel=p.amplitude, att_db=p.attenuation_db, refl_mag=refl_mag,
            spread=float(spread), rx_gain=gain, amp_total=float(amp),
            phase_user_deg=p.phase_deg, phase_refl_deg=phi_refl,
            phase_prop_deg=float(phi_prop), phase_total_deg=float(total % 360.0),
            polyline=polyline))
    return terms


def active_terms(terms: list[PathTerm]) -> list[PathTerm]:
    return [t for t in terms if t.active]


# --------------------------------------------------------------------------- synthesis
def path_signal(t: np.ndarray, tx: sg.TxConfig, term: PathTerm, model: ModelConfig) -> np.ndarray:
    """Time-domain contribution y_i(t) of one path."""
    extra = np.deg2rad(term.phase_user_deg + term.phase_refl_deg)
    t_env = t - term.delay_s
    t_car = t_env if model.mode == "detailed" else t
    return sg.tx_wave(t_env, t_car, tx, extra_phase_rad=extra, scale=term.amp_total)


@dataclass
class SimResult:
    t: np.ndarray
    fs: float
    n_cycles: int
    terms: list
    x_tx: np.ndarray                 # transmitted waveform
    x_ref: np.ndarray                # reference used for the metrics
    ref_label: str
    path_signals: dict               # idx -> array (active paths only)
    y_clean: np.ndarray              # noise-free received waveform
    noise: np.ndarray
    y_noisy: np.ndarray
    sigma: float
    warnings: list


def simulate(tx: sg.TxConfig, paths: list[Path], model: ModelConfig, scene: Scene) -> SimResult:
    """Run the whole simulation for the given configuration."""
    tx = sg.validate_tx(replace(tx))
    scene = validate_scene(scene)
    t, fs, n_cycles = sg.make_time_axis(tx)
    terms = compute_terms(paths, tx, model, scene)
    warnings = []
    for p, tm in zip(paths, terms):
        if p.enabled and not tm.active:
            warnings.append(f"{tm.name} is enabled but not available: {tm.note}.")

    x_tx = sg.reference_wave(t, tx)
    sigs = {tm.idx: path_signal(t, tx, tm, model) for tm in active_terms(terms)}
    y = np.sum(list(sigs.values()), axis=0) if sigs else np.zeros_like(t)

    x_ref, ref_label = x_tx, "transmitted signal"
    if tx.reference_mode == "los":
        los = next((tm for tm in active_terms(terms) if tm.kind == "direct"), None)
        if los is not None:
            x_ref, ref_label = sigs[los.idx], f"direct path only ({los.name})"
        else:
            warnings.append("No active direct path: the reference falls back to the transmitted signal.")

    noise = sg.make_noise(len(t), tx)
    return SimResult(t=t, fs=fs, n_cycles=n_cycles, terms=terms, x_tx=x_tx, x_ref=x_ref,
                     ref_label=ref_label, path_signals=sigs, y_clean=y, noise=noise,
                     y_noisy=y + noise, sigma=sg.noise_sigma(tx), warnings=warnings)


# --------------------------------------------------------------------------- phasor (steady-state carrier) tools
def carrier_phasor(tx: sg.TxConfig, terms: list[PathTerm]) -> complex:
    """Complex amplitude V such that y(t) = Re{V e^{j 2 pi f t}} (envelope ignored)."""
    phi0 = np.deg2rad(tx.phase_deg)
    return complex(sum(tm.amp_total * tx.amplitude * np.exp(1j * (phi0 + np.deg2rad(tm.phase_total_deg)))
                       for tm in active_terms(terms)))


def sweep_receiver(paths, tx, model, scene, xs, y=None):
    """Received carrier strength while the receiver moves along x.

    Returns dict with ``combined`` and ``los`` magnitudes relative to the
    transmitted amplitude (1.0 = 0 dB) and the number of active paths.
    Only scene-linked paths change with position; manual paths keep their lengths.
    """
    combined, los = [], []
    for x in xs:
        s = replace(scene, rx_x=float(x))
        if y is not None:
            s = replace(s, rx_y=float(y))
        s = validate_scene(s)
        terms = compute_terms(paths, tx, model, s)
        combined.append(abs(carrier_phasor(tx, terms)) / tx.amplitude)
        first = next((tm for tm in active_terms(terms) if tm.kind == "direct"), None)
        los.append(first.amp_total if first else 0.0)
    return {"x": np.asarray(xs, float), "combined": np.array(combined), "los": np.array(los)}


def sweep_phase(paths, tx, model, scene, path_index: int, phases_deg):
    """Received carrier strength (relative to A) while one path's phase sweeps 0-360 deg."""
    out = []
    for ph in phases_deg:
        trial = [replace(p) for p in paths]
        trial[path_index] = replace(trial[path_index], phase_deg=float(ph) % 360.0)
        terms = compute_terms(trial, tx, model, scene)
        out.append(abs(carrier_phasor(tx, terms)) / tx.amplitude)
    return np.array(out)


# --------------------------------------------------------------------------- channel responses
def frequency_response(terms, tx, model, freqs_hz: np.ndarray) -> np.ndarray:
    """Complex channel response H(f) (relative to the transmitted signal).

    detailed  : H(f) = sum a_i e^{j(phi_user+phi_refl)} e^{-j 2 pi f tau_i}
    simplified: H(f) = sum a_i e^{j(phi_user+phi_refl)} e^{-j 2 pi (f - f0) tau_i}
                (propagation phase at the carrier is replaced by the phase slider)
    """
    h = np.zeros(len(freqs_hz), dtype=complex)
    for tm in active_terms(terms):
        fixed = np.deg2rad(tm.phase_user_deg + tm.phase_refl_deg)
        f_eff = freqs_hz if model.mode == "detailed" else freqs_hz - tx.freq_hz
        h += tm.amp_total * np.exp(1j * fixed) * np.exp(-1j * 2 * np.pi * f_eff * tm.delay_s)
    return h


def delay_stats(terms: list[PathTerm]) -> dict:
    """First arrival, maximum excess delay and power-weighted RMS delay spread."""
    act = active_terms(terms)
    if not act:
        return {"first_ns": 0.0, "last_ns": 0.0, "excess_ns": 0.0, "mean_ns": 0.0, "rms_ns": 0.0}
    d = np.array([tm.delay_ns for tm in act])
    p = np.array([tm.amp_total ** 2 for tm in act])
    if p.sum() <= 0:
        mean = rms = 0.0
    else:
        mean = float((p * d).sum() / p.sum())
        rms = float(np.sqrt((p * (d - mean) ** 2).sum() / p.sum()))
    return {"first_ns": float(d.min()), "last_ns": float(d.max()),
            "excess_ns": float(d.max() - d.min()), "mean_ns": mean, "rms_ns": rms}
