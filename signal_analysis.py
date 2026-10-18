"""Metrics, detection and automatic interpretation of a simulation result.

Quantities and their meaning
----------------------------
* instantaneous voltage  v(t)            -- the waveform samples themselves [V]
* peak amplitude         A_pk            -- max |v| (tone: sqrt(2) * RMS) [V]
* RMS amplitude          V_rms = sqrt(mean(v^2)) [V]
* normalized power       P = V_rms^2     -- power dissipated in 1 ohm [W]; a *relative*
                                            quantity, not a physical antenna power
* power ratio            P_rx / P_ref = (V_rms,rx / V_rms,ref)^2   (equal impedance)
* power change           10 log10(power ratio) [dB]

All metrics are computed from the *noise-free* received waveform; noise only
enters the SNR and the bit-error estimates.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

import propagation_model as pm
import signal_generator as sg

EPS = 1e-12
DB_FLOOR = -120.0


# --------------------------------------------------------------------------- scalar helpers
def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0


def peak_amplitude(y: np.ndarray, is_tone: bool) -> float:
    """Peak amplitude.  For a single-frequency tone the exact value is sqrt(2)*RMS,
    which avoids the small sampling error of max|y|."""
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(2.0) * rms(y)) if is_tone else float(np.max(np.abs(y)))


def amplitude_change_pct(rx_amp: float, ref_amp: float) -> float:
    """100 * (rx - ref) / ref ; NaN if the reference is (near) zero."""
    if abs(ref_amp) < EPS:
        return float("nan")
    return 100.0 * (rx_amp - ref_amp) / ref_amp


def power_ratio(rx_rms: float, ref_rms: float) -> float:
    """(V_rms,rx / V_rms,ref)^2 ; NaN if the reference is (near) zero."""
    if abs(ref_rms) < EPS:
        return float("nan")
    return float((rx_rms / ref_rms) ** 2)


def power_db(ratio: float) -> float:
    """10 log10(ratio), floored at -120 dB so that exact cancellation stays finite."""
    if np.isnan(ratio):
        return float("nan")
    if ratio <= 10 ** (DB_FLOOR / 10):
        return DB_FLOOR
    return float(10.0 * np.log10(ratio))


def snr_db(signal_power: float, sigma: float) -> float:
    """SNR = P_signal / sigma^2 in dB; +inf when noise is off."""
    if sigma <= 0:
        return float("inf")
    if signal_power <= EPS:
        return DB_FLOOR
    return float(10.0 * np.log10(signal_power / sigma ** 2))


def q_function(x: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2.0))


# --------------------------------------------------------------------------- distortion (shape error)
def _baseband(y: np.ndarray, t: np.ndarray, f: float, spc: int) -> np.ndarray:
    """Complex envelope: mix down and average over exactly one carrier cycle
    (a boxcar low-pass that nulls the 2f term)."""
    z = y * np.exp(-2j * np.pi * f * t)
    kernel = np.ones(spc) / spc
    return np.fft.ifft(np.fft.fft(z) * np.fft.fft(kernel, n=len(z)))


def waveform_distortion(y: np.ndarray, x_tx: np.ndarray, t: np.ndarray, tx: sg.TxConfig) -> float:
    """Shape distortion d = sqrt(1 - rho^2) in [0, 1].

    rho is the peak normalised correlation between the complex envelopes of the
    received and the transmitted waveform, maximised over time lag and over a
    complex scale.  Pure attenuation, a pure delay or a carrier phase shift give
    d = 0; inter-symbol interference gives d > 0.  A CW tone is always d = 0.
    """
    zb = _baseband(y, t, tx.freq_hz, tx.samples_per_cycle)
    zr = _baseband(x_tx, t, tx.freq_hz, tx.samples_per_cycle)
    nb, nr = np.linalg.norm(zb), np.linalg.norm(zr)
    if nb < EPS or nr < EPS:
        return 0.0
    cc = np.abs(np.fft.ifft(np.fft.fft(zb) * np.conj(np.fft.fft(zr))))
    k = int(np.argmax(cc))
    best = cc[k]
    if len(cc) > 2:   # 3-point parabolic refinement for the sub-sample delay
        a, b, c = cc[(k - 1) % len(cc)], cc[k], cc[(k + 1) % len(cc)]
        den = a - 2 * b + c
        if abs(den) > EPS:
            best = b - 0.125 * (c - a) ** 2 / den
    rho = min(1.0, best / (nb * nr))
    return float(np.sqrt(max(0.0, 1.0 - rho ** 2)))


# --------------------------------------------------------------------------- BPSK detection
def detect_bpsk(sim: pm.SimResult, tx: sg.TxConfig) -> dict:
    """Coherent BPSK detection with ideal carrier recovery and bit timing locked
    to the first-arriving path.

    Decision variable per bit  d_k = 2 Re{ mean_bit( y e^{-j w t} ) e^{-j theta} }.
    Ideal single-path case: d_k = A * b_k.  Returns the normalised eye opening
    (``margin``), the measured BER from the simulated noisy waveform and an
    analytic BER estimate Q(b_k d_k / sigma_d) with sigma_d^2 = 2 sigma^2 / N_bit.
    """
    spb = tx.cycles_per_bit * tx.samples_per_cycle
    n_bits_total = len(sim.t) // spb
    bits = np.tile(tx.bit_array, n_bits_total // len(tx.bits))
    act = pm.active_terms(sim.terms)
    out = {"n_bits": int(len(bits)), "errors": 0, "ber_measured": float("nan"),
           "ber_estimate": float("nan"), "margin": float("nan")}
    if not act or len(bits) == 0:
        return out

    shift = int(round(min(tm.delay_s for tm in act) * sim.fs))     # timing sync to first arrival
    carrier = np.exp(-2j * np.pi * tx.freq_hz * sim.t)

    def bit_stats(sig, sh=shift):
        z = np.roll(sig * carrier, -sh)[: len(bits) * spb]
        return z.reshape(len(bits), spb).mean(axis=1)

    z_clean = bit_stats(sim.y_clean)
    theta = np.angle(np.sum(bits * z_clean))
    d_clean = 2.0 * np.real(z_clean * np.exp(-1j * theta))
    z_ref = bit_stats(sim.x_tx, 0)          # ideal, un-delayed reference
    d_ref = 2.0 * np.real(z_ref * np.exp(-1j * np.angle(np.sum(bits * z_ref))))
    ideal = float(np.min(bits * d_ref))
    if ideal > EPS:
        out["margin"] = float(np.min(bits * d_clean) / ideal)

    d_noisy = 2.0 * np.real(bit_stats(sim.y_noisy) * np.exp(-1j * theta))
    errors = int(np.sum(np.sign(d_noisy) != bits))
    out["errors"] = errors
    out["ber_measured"] = errors / len(bits)

    sigma_d = np.sqrt(2.0 * sim.sigma ** 2 / spb)
    if sigma_d > 0:
        out["ber_estimate"] = float(np.mean(q_function(bits * d_clean / sigma_d)))
    else:
        out["ber_estimate"] = float(np.mean(bits * d_clean <= 0))
    return out


# --------------------------------------------------------------------------- metrics
def compute_metrics(sim: pm.SimResult, tx: sg.TxConfig, model: pm.ModelConfig) -> dict:
    """Everything shown in the numerical dashboard and the observation table."""
    tx = sg.validate_tx(tx)
    is_tone = not tx.is_digital
    act = pm.active_terms(sim.terms)

    ref_rms, rx_rms = rms(sim.x_ref), rms(sim.y_clean)
    ref_amp = peak_amplitude(sim.x_ref, is_tone)
    rx_amp = peak_amplitude(sim.y_clean, is_tone)
    ratio = power_ratio(rx_rms, ref_rms)

    amps = np.array([tm.amp_total for tm in act]) if act else np.array([0.0])
    v_rel = abs(pm.carrier_phasor(tx, sim.terms)) / tx.amplitude
    los = next((tm for tm in act if tm.kind == "direct"), None)
    dstats = pm.delay_stats(sim.terms)
    f = tx.freq_hz

    m = {
        "is_digital": tx.is_digital,
        "ref_label": sim.ref_label,
        "n_active": len(act),
        "n_total": len(sim.terms),
        "ref_amp": ref_amp, "rx_amp": rx_amp,
        "ref_rms": ref_rms, "rx_rms": rx_rms,
        "amp_change_pct": amplitude_change_pct(rx_amp, ref_amp),
        "ref_power": ref_rms ** 2, "rx_power": rx_rms ** 2,
        "power_ratio": ratio, "power_db": power_db(ratio),
        "ref_zero": bool(ref_rms < EPS),
        "v_rel": float(v_rel),
        "sum_abs": float(amps.sum()), "sum_power": float(np.sqrt(np.sum(amps ** 2))),
        "strongest": float(amps.max()),
        "los_amp": float(los.amp_total) if los else float("nan"),
        "gain_vs_los_db": (float(20 * np.log10(max(v_rel, 1e-6) / los.amp_total))
                           if los and los.amp_total > EPS else float("nan")),
        "delay": dstats,
        "excess_cycles": dstats["excess_ns"] * 1e-9 * f,
        "bit_ns": tx.cycles_per_bit / f * 1e9 if tx.is_digital else float("nan"),
        "sigma": sim.sigma,
        "snr_db": snr_db(rms(sim.y_clean) ** 2, sim.sigma),
        "noise_on": tx.noise_level > 0,
        "distortion": waveform_distortion(sim.y_clean, sim.x_tx, sim.t, tx),
        "margin": float("nan"), "ber_measured": float("nan"), "ber_estimate": float("nan"),
        "n_bits": 0, "bit_errors": 0,
    }
    if tx.is_digital:
        det = detect_bpsk(sim, tx)
        m.update(margin=det["margin"], ber_measured=det["ber_measured"],
                 ber_estimate=det["ber_estimate"], n_bits=det["n_bits"], bit_errors=det["errors"])
    m["label"], m["tone"], m["points"] = interpret(m)
    return m


# --------------------------------------------------------------------------- interpretation
def interference_class(m: dict) -> str:
    """Classify how the paths combined, from the phasor sum |V| against the
    strongest single path (smax), the fully-constructive sum (S) and the
    incoherent power sum (P).  Returns '' for fewer than two paths."""
    if m["n_active"] < 2 or m["strongest"] < EPS:
        return ""
    v, smax, s = m["v_rel"], m["strongest"], m["sum_abs"]
    if v <= 0.05 * smax:
        return "near_cancellation"
    if v > 1.05 * smax:
        return "constructive" if v >= 0.9 * s else "partial_constructive"
    if v < 0.95 * smax:
        return "partial_destructive"
    return "neutral"


def _fmt_db(x: float) -> str:
    return "n/a" if np.isnan(x) else ("<= -120 dB" if x <= DB_FLOOR else f"{x:+.2f} dB")


def interpret(m: dict):
    """Turn the calculated numbers into plain statements.

    Returns (headline, tone, list_of_statements).  Nothing here is hard-coded
    example output: every sentence is chosen by thresholds on the metrics.
    """
    pts: list[str] = []
    if m["n_active"] == 0:
        return "No active path", "bad", ["No path reaches the receiver, so the received signal is zero."]
    if m["ref_zero"]:
        return "Reference is zero", "neutral", ["The reference signal has zero amplitude; ratios are not defined."]

    db = m["power_db"]
    # phasor-sum classes describe a steady carrier; for BPSK data the bit-by-bit overlap matters instead
    cls = "" if m["is_digital"] else interference_class(m)
    cls_text = {
        "near_cancellation": "Near cancellation: the paths almost completely cancel each other.",
        "constructive": "Constructive interference: the paths are nearly in phase and add up.",
        "partial_constructive": "Partial constructive interference: the paths reinforce each other, but not fully.",
        "partial_destructive": "Partial destructive interference: the paths partly cancel, leaving less than the strongest path alone.",
        "neutral": "The paths combine without a large change compared with the strongest path.",
    }
    if cls:
        pts.append(cls_text[cls])

    if db >= 0.5:
        level = "Signal strengthened"
        tone = "good"
    elif db <= -10:
        level = "Signal strongly attenuated"
        tone = "bad"
    elif db <= -0.5:
        level = "Signal weakened"
        tone = "warn"
    else:
        level = "Signal approximately unchanged"
        tone = "neutral"
    pts.append(f"{level}: received power is {_fmt_db(db)} relative to the reference "
               f"({m['ref_label']}); amplitude change {m['amp_change_pct']:+.1f} %.")

    if m["n_active"] >= 2 and not m["is_digital"] and not np.isnan(m["gain_vs_los_db"]):
        g = m["gain_vs_los_db"]
        if g > 0.5:
            pts.append(f"Multipath helped here: the combined carrier is {g:+.2f} dB stronger than the direct path alone.")
        elif g < -0.5:
            pts.append(f"Multipath hurt here: the combined carrier is {g:+.2f} dB weaker than the direct path alone.")
        else:
            pts.append("The extra paths changed the carrier level by less than 0.5 dB compared with the direct path alone.")

    headline = {"near_cancellation": "Near cancellation", "constructive": "Constructive interference",
                "partial_constructive": "Partial constructive interference",
                "partial_destructive": "Partial destructive interference"}.get(cls, level)
    if cls in ("constructive", "partial_constructive") and db < 0.5:
        headline = level
    if cls == "near_cancellation":
        tone = "bad"

    if m["distortion"] > 0.05:
        pts.append(f"Signal waveform distorted: shape error {m['distortion'] * 100:.1f} % "
                   f"(delay spread {m['delay']['excess_ns']:.1f} ns).")
        if m["is_digital"]:
            headline, tone = "Waveform distorted (inter-symbol interference)", "warn"
            if db > 0.5:
                pts.append("More received power does not mean better data here: the extra energy is echo energy "
                           "that lands on neighbouring bits.")
        elif tone == "neutral":
            tone = "warn"
            headline = "Waveform distorted"
    elif m["is_digital"]:
        pts.append("Waveform shape is preserved (no significant inter-symbol interference).")
    else:
        pts.append("A single-frequency tone keeps its shape in multipath; only its amplitude and phase change.")

    if m["is_digital"] and not np.isnan(m["margin"]):
        if m["margin"] <= 0:
            pts.append("Eye closed: the decision margin is zero or negative, so bit errors occur even without noise.")
            tone = "bad"
        elif m["margin"] < 0.5:
            pts.append(f"Eye partly closed: only {m['margin'] * 100:.0f} % of the ideal decision margin remains (inter-symbol interference).")
    return headline, tone, pts


# --------------------------------------------------------------------------- tables and explanations
def path_rows(sim: pm.SimResult, tx: sg.TxConfig, model: pm.ModelConfig) -> list[dict]:
    """One row per path for the path list / observation tables."""
    rows = []
    act = pm.active_terms(sim.terms)
    ref_phase = act[0].phase_total_deg if act else 0.0
    for tm in sim.terms:
        rel = (tm.phase_total_deg - ref_phase) % 360.0 if tm.active else float("nan")
        rows.append({
            "Path": tm.name, "Type": tm.kind, "Active": "yes" if tm.active else "no",
            "Length (m)": round(tm.length_m, 3), "Delay (ns)": round(tm.delay_ns, 3),
            "Arrival angle (deg)": round(tm.angle_deg, 1),
            "Strength": round(tm.amp_rel, 3), "Attenuation (dB)": round(tm.att_db, 2),
            "|Reflection coeff.|": round(tm.refl_mag, 3),
            "Final amplitude a_i": round(tm.amp_total, 4),
            "Phase: user (deg)": round(tm.phase_user_deg, 1),
            "Phase: propagation (deg)": round(tm.phase_prop_deg, 1),
            "Phase: reflection (deg)": round(tm.phase_refl_deg, 1),
            "Phase applied (deg)": round(tm.phase_total_deg, 1),
            "Phase vs first path (deg)": None if np.isnan(rel) else round(rel, 1),
            "Note": tm.note,
        })
    return rows


def explanations(sim: pm.SimResult, tx: sg.TxConfig, model: pm.ModelConfig, m: dict,
                 advanced: bool) -> list[str]:
    """'Why does this result occur?' text generated from the actual numbers."""
    out: list[str] = []
    f = tx.freq_hz
    lam = pm.C / f
    act = pm.active_terms(sim.terms)
    if not act:
        return ["Enable at least one valid path to see an explanation."]

    out.append(f"The carrier is {tx.freq_mhz:g} MHz, so one wavelength is {lam:.3f} m and one period lasts "
               f"{1e3 / tx.freq_mhz:.3f} ns. A path that is one wavelength longer arrives exactly one cycle later.")
    for tm in act:
        cyc = tm.delay_s * f
        if model.mode == "detailed":
            ph = (f"its propagation phase is {tm.phase_prop_deg:+.0f} deg "
                  f"({cyc:.2f} cycles of delay, only the fractional part matters)")
        else:
            ph = "propagation phase is not applied in simplified mode - only the phase slider sets its phase"
        out.append(f"{tm.name} ({tm.kind}): {tm.length_m:.2f} m, delay {tm.delay_ns:.2f} ns; {ph}. "
                   f"Strength {tm.amp_total:.3f} of the transmitted amplitude ({20 * np.log10(max(tm.amp_total, 1e-9)):+.1f} dB).")
    if m["n_active"] >= 2:
        out.append(f"Adding the paths as rotating arrows (phasors) gives a resultant of {m['v_rel']:.3f}, versus {m['strongest']:.3f} "
                   f"for the strongest path alone and {m['sum_abs']:.3f} if all paths pointed the same way. "
                   "Where the arrows line up the signal grows; where they point apart it shrinks.")
    else:
        out.append("With one path the receiver sees a scaled and delayed copy of the transmitted signal.")
    if m["is_digital"]:
        out.append(f"Each bit lasts {m['bit_ns']:.2f} ns. The delay spread of {m['delay']['excess_ns']:.2f} ns is "
                   f"{m['delay']['excess_ns'] / m['bit_ns']:.2f} bit periods, so delayed copies overlap neighbouring bits "
                   "(inter-symbol interference).")
    if advanced:
        out.append(f"Equations used: V_rms = {m['rx_rms']:.4f} V (received) and {m['ref_rms']:.4f} V (reference); "
                   f"power ratio = ({m['rx_rms']:.4f}/{m['ref_rms']:.4f})^2 = {m['power_ratio']:.4g}; "
                   f"10 log10(ratio) = {_fmt_db(m['power_db'])}.")
        rms_ns = m["delay"]["rms_ns"]
        coh = "unbounded (single delay)" if rms_ns < 1e-6 else "%.1f MHz" % (1e3 / (5 * rms_ns))
        out.append("RMS delay spread %.2f ns; an approximate coherence bandwidth 1/(5 tau_rms) is %s." % (rms_ns, coh))
    return out


# --------------------------------------------------------------------------- envelopes for the digital plots
def complex_envelope(y: np.ndarray, t: np.ndarray, tx: sg.TxConfig) -> np.ndarray:
    """Complex envelope  z(t) = 2 * LPF{ y e^{-j 2 pi f t} }  (|z| = carrier amplitude)."""
    return 2.0 * _baseband(y, t, tx.freq_hz, tx.samples_per_cycle)


def aligned_envelopes(sim: pm.SimResult, tx: sg.TxConfig) -> dict:
    """Real baseband bit waveforms for the distortion / eye-diagram plots.

    The received envelope is rotated by the carrier phase that best matches the
    (delayed) transmitted envelope - the same ideal carrier recovery the BPSK
    detector uses - so a clean single path appears as +/-a.
    """
    act = pm.active_terms(sim.terms)
    shift = int(round(min(tm.delay_s for tm in act) * sim.fs)) if act else 0
    zr = complex_envelope(sim.x_tx, sim.t, tx)
    zb = complex_envelope(sim.y_clean, sim.t, tx)
    phi0 = np.deg2rad(tx.phase_deg)
    c = np.sum(zb * np.conj(np.roll(zr, shift)))
    theta = np.angle(c) if abs(c) > EPS else phi0
    return {"tx_env": np.real(zr * np.exp(-1j * phi0)),
            "rx_env": np.real(zb * np.exp(-1j * theta)),
            "shift": shift}
