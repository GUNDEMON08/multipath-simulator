"""Observation tables, CSV exports and the downloadable experiment summary.

Everything exported is labelled as a *simulated observation* - it comes from the
mathematical model, not from a physical laboratory measurement.
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

import propagation_model as pm
import signal_analysis as sa
import signal_generator as sg

DISCLAIMER = ("Simulated observations from a simplified 2-D multipath model - "
              "not physical laboratory measurements.")


def _num(x, digits=4):
    """JSON/CSV-friendly number: NaN -> None, +/-inf -> text."""
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if np.isnan(x):
        return None
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return round(float(x), digits)


def observation_row(name: str, sim: pm.SimResult, tx: sg.TxConfig, model: pm.ModelConfig, m: dict) -> dict:
    """One row of the automatic observation table."""
    act = pm.active_terms(sim.terms)
    first_phase = act[0].phase_total_deg if act else 0.0
    return {
        "Experiment": name,
        "Data type": "Simulated observation",
        "Signal": "BPSK digital" if tx.is_digital else "Sine wave",
        "Model mode": model.mode,
        "Frequency (MHz)": tx.freq_mhz,
        "Active paths": m["n_active"],
        "Path strength / attenuation": "; ".join(
            f"{t.name}: a={t.amp_total:.3f} ({t.att_db:.1f} dB att.)" for t in act) or "-",
        "Relative phase (deg)": "; ".join(
            f"{t.name}: {(t.phase_total_deg - first_phase) % 360:.0f}" for t in act) or "-",
        "Propagation delay (ns)": "; ".join(f"{t.name}: {t.delay_ns:.2f}" for t in act) or "-",
        "Reference": m["ref_label"],
        "Reference amplitude (V)": _num(m["ref_amp"]),
        "Received peak amplitude (V)": _num(m["rx_amp"]),
        "Received RMS (V)": _num(m["rx_rms"]),
        "Amplitude change (%)": _num(m["amp_change_pct"], 2),
        "Power ratio": _num(m["power_ratio"], 5),
        "Power change (dB)": _num(m["power_db"], 3),
        "SNR (dB)": _num(m["snr_db"], 2) if m["noise_on"] else "noise off",
        "Distortion (shape error)": _num(m["distortion"], 4),
        "Eye margin (1 = ideal)": _num(m["margin"], 3) if m["is_digital"] else "n/a",
        "BER estimate": _num(m["ber_estimate"], 6) if m["is_digital"] else "n/a",
        "Interpretation": m["label"] + ". " + " ".join(m["points"][:2]),
    }


def observations_csv(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_csv(index=False)


def samples_csv(sim: pm.SimResult, tx: sg.TxConfig) -> str:
    """Time-domain samples: transmitted, every active path, received (clean and noisy)."""
    cols = {"t_ns": sim.t * 1e9, "transmitted_V": sim.x_tx, "reference_V": sim.x_ref}
    for tm in pm.active_terms(sim.terms):
        cols[f"path_{tm.idx + 1}_{tm.name}_V"] = sim.path_signals[tm.idx]
    cols["received_clean_V"] = sim.y_clean
    cols["received_noisy_V"] = sim.y_noisy
    buf = io.StringIO()
    pd.DataFrame(cols).to_csv(buf, index=False, float_format="%.9g")
    return buf.getvalue()


def config_json(tx: sg.TxConfig, paths: list[pm.Path], model: pm.ModelConfig, scene: pm.Scene,
                name: str = "") -> str:
    return json.dumps({"name": name, "tx": tx.to_dict(), "model": model.to_dict(),
                       "scene": scene.to_dict(), "paths": [p.to_dict() for p in paths]}, indent=2)


def summary_markdown(name: str, sim: pm.SimResult, tx: sg.TxConfig, model: pm.ModelConfig, m: dict) -> str:
    """Self-contained experiment report (Markdown)."""
    rows = sa.path_rows(sim, tx, model)
    table = pd.DataFrame(rows)[["Path", "Type", "Active", "Length (m)", "Delay (ns)", "Final amplitude a_i",
                                "Phase: user (deg)", "Phase: propagation (deg)", "Phase: reflection (deg)",
                                "Phase applied (deg)"]]
    header = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in table.itertuples(index=False))

    def fmt(x, spec):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)

    snr = "noise off" if not m["noise_on"] else f"{m['snr_db']:.2f} dB"
    lines = [
        f"# Experiment summary - {name}",
        "",
        f"*{DISCLAIMER}*",
        "",
        "## Transmitter",
        f"- Signal: {'BPSK digital (bits ' + tx.bits + ', ' + str(tx.cycles_per_bit) + ' carrier cycles/bit)' if tx.is_digital else 'sine wave'}",
        f"- Amplitude {tx.amplitude:g} V, frequency {tx.freq_mhz:g} MHz (wavelength {pm.C / tx.freq_hz:.3f} m), initial phase {tx.phase_deg:g} deg",
        f"- Window {len(sim.t)} samples ({sim.n_cycles} carrier cycles), {tx.samples_per_cycle} samples/cycle, noise sigma = {tx.noise_level:g} x A",
        f"- Model mode: {model.mode}; spreading loss {'on' if model.spreading_loss else 'off'}; "
        f"receive antenna {model.rx_pattern}",
        "",
        "## Paths",
        header + body,
        "## Results",
        f"- Reference: {m['ref_label']}",
        f"- Reference amplitude {m['ref_amp']:.4f} V, received peak amplitude {m['rx_amp']:.4f} V, received RMS {m['rx_rms']:.4f} V",
        f"- Amplitude change {fmt(m['amp_change_pct'], '+.2f')} %",
        f"- Reference power {m['ref_power']:.4g} W (1 ohm normalised), received power {m['rx_power']:.4g} W",
        f"- Power ratio {fmt(m['power_ratio'], '.5g')}, power change {fmt(m['power_db'], '+.3f')} dB",
        f"- Active paths {m['n_active']}; excess delay {m['delay']['excess_ns']:.2f} ns; RMS delay spread {m['delay']['rms_ns']:.2f} ns",
        f"- SNR: {snr}",
        f"- Waveform distortion (shape error): {m['distortion']:.4f}",
    ]
    if m["is_digital"]:
        lines.append(f"- Eye margin {fmt(m['margin'], '.3f')} (1 = ideal); BER measured {fmt(m['ber_measured'], '.4g')} "
                     f"({m['bit_errors']} errors in {m['n_bits']} bits), BER estimate {fmt(m['ber_estimate'], '.4g')}")
    lines += ["", f"## Interpretation: {m['label']}"] + [f"- {p}" for p in m["points"]]
    lines += [
        "",
        "## Model and equations",
        "- Received signal: y(t) = sum_i A a_i cos(2 pi f (t - tau_i) + phi0 + phi_user_i + phi_refl_i)",
        "- Delay: tau_i = L_i / c (+ optional extra delay); propagation phase is -2 pi f tau_i (applied once, via the delay)",
        "- a_i = strength_i x 10^(-att_i/20) x |Gamma_i| x spreading_i x rx_gain(theta_i)",
        "- Amplitude change % = 100 (A_rx - A_ref) / A_ref;  power ratio = (V_rms,rx / V_rms,ref)^2;  dB = 10 log10(power ratio)",
        "- Assumptions: single-bounce image-method geometry, real reflection coefficients, static channel, "
        "no diffraction, scattering or Doppler; this is not a full electromagnetic solver.",
        "",
    ]
    return "\n".join(lines)
