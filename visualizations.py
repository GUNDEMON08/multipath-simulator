"""Plotly figures for the dashboard.

Colour convention (used consistently in every plot):
    transmitted signal  -> blue   (TX_COLOR)
    reference           -> grey   (REF_COLOR)
    individual paths    -> PATH_COLORS[i]
    received / combined -> orange (RX_COLOR)
Time is shown in ns, frequency in MHz, voltages in V.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import propagation_model as pm
import signal_analysis as sa
import signal_generator as sg

TX_COLOR = "#4DA3FF"
REF_COLOR = "#9AA7BD"
RX_COLOR = "#FF8A3D"
NOISE_COLOR = "rgba(255,138,61,0.35)"
PATH_COLORS = ["#2EC4B6", "#B388FF", "#FFD166", "#EF476F", "#7BD389",
               "#5CE1E6", "#C77DFF", "#B5E48C", "#FF6B6B", "#E9C46A"]
GRID = "rgba(150,165,190,0.18)"


def path_color(idx: int) -> str:
    return PATH_COLORS[idx % len(PATH_COLORS)]


def _base(title: str, xlabel: str, ylabel: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=15), x=0.01, xanchor="left"),
        height=height,
        margin=dict(l=55, r=15, t=48, b=48),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,27,46,0.6)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(size=11)),
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=12),
    )
    fig.update_xaxes(title_text=xlabel, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=ylabel, gridcolor=GRID, zeroline=True, zerolinecolor="rgba(150,165,190,0.35)")
    return fig


def _line(fig, x, y, name, color, width=2.0, dash=None, unit="V", xunit="ns", fmt=".4f"):
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name, line=dict(color=color, width=width, dash=dash),
        hovertemplate="%{y:" + fmt + "} " + unit + "<extra>" + name + "</extra>"))


# --------------------------------------------------------------------------- time-domain plots
def fig_transmitted(sim: pm.SimResult, tx: sg.TxConfig) -> go.Figure:
    fig = _base("1. Transmitted waveform", "Time (ns)", "Voltage (V)")
    kind = "BPSK data" if tx.is_digital else "carrier"
    _line(fig, sim.t * 1e9, sim.x_tx, f"Transmitted {kind} (A = {tx.amplitude:g} V)", TX_COLOR)
    fig.update_layout(hovermode="x unified")
    return fig


def fig_paths(sim: pm.SimResult, tx: sg.TxConfig) -> go.Figure:
    fig = _base("2. Individual path contributions", "Time (ns)", "Voltage (V)", height=400)
    t = sim.t * 1e9
    for tm in pm.active_terms(sim.terms):
        label = f"{tm.name} (a={tm.amp_total:.2f}, {tm.delay_ns:.1f} ns)"
        _line(fig, t, sim.path_signals[tm.idx], label, path_color(tm.idx), width=1.6)
    _line(fig, t, sim.y_clean, "Sum = received", RX_COLOR, width=2.6, dash="dot")
    return fig


def fig_received(sim: pm.SimResult, tx: sg.TxConfig) -> go.Figure:
    fig = _base("3. Combined received waveform", "Time (ns)", "Voltage (V)")
    t = sim.t * 1e9
    if tx.noise_level > 0:
        _line(fig, t, sim.y_noisy, "Received + noise", NOISE_COLOR, width=1.0)
    _line(fig, t, sim.y_clean, "Received (noise-free)", RX_COLOR, width=2.4)
    return fig


def fig_tx_vs_rx(sim: pm.SimResult, tx: sg.TxConfig, ref_label: str) -> go.Figure:
    fig = _base("4. Transmitted vs received", "Time (ns)", "Voltage (V)", height=400)
    t = sim.t * 1e9
    _line(fig, t, sim.x_tx, "Transmitted", TX_COLOR, width=2.0)
    if sim.ref_label != "transmitted signal":
        _line(fig, t, sim.x_ref, f"Reference ({sim.ref_label})", REF_COLOR, width=1.6, dash="dash")
    _line(fig, t, sim.y_clean, "Received", RX_COLOR, width=2.4)
    return fig


def fig_envelope_power(sim: pm.SimResult, tx: sg.TxConfig) -> go.Figure:
    """Received amplitude envelope and instantaneous normalised power versus time."""
    fig = _base("5. Received amplitude and power vs time", "Time (ns)", "Envelope amplitude (V)", height=400)
    t = sim.t * 1e9
    env = np.abs(sa.complex_envelope(sim.y_clean, sim.t, tx))
    ref_env = np.abs(sa.complex_envelope(sim.x_ref, sim.t, tx))
    _line(fig, t, ref_env, "Reference envelope", REF_COLOR, width=1.6, dash="dash")
    _line(fig, t, env, "Received envelope |z(t)|", RX_COLOR, width=2.4)
    power = env ** 2 / 2.0
    fig.add_trace(go.Scatter(x=t, y=power, mode="lines", name="Normalised power |z|^2/2 (V^2, 1 ohm)",
                             line=dict(color="#C77DFF", width=1.8), yaxis="y2",
                             hovertemplate="%{y:.4f} V^2<extra>power</extra>"))
    fig.update_layout(yaxis2=dict(title="Normalised power (V^2 = W into 1 ohm)", overlaying="y", side="right",
                                  showgrid=False, rangemode="tozero"))
    fig.update_layout(yaxis_rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- sweeps
def fig_position(sweep: dict, current_x: float, current_val: float, width_label: str = "") -> go.Figure:
    """Received strength versus receiver position (narrow-band, steady-state)."""
    fig = _base("6. Received strength vs receiver position", "Receiver position x (m)",
                "Strength relative to transmitted (dB)", height=400)
    comb_db = 20 * np.log10(np.maximum(sweep["combined"], 1e-6))
    los_db = 20 * np.log10(np.maximum(sweep["los"], 1e-6))
    fig.add_trace(go.Scatter(x=sweep["x"], y=los_db, mode="lines", name="Direct path only",
                             line=dict(color=REF_COLOR, width=1.6, dash="dash"),
                             hovertemplate="%{y:.2f} dB<extra>direct only</extra>"))
    fig.add_trace(go.Scatter(x=sweep["x"], y=comb_db, mode="lines", name="All active paths (multipath)",
                             line=dict(color=RX_COLOR, width=2.4),
                             hovertemplate="%{y:.2f} dB<extra>multipath</extra>"))
    fig.add_trace(go.Scatter(x=[current_x], y=[20 * np.log10(max(current_val, 1e-6))], mode="markers",
                             name="Current receiver", marker=dict(size=12, color="#00E5A8", symbol="diamond",
                                                                  line=dict(color="white", width=1)),
                             hovertemplate="x=%{x:.2f} m<br>%{y:.2f} dB<extra>receiver</extra>"))
    fig.update_yaxes(range=[max(-60, float(np.min(comb_db)) - 3), float(max(np.max(comb_db), np.max(los_db))) + 3])
    return fig


def fig_phase(phases: np.ndarray, mags: np.ndarray, current_phase: float, current_mag: float,
              path_name: str) -> go.Figure:
    fig = _base(f"7. Relative phase vs received strength (sweeping {path_name})",
                f"Phase shift of {path_name} (deg)", "Received amplitude relative to A", height=400)
    fig.add_trace(go.Scatter(x=phases, y=mags, mode="lines", name="Received amplitude",
                             line=dict(color=RX_COLOR, width=2.4),
                             hovertemplate="phase %{x:.0f} deg<br>%{y:.3f} A<extra></extra>"))
    db = 20 * np.log10(np.maximum(mags, 1e-6))
    fig.add_trace(go.Scatter(x=phases, y=db, mode="lines", name="Same in dB (right axis)", yaxis="y2",
                             line=dict(color="#C77DFF", width=1.4, dash="dot"),
                             hovertemplate="%{y:.2f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=[current_phase], y=[current_mag], mode="markers", name="Current setting",
                             marker=dict(size=12, color="#00E5A8", symbol="diamond", line=dict(color="white", width=1)),
                             hovertemplate="phase %{x:.0f} deg<br>%{y:.3f} A<extra>current</extra>"))
    fig.update_layout(yaxis2=dict(title="dB re A", overlaying="y", side="right", showgrid=False),
                      hovermode="closest")
    fig.update_xaxes(range=[0, 360], dtick=45)
    fig.update_layout(yaxis_rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- digital
def fig_digital(sim: pm.SimResult, tx: sg.TxConfig) -> go.Figure:
    """Transmitted vs received bit waveform (carrier removed)."""
    fig = _base("8a. Digital waveform distortion (bit envelopes)", "Time (ns)", "Envelope (V, carrier removed)",
                height=400)
    env = sa.aligned_envelopes(sim, tx)
    t = sim.t * 1e9
    _line(fig, t, env["tx_env"], "Transmitted bits", TX_COLOR, width=2.0)
    _line(fig, t, env["rx_env"], "Received (carrier phase recovered)", RX_COLOR, width=2.2)
    t_bit = tx.cycles_per_bit / tx.freq_hz * 1e9
    shift_ns = env["shift"] / sim.fs * 1e9
    n_bits = int(len(sim.t) / (tx.cycles_per_bit * tx.samples_per_cycle))
    for k in range(n_bits + 1):
        fig.add_vline(x=k * t_bit + shift_ns, line=dict(color="rgba(150,165,190,0.25)", width=1, dash="dot"))
    fig.update_xaxes(range=[0, sim.t[-1] * 1e9])
    return fig


def fig_eye(sim: pm.SimResult, tx: sg.TxConfig) -> go.Figure:
    """Eye diagram of the recovered bit envelope (two bit periods wide)."""
    fig = _base("8b. Eye diagram", "Time within two bit periods (ns)", "Envelope (V)", height=380)
    env = sa.aligned_envelopes(sim, tx)
    spb = tx.cycles_per_bit * tx.samples_per_cycle
    r = np.roll(env["rx_env"], -env["shift"] + spb // 2)
    n_bits = len(r) // spb
    seg_t = np.arange(2 * spb) / sim.fs * 1e9
    for k in range(n_bits):
        idx = (np.arange(2 * spb) + k * spb) % len(r)
        fig.add_trace(go.Scatter(x=seg_t, y=r[idx], mode="lines", showlegend=False,
                                 line=dict(color="rgba(255,138,61,0.55)", width=1),
                                 hoverinfo="skip"))
    t_bit = tx.cycles_per_bit / tx.freq_hz * 1e9
    fig.add_vline(x=t_bit, line=dict(color="#00E5A8", width=1.5, dash="dash"),
                  annotation_text="decision instant", annotation_position="top left")
    fig.update_layout(hovermode="closest")
    return fig


# --------------------------------------------------------------------------- channel responses
def frequency_grid(terms, tx: sg.TxConfig, points: int = 1201) -> np.ndarray:
    """Frequency axis around the carrier, wide enough to show several nulls."""
    st = pm.delay_stats(terms)
    f0 = tx.freq_hz
    if st["excess_ns"] > 1e-3:
        span = float(np.clip(6.0 / (st["excess_ns"] * 1e-9), 40e6, 3000e6))
    else:
        span = 200e6
    lo = max(1e6, f0 - span / 2)
    return np.linspace(lo, lo + span, points)


def fig_frequency(terms, tx: sg.TxConfig, model: pm.ModelConfig) -> go.Figure:
    fig = _base("9. Channel frequency response |H(f)|", "Frequency (MHz)", "|H(f)| (dB)", height=380)
    f = frequency_grid(terms, tx)
    h = pm.frequency_response(terms, tx, model, f)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    phase = np.degrees(np.angle(h))
    fig.add_trace(go.Scatter(x=f / 1e6, y=mag_db, mode="lines", name="|H(f)|", customdata=phase,
                             line=dict(color=RX_COLOR, width=2.4),
                             hovertemplate="%{y:.2f} dB, phase %{customdata:.0f} deg<extra>|H|</extra>"))
    h0 = pm.frequency_response(terms, tx, model, np.array([tx.freq_hz]))[0]
    fig.add_trace(go.Scatter(x=[tx.freq_mhz], y=[20 * np.log10(max(abs(h0), 1e-6))], mode="markers",
                             name="Carrier frequency", marker=dict(size=12, color="#00E5A8", symbol="diamond",
                                                                   line=dict(color="white", width=1)),
                             hovertemplate="%{x:.1f} MHz<br>%{y:.2f} dB<extra>carrier</extra>"))
    fig.update_layout(hovermode="closest")
    return fig


def fig_impulse(terms) -> go.Figure:
    fig = _base("10. Channel impulse response (path taps)", "Delay (ns)", "Tap amplitude a_i", height=360)
    act = pm.active_terms(terms)
    if not act:
        return fig
    for tm in act:
        fig.add_trace(go.Scatter(
            x=[tm.delay_ns, tm.delay_ns], y=[0, tm.amp_total], mode="lines+markers", name=tm.name,
            line=dict(color=path_color(tm.idx), width=3), marker=dict(size=[1, 10]),
            customdata=[[tm.phase_total_deg, tm.length_m]] * 2,
            hovertemplate="delay %{x:.2f} ns<br>a = %{y:.3f}<br>phase %{customdata[0]:.0f} deg, "
                          "length %{customdata[1]:.2f} m<extra>" + tm.name + "</extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- 2-D scene
def fig_scene(scene: pm.Scene, terms, height: int = 380) -> go.Figure:
    """Top-down room with transmitter, receiver, reflecting surfaces and every scene-linked path."""
    scene = pm.validate_scene(scene)
    W, H = scene.wall_x, scene.ceiling_y
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=40, r=10, t=36, b=40),
        title=dict(text="2-D propagation scene (side view)", font=dict(size=15), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,27,46,0.6)",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(size=11)),
        hovermode="closest", font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=12))
    fig.update_xaxes(title_text="x (m)", range=[-0.5, W + 0.5], gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text="y (m)", range=[-0.5, H + 0.5], gridcolor=GRID, zeroline=False,
                     scaleanchor="x", scaleratio=1, constrain="domain")

    # reflecting surfaces
    surf = [("Ground", scene.ground, [0, W], [0, 0], "#A98467"),
            ("Ceiling", scene.ceiling, [0, W], [H, H], "#8D99AE"),
            ("Wall", scene.wall, [W, W], [0, H], "#C9ADA7")]
    for name, on, xs, ys, col in surf:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name + (" (reflector)" if on else " (off)"),
                                 line=dict(color=col if on else "rgba(150,165,190,0.25)",
                                           width=6 if on else 2, dash="solid" if on else "dot"),
                                 hoverinfo="skip"))
    left = go.Scatter(x=[0, 0], y=[0, H], mode="lines", showlegend=False, hoverinfo="skip",
                      line=dict(color="rgba(150,165,190,0.25)", width=2, dash="dot"))
    fig.add_trace(left)

    # paths
    for tm in terms:
        if not (tm.active and tm.polyline):
            continue
        pts = tm.polyline
        col = path_color(tm.idx)
        reflected = len(pts) == 3
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{tm.name}: {tm.length_m:.2f} m, {tm.delay_ns:.1f} ns",
            line=dict(color=col, width=2.8, dash="dash" if reflected else "solid"),
            hovertemplate=f"{tm.name}<br>length {tm.length_m:.2f} m<br>delay {tm.delay_ns:.2f} ns<br>"
                          f"arrival {tm.angle_deg:.0f} deg<extra></extra>"))
        # direction arrow on the segment that ends at the receiver
        a, b = pts[-2], pts[-1]
        hx, hy = a[0] + 0.62 * (b[0] - a[0]), a[1] + 0.62 * (b[1] - a[1])
        tx_, ty_ = a[0] + 0.50 * (b[0] - a[0]), a[1] + 0.50 * (b[1] - a[1])
        fig.add_annotation(x=hx, y=hy, ax=tx_, ay=ty_, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2, arrowcolor=col, text="")
        # label near the middle of the path
        mid = pts[1] if reflected else ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        fig.add_annotation(x=mid[0], y=mid[1], text=f"<b>P{tm.idx + 1}</b> {tm.length_m:.1f} m", showarrow=False,
                           font=dict(color=col, size=11), bgcolor="rgba(11,18,32,0.75)", yshift=10 if not reflected else 14)
        if reflected:
            fig.add_trace(go.Scatter(x=[pts[1][0]], y=[pts[1][1]], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=7, color=col, symbol="x")))

    fig.add_trace(go.Scatter(x=[scene.tx_x], y=[scene.tx_y], mode="markers+text", text=["TX"], textposition="top center",
                             name="Transmitter", marker=dict(size=16, color=TX_COLOR, symbol="triangle-up",
                                                             line=dict(color="white", width=1)),
                             hovertemplate=f"TX ({scene.tx_x:.1f}, {scene.tx_y:.1f}) m<extra></extra>"))
    fig.add_trace(go.Scatter(x=[scene.rx_x], y=[scene.rx_y], mode="markers+text", text=["RX"], textposition="top center",
                             name="Receiver", marker=dict(size=16, color=RX_COLOR, symbol="circle",
                                                          line=dict(color="white", width=1)),
                             hovertemplate=f"RX ({scene.rx_x:.1f}, {scene.rx_y:.1f}) m<extra></extra>"))
    return fig


# --------------------------------------------------------------------------- comparison
def fig_compare(sim_a: pm.SimResult, sim_b: pm.SimResult, name_a: str, name_b: str) -> go.Figure:
    fig = _base("Received waveforms: side-by-side comparison", "Time (ns)", "Voltage (V)", height=400)
    _line(fig, sim_a.t * 1e9, sim_a.y_clean, name_a, "#2EC4B6", width=2.2)
    _line(fig, sim_b.t * 1e9, sim_b.y_clean, name_b, "#FF8A3D", width=2.2)
    return fig
