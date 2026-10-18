"""Transmitted-signal generation.

Units used everywhere in the simulation core (SI): seconds, hertz, metres.
The web interface converts to ns / MHz for display only.

Two transmit signal types are supported:

* ``"sine"``  -- a continuous wave  x(t) = A cos(2*pi*f*t + phi0)
* ``"bpsk"``  -- a digital BPSK waveform  x(t) = A * b(t) * cos(2*pi*f*t + phi0)
                 where b(t) is a +1/-1 bit envelope with smooth (raised-cosine
                 style) transitions.  The bit pattern repeats periodically, so
                 delayed copies are defined for the whole window and there is
                 no start-up transient.

Because the waveform is evaluated analytically at any time argument, a delayed
copy x(t - tau) is *exact* for any fractional delay -- no interpolation error.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

MAX_SAMPLES = 40_000          # hard cap so the browser stays responsive
MAX_FREQ_MHZ = 6000.0
MIN_FREQ_MHZ = 1.0


@dataclass
class TxConfig:
    """Transmitter / simulation-grid settings."""
    signal_type: str = "sine"        # "sine" | "bpsk"
    amplitude: float = 1.0           # peak amplitude A of the transmitted carrier [V]
    freq_mhz: float = 300.0          # carrier frequency f [MHz]
    phase_deg: float = 0.0           # initial carrier phase phi0 [deg]
    duration_ns: float = 20.0        # requested window length [ns] (rounded to whole cycles / patterns)
    samples_per_cycle: int = 64      # sampling resolution
    noise_level: float = 0.0         # noise sigma as a fraction of A (0 = noise off)
    seed: int = 7                    # random seed for repeatable noise
    bits: str = "10110010"           # BPSK bit pattern (repeats)
    cycles_per_bit: int = 4          # carrier cycles in one bit -> bit rate = f / cycles_per_bit
    edge_fraction: float = 0.25      # BPSK transition time as a fraction of a bit
    reference_mode: str = "transmitted"   # "transmitted" | "los"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def freq_hz(self) -> float:
        return self.freq_mhz * 1e6

    @property
    def is_digital(self) -> bool:
        return self.signal_type == "bpsk"

    @property
    def bit_array(self) -> np.ndarray:
        """Bit pattern as +1/-1 values."""
        return np.array([1.0 if ch == "1" else -1.0 for ch in self.bits])


def validate_tx(tx: TxConfig) -> TxConfig:
    """Clamp / repair settings so the simulation can never crash on bad input."""
    if tx.signal_type not in ("sine", "bpsk"):
        raise ValueError("signal_type must be 'sine' or 'bpsk'")
    tx.amplitude = float(np.clip(tx.amplitude, 1e-6, 1e6))
    tx.freq_mhz = float(np.clip(tx.freq_mhz, MIN_FREQ_MHZ, MAX_FREQ_MHZ))
    tx.phase_deg = float(tx.phase_deg) % 360.0
    tx.samples_per_cycle = int(np.clip(tx.samples_per_cycle, 8, 256))
    tx.noise_level = float(np.clip(tx.noise_level, 0.0, 5.0))
    tx.cycles_per_bit = int(np.clip(tx.cycles_per_bit, 1, 50))
    tx.edge_fraction = float(np.clip(tx.edge_fraction, 0.0, 1.0))
    if tx.reference_mode not in ("transmitted", "los"):
        tx.reference_mode = "transmitted"
    clean = "".join(ch for ch in str(tx.bits) if ch in "01")
    tx.bits = clean if len(clean) >= 2 else "10110010"
    tx.bits = tx.bits[:32]
    return tx


def make_time_axis(tx: TxConfig) -> tuple[np.ndarray, float, int]:
    """Return (t [s], sampling rate fs [Hz], number of carrier cycles).

    The window always holds a whole number of carrier cycles (sine) or a whole
    number of bit-pattern repetitions (BPSK).  This makes RMS values exact and
    lets the analysis treat the waveform as periodic.
    """
    f = tx.freq_hz
    spc = tx.samples_per_cycle
    if tx.is_digital:
        cycles_per_pattern = len(tx.bits) * tx.cycles_per_bit
        pattern_duration = cycles_per_pattern / f
        reps = max(1, int(round(tx.duration_ns * 1e-9 / pattern_duration)))
        n_cycles = reps * cycles_per_pattern
    else:
        n_cycles = max(2, int(round(tx.duration_ns * 1e-9 * f)))
    # keep the grid inside the sample cap by trimming whole repetitions
    unit = len(tx.bits) * tx.cycles_per_bit if tx.is_digital else 1
    while n_cycles * spc > MAX_SAMPLES and n_cycles > unit:
        n_cycles -= unit
    fs = spc * f
    n = n_cycles * spc
    t = np.arange(n) / fs
    return t, fs, n_cycles


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def envelope(t: np.ndarray, tx: TxConfig) -> np.ndarray:
    """Baseband bit envelope b(t) in [-1, 1]; ones for a sine wave."""
    if not tx.is_digital:
        return np.ones_like(t)
    bits = tx.bit_array
    n = len(bits)
    t_bit = tx.cycles_per_bit / tx.freq_hz
    period = n * t_bit
    tt = np.mod(t, period)
    k = np.floor(tt / t_bit).astype(int) % n
    u = tt - k * t_bit                         # time inside the current bit
    cur = bits[k]
    prev = bits[(k - 1) % n]
    nxt = bits[(k + 1) % n]
    tr = max(tx.edge_fraction * t_bit, 1e-18)  # transition duration, centred on bit boundaries
    level = cur.copy()
    # leading edge (first half of the transition belongs to the current bit)
    lead = u < tr / 2
    level = np.where(lead, prev + (cur - prev) * _smoothstep(0.5 + u / tr), level)
    # trailing edge
    trail = u > t_bit - tr / 2
    level = np.where(trail, cur + (nxt - cur) * _smoothstep(0.5 + (u - t_bit) / tr), level)
    return level


def tx_wave(t_env: np.ndarray, t_car: np.ndarray, tx: TxConfig,
            extra_phase_rad: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Evaluate scale * A * b(t_env) * cos(2*pi*f*t_car + phi0 + extra_phase).

    ``t_env`` and ``t_car`` are separate so the propagation model can delay the
    envelope and the carrier independently (needed by the 'simplified' phase
    mode, see propagation_model.py).
    """
    phi0 = np.deg2rad(tx.phase_deg)
    return scale * tx.amplitude * envelope(t_env, tx) * np.cos(
        2 * np.pi * tx.freq_hz * t_car + phi0 + extra_phase_rad)


def reference_wave(t: np.ndarray, tx: TxConfig) -> np.ndarray:
    """The undelayed transmitted waveform x(t)."""
    return tx_wave(t, t, tx)


def noise_sigma(tx: TxConfig) -> float:
    """Noise standard deviation per sample [V]."""
    return tx.noise_level * tx.amplitude


def make_noise(n: int, tx: TxConfig) -> np.ndarray:
    """Repeatable white Gaussian noise with sigma = noise_level * A."""
    if tx.noise_level <= 0:
        return np.zeros(n)
    rng = np.random.default_rng(tx.seed)
    return rng.normal(0.0, noise_sigma(tx), n)
