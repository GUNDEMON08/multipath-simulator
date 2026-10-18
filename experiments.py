"""The eight presentation experiments (A-H).

Every preset is just a configuration of the same mathematical model
(transmitter + paths + scene); nothing here contains pre-computed results.
Path lengths for the interference demos are built from the wavelength so that
the phase relationship between paths is exact and easy to explain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import propagation_model as pm
import signal_generator as sg

F_MHZ = 300.0
WAVELENGTH = pm.C / (F_MHZ * 1e6)       # ~0.99931 m


@dataclass
class Experiment:
    key: str
    title: str
    goal: str
    try_this: str
    why: str
    tx: sg.TxConfig
    model: pm.ModelConfig
    scene: pm.Scene
    paths: list
    focus: str = "signals"              # analysis tab to open: signals | position | phase
    sweep_path: int = 1                 # path whose phase is swept (index)
    sweep_x: tuple = (4.0, 18.0)        # receiver sweep range [m]


def _tx(**kw) -> sg.TxConfig:
    base = dict(signal_type="sine", amplitude=1.0, freq_mhz=F_MHZ, phase_deg=0.0,
                duration_ns=20.0, samples_per_cycle=64, noise_level=0.0)
    base.update(kw)
    return sg.TxConfig(**base)


def _scene() -> pm.Scene:
    return pm.Scene()


def experiment_a() -> Experiment:
    return Experiment(
        "A", "A - Single direct path",
        "Establish the reference case: one unattenuated direct path.",
        "Increase the attenuation or the length of Path 1 and watch the received amplitude fall while the shape stays the same.",
        "With a single path the receiver sees only a delayed, scaled copy of the transmitted wave, so there is nothing to interfere with.",
        _tx(), pm.ModelConfig("detailed"), _scene(),
        [pm.Path("Path 1", kind="direct", length_m=6.0, amplitude=1.0)])


def experiment_b() -> Experiment:
    l1 = 6.0
    return Experiment(
        "B", "B - Constructive interference",
        "Two paths whose lengths differ by exactly two wavelengths arrive in phase.",
        "Change Path 2's length by half a wavelength (0.5 m) and see the gain turn into a loss.",
        "A length difference of a whole number of wavelengths is a whole number of cycles, so the two waves peak together and their amplitudes add.",
        _tx(), pm.ModelConfig("detailed"), _scene(),
        [pm.Path("Path 1", kind="direct", length_m=l1, amplitude=1.0),
         pm.Path("Path 2", kind="reflected", length_m=l1 + 2 * WAVELENGTH, amplitude=0.6,
                 reflection_coeff=1.0, angle_deg=35.0)])


def experiment_c() -> Experiment:
    l1 = 6.0
    return Experiment(
        "C", "C - Destructive interference",
        "Path 2 is 1.5 wavelengths longer, so it arrives 180 degrees out of phase.",
        "Set Path 2 strength to 1.0 for ideal cancellation, then change its length by a few centimetres.",
        "Opposite phase subtracts the amplitudes: the residual is |1.0 - 0.8| = 0.2. Perfect cancellation needs equal amplitude AND exactly opposite phase (set by delay).",
        _tx(), pm.ModelConfig("detailed"), _scene(),
        [pm.Path("Path 1", kind="direct", length_m=l1, amplitude=1.0),
         pm.Path("Path 2", kind="reflected", length_m=l1 + 1.5 * WAVELENGTH, amplitude=0.8,
                 reflection_coeff=1.0, angle_deg=35.0)])


def experiment_d() -> Experiment:
    sc = pm.Scene(tx_x=2.0, tx_y=3.0, rx_x=12.0, rx_y=1.5, ceiling=False)
    return Experiment(
        "D", "D - Multipath fading",
        "Move the receiver and watch the received strength rise and fall.",
        "Move the receiver X slider in the Scene tab, then open 'Strength vs receiver position'.",
        "Moving the receiver changes the length difference between the direct and the reflected paths. Every extra wavelength of difference swings the sum from constructive to destructive.",
        _tx(), pm.ModelConfig("detailed", spreading_loss=True), sc,
        [pm.Path("LOS", kind="direct", source="direct", amplitude=1.0),
         pm.Path("Ground bounce", kind="reflected", source="ground", amplitude=1.0, reflection_coeff=-0.8),
         pm.Path("Wall bounce", kind="reflected", source="wall", amplitude=1.0, reflection_coeff=0.6)],
        focus="position", sweep_x=(4.0, 18.0))


def experiment_e() -> Experiment:
    cycles_per_bit = 4
    t_bit_ns = cycles_per_bit / F_MHZ * 1e3
    return Experiment(
        "E", "E - Delay-induced distortion (ISI)",
        "A BPSK data stream arrives together with a strong echo delayed by most of a bit.",
        "Reduce the echo's extra length until the eye reopens, or raise the noise and compare the measured and estimated BER (the window holds 32 bits).",
        "The echo of one bit lands on top of the next bit. Because its phase differs, it can partly cancel the wanted bit - this is inter-symbol interference.",
        _tx(signal_type="bpsk", duration_ns=4 * 8 * t_bit_ns, cycles_per_bit=cycles_per_bit,
            bits="10110010", noise_level=0.5, samples_per_cycle=32),
        pm.ModelConfig("detailed"), _scene(),
        [pm.Path("Direct", kind="direct", length_m=10.0, amplitude=1.0),
         pm.Path("Echo", kind="reflected", length_m=10.0 + 3.6, amplitude=0.9, reflection_coeff=1.0,
                 angle_deg=40.0)])


def experiment_f() -> Experiment:
    l1 = 6.0
    return Experiment(
        "F", "F - Attenuation of individual paths",
        "Three paths with 0, 6 and 15 dB of extra loss.",
        "Raise or lower one path's attenuation and compare the 'Individual path contributions' plot.",
        "Each path's amplitude is 10^(-dB/20): 6 dB halves it and 15 dB leaves about 18 %. A weak path barely changes the sum, a strong one dominates.",
        _tx(), pm.ModelConfig("detailed"), _scene(),
        [pm.Path("Path 1", kind="direct", length_m=l1, amplitude=1.0, attenuation_db=0.0),
         pm.Path("Path 2", kind="reflected", length_m=l1 + 1.0 * WAVELENGTH, amplitude=1.0,
                 attenuation_db=6.0, angle_deg=30.0),
         pm.Path("Path 3", kind="reflected", length_m=l1 + 2.25 * WAVELENGTH, amplitude=1.0,
                 attenuation_db=15.0, angle_deg=-50.0)])


def experiment_g() -> Experiment:
    return Experiment(
        "G", "G - Phase sweep",
        "Sweep the phase of Path 2 from 0 to 360 degrees (simplified mode: the slider is the phase).",
        "Drag Path 2's phase slider and watch the operating point move along the curve.",
        "The sum of two equal-frequency waves is a wave of amplitude sqrt(a1^2 + a2^2 + 2 a1 a2 cos(delta phi)): maximum a1 + a2 at 0 degrees, minimum |a1 - a2| at 180 degrees.",
        _tx(), pm.ModelConfig("simplified"), _scene(),
        [pm.Path("Path 1", kind="direct", length_m=6.0, amplitude=1.0),
         pm.Path("Path 2", kind="reflected", length_m=8.0, amplitude=0.7, phase_deg=60.0,
                 reflection_coeff=1.0, angle_deg=35.0)],
        focus="phase", sweep_path=1)


def experiment_h() -> Experiment:
    sc = pm.Scene()
    return Experiment(
        "H", "H - Custom experiment",
        "A free-form starting point: direct path, ground bounce and wall bounce taken from the scene.",
        "Add, edit or disable paths, move the transmitter/receiver and compare snapshots.",
        "Everything in the other experiments is a special case of this model: the received signal is the sum of delayed, scaled, phase-shifted copies.",
        _tx(), pm.ModelConfig("detailed", spreading_loss=True), sc,
        [pm.Path("LOS", kind="direct", source="direct", amplitude=1.0),
         pm.Path("Ground bounce", kind="reflected", source="ground", amplitude=1.0, reflection_coeff=-0.7),
         pm.Path("Wall bounce", kind="reflected", source="wall", amplitude=1.0, reflection_coeff=0.5)],
        focus="signals")


EXPERIMENTS = {
    "A": experiment_a, "B": experiment_b, "C": experiment_c, "D": experiment_d,
    "E": experiment_e, "F": experiment_f, "G": experiment_g, "H": experiment_h,
}


def get_experiment(key: str) -> Experiment:
    """Return a fresh copy of the preset (never share mutable state between calls)."""
    return EXPERIMENTS[key]()


def experiment_titles() -> dict:
    return {k: fn().title for k, fn in EXPERIMENTS.items()}
