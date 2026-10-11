#!/usr/bin/env python3
"""
FALCON 9 -- original score and sound design, synthesized entirely in code.

    python film/score.py --cues film/cues.json --outdir film/audio

Writes (48 kHz, 24-bit PCM, stereo, exactly `duration` seconds long):
    mix.wav             final mix (about -14 LUFS integrated, true peak <= -1 dBTP)
    music.wav           music stem
    sfx.wav             sound-design stem
    score_report.json   measurements (loudness, peaks, spectrum, stereo, event timing)

No samples and no external audio: every sound is additive / wavetable / filtered-noise
synthesis, then FFT-convolved with synthetic impulse responses.  Every time position is
derived from cues.json, so the script can simply be re-run when the edit changes.
Deterministic: all randomness comes from generators seeded by fixed names.

Music: D minor -> D major.  Motif "1 2 5 4 3" (D E A G F), answered by its sequence a third
higher (F G C Bb A); it returns in D major (with F#) at touchdown.
Only numpy and scipy are used.
"""
import argparse
import json
import os
import re
import sys
import time
import wave
import zlib

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

SR = 48000
TWO_PI = 2.0 * np.pi
TABLE_N = 4096
T_START = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T_START, msg), flush=True)


def rng_for(name):
    """Deterministic generator per named layer (stable across runs and edits)."""
    return np.random.default_rng(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)


def hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


_NOTE_INDEX = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nm(name):
    """Note name -> MIDI number: 'D4' -> 62, 'F#3' -> 54, 'Bb2' -> 46."""
    letter, rest, acc = name[0], name[1:], 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + _NOTE_INDEX[letter] + acc


def chord(*names):
    return [nm(x) for x in names]


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def undb(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 20.0)


# --------------------------------------------------------------------------------------
# cues.json
# --------------------------------------------------------------------------------------
class Cues:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            c = json.load(fh)
        self.raw = c
        self.duration = float(c.get("duration", 65.0))
        self.n = int(round(self.duration * SR))
        self.events, self.event_times, self.event_desc = {}, {}, {}
        for e in c.get("events", []):
            self.events[e["id"]] = float(e["t"])
            self.event_times[e["id"]] = [float(x) for x in e.get("times", [e["t"]])]
            self.event_desc[e["id"]] = str(e.get("desc", ""))
        self.acts = c.get("acts", [])
        self.shots = {s["id"]: (float(s["start"]), float(s["end"])) for s in c.get("shots", [])}
        pts = sorted((float(a), float(b)) for a, b in c.get("intensity", [[0.0, 0.0]]))
        if pts[-1][0] < self.duration:
            pts.append((self.duration, pts[-1][1]))
        self.int_t = np.array([p[0] for p in pts])
        self.int_v = np.array([p[1] for p in pts])

    def t(self, key, default):
        return self.events.get(key, default)

    def times(self, key, default):
        return self.event_times.get(key, default)

    def desc_number(self, key, pattern, default):
        """Pull a number out of an event description (e.g. 'clunk at 54.4', '0.15 s apart')."""
        m = re.search(pattern, self.event_desc.get(key, ""))
        return float(m.group(1)) if m else default

    def shot_start(self, sid, default):
        return self.shots.get(sid, (default, default))[0]

    def intensity(self, t):
        return np.interp(t, self.int_t, self.int_v)


class Plan:
    """Every time position of the score, derived from the cues."""

    def __init__(self, cues):
        T = cues.t
        self.cues = cues
        self.dur = cues.duration
        self.n = cues.n
        self.fade_in = T("fade_in", 0.0)
        self.s02 = cues.shot_start("s02_lattice", 5.0)
        self.s03 = cues.shot_start("s03_livery", 9.0)
        self.s04 = cues.shot_start("s04_interstage", 13.0)
        self.s05 = cues.shot_start("s05_reveal", 17.0)
        self.title = T("title_in", 19.0)
        self.riser = T("riser_peak", 23.4)
        self.ascent = T("ascent_cut", 24.0)
        self.jelly = T("jellyfish_cut", 28.0)
        self.meco = T("meco", 34.0)
        self.sep = T("separation", 35.0)
        self.mvac = T("mvac_ignition", 37.0)
        self.fairing = T("fairing_sep", 42.0)
        self.fairing_gap = cues.desc_number("fairing_sep", r"([0-9]*\.?[0-9]+)\s*s\s*apart", 0.15)
        self.ret = T("return_cut", 45.0)
        self.cold = cues.times("cold_gas", [45.4, 45.9, 46.3, 46.9, 47.3])
        self.grid = T("gridfins", 47.6)
        self.landing = T("landing_cut", 50.0)
        self.legs = T("legs_deploy", 53.0)
        self.latch = cues.desc_number("legs_deploy", r"clunk at\s*([0-9]*\.?[0-9]+)", self.legs + 1.4)
        self.touch = T("touchdown", 56.6)
        self.cutoff = T("engine_cutoff", self.touch + 0.2)
        self.s12 = cues.shot_start("s12_end", self.touch + 1.4)
        self.end_titles = T("end_titles", 59.0)
        self.fade_out = T("fade_out", self.dur - 1.0)
        self._sanitize()
        # the riser "cuts dead" at its peak; everything is silent until the hard cut
        self.riser_cut = min(self.riser, self.ascent - 0.02)

        # Act II grid: motif statements are fitted to the two Act II shots
        self.beat2a = max(0.15, (self.jelly - self.ascent) / 8.0)    # heroic statement, 8 beats
        self.beat2b = max(0.15, (self.meco - self.jelly) / 8.0)      # wonder statement, 8 beats

        # Act IV grid: an integer number of beats (~84 bpm, whole bars if possible) from the
        # return cut to touchdown, so the ostinato lands the touchdown on a downbeat.
        span = self.touch - self.ret
        target = 60.0 / 84.0
        nb = max(4, int(round(span / target)))
        nb4 = max(4, int(round(nb / 4.0)) * 4)
        if abs(span / nb4 - target) < 0.15:
            nb = nb4
        self.nbeats4 = nb
        self.beat4 = span / nb
        # four harmonic areas (bars when nb == 16)
        self.bars4 = [self.ret + round(i * nb / 4.0) * self.beat4 for i in range(5)]

    def _sanitize(self):
        """Guard against inconsistent edits: the key times must be ordered with sensible gaps.
        Only violations are touched (and reported); a consistent cues.json is used verbatim."""
        order = [("fade_in", 0.0), ("s02", 1.0), ("s03", 1.0), ("s04", 1.0), ("s05", 1.0), ("title", 0.5),
                 ("riser", 1.5), ("ascent", 0.1), ("jelly", 1.0), ("meco", 1.0), ("sep", 0.3), ("mvac", 0.3),
                 ("fairing", 0.5), ("ret", 1.0), ("landing", 1.0), ("legs", 0.5), ("touch", 0.5),
                 ("cutoff", 0.0), ("end_titles", 0.0), ("fade_out", 0.5)]
        prev = None
        for key, gap in order:
            v = getattr(self, key)
            if prev is not None and v < getattr(self, prev) + gap:
                nv = getattr(self, prev) + gap
                log("warning: cue '%s' at %.3f moved to %.3f to keep the timeline ordered" % (key, v, nv))
                setattr(self, key, nv)
            prev = key
        self.latch = min(max(self.latch, self.legs + 0.3), self.touch - 0.1)
        self.grid = min(max(self.grid, self.ret + 0.1), self.landing - 0.1)
        self.cold = [t for t in self.cold if self.ret <= t < self.landing] or [self.ret + 0.4]
        self.fade_out = min(self.fade_out, self.dur - 0.2)

    def summary(self):
        return {
            "act4_beats": self.nbeats4, "act4_beat_s": round(self.beat4, 5),
            "act4_bpm": round(60.0 / self.beat4, 3), "act4_bars": [round(b, 4) for b in self.bars4],
            "act2_beat_heroic_s": round(self.beat2a, 4), "act2_beat_wonder_s": round(self.beat2b, 4),
            "latch_t": self.latch, "fairing_gap_s": self.fairing_gap, "riser_cut_t": self.riser_cut,
        }


# --------------------------------------------------------------------------------------
# DSP primitives
# --------------------------------------------------------------------------------------
_SOS_CACHE = {}


def sos(kind, freq, order=2):
    key = (kind, tuple(np.atleast_1d(freq)), order)
    if key not in _SOS_CACHE:
        wn = np.atleast_1d(np.asarray(freq, dtype=float)) / (SR / 2.0)
        wn = np.clip(wn, 1e-5, 0.999)
        _SOS_CACHE[key] = signal.butter(order, wn if len(wn) > 1 else wn[0], btype=kind, output="sos")
    return _SOS_CACHE[key]


def filt(x, kind, freq, order=2, zero_phase=False):
    s = sos(kind, freq, order)
    return signal.sosfiltfilt(s, x, axis=-1) if zero_phase else signal.sosfilt(s, x, axis=-1)


def ramp_shape(x):
    """0..1 -> 0..1 raised-cosine (click-free fades)."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def note_env(n, attack=0.01, release=0.05, decay=None, sustain=1.0):
    """Raised-cosine attack, optional exponential decay to `sustain`, raised-cosine release
    over the last `release` seconds.  Starts and ends at zero (no clicks)."""
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / SR
    if decay:
        e = sustain + (1.0 - sustain) * np.exp(-np.maximum(t - attack, 0.0) / decay)
    else:
        e = np.ones(n)
    na = int(min(max(1, attack * SR), n))
    nr = int(min(max(1, release * SR), n))
    e[:na] *= ramp_shape(np.arange(na) / na)
    e[n - nr:] *= ramp_shape(np.arange(nr, 0, -1) / (nr + 1.0))
    return e


def gate(n_total, t_on, t_off, fade_in=0.005, fade_out=0.012):
    """Full-length gain curve: 0 before t_on, 1 between, 0 after t_off (raised-cosine edges)."""
    t = np.arange(n_total) / SR
    g = np.ones(n_total)
    if t_on is not None:
        g *= ramp_shape((t - t_on) / max(fade_in, 1e-4))
    if t_off is not None:
        g *= ramp_shape((t_off - t) / max(fade_out, 1e-4))
    return g


def smooth_noise_at(times, rate_hz, rng, ctrl=400.0):
    """Band-limited random control signal (unit std, wandering at about rate_hz), sampled at
    the given times (seconds from the start of the signal)."""
    m = int(times[-1] * ctrl) + 8
    x = rng.standard_normal(m)
    key = ("smooth", rate_hz, ctrl)
    if key not in _SOS_CACHE:
        _SOS_CACHE[key] = signal.butter(2, min(rate_hz / (ctrl / 2.0), 0.95), output="sos")
    y = signal.sosfiltfilt(_SOS_CACHE[key], x)
    y /= np.std(y) + 1e-12
    return np.interp(times, np.arange(m) / ctrl, y)


def smooth_noise(n, rate_hz, rng, ctrl=400.0):
    return smooth_noise_at(np.arange(n) / SR, rate_hz, rng, ctrl)


def white(n, rng):
    return rng.standard_normal(n)


def pink(n, rng):
    """Pink noise (Paul Kellet's -3 dB/oct filter), normalised to unit RMS."""
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    y = signal.lfilter(b, a, rng.standard_normal(n + 4800))[4800:]
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def brown(n, rng, hp=18.0):
    """Leaky-integrated (red) noise, high-passed, unit RMS."""
    y = signal.lfilter([1.0], [1.0, -0.9985], rng.standard_normal(n + 9600))[9600:]
    y = filt(y, "highpass", hp, 2)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def soft_clip(x, thr):
    """Gentle tanh peak shaping above ~thr (used on noise layers to tame crest factor)."""
    return thr * np.tanh(x / thr)


def high_shelf(x, f0, gain_db, slope=0.7):
    """RBJ high-shelf biquad."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = TWO_PI * f0 / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    sa = 2.0 * np.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
    a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=-1)


def peaking_eq(x, f0, gain_db, q=0.7):
    """RBJ peaking biquad (used to carve room for the music inside the roar)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = TWO_PI * f0 / SR
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return signal.lfilter(b / a[0], a / a[0], x, axis=-1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def norm_rms(x, target=1.0):
    return x * (target / rms(x))


def pan_gains(p):
    """Constant-power amplitude pan, p in [-1, 1] (same phase in both channels)."""
    a = (np.clip(p, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.cos(a), np.sin(a)


def to_stereo(mono, p=0.0):
    gl, gr = pan_gains(p)
    return np.vstack([mono * gl, mono * gr])


def corr_stereo(common, side_l, side_r, rho):
    """Two channels with correlation ~rho built from a common and two independent parts."""
    a, b = np.sqrt(rho), np.sqrt(max(1.0 - rho, 0.0))
    return np.vstack([a * common + b * side_l, a * common + b * side_r])


class Bus:
    """A stereo track spanning the whole film."""

    def __init__(self, n, name):
        self.name = name
        self.x = np.zeros((2, n))

    def add(self, sig, t0, gain=1.0):
        i0 = int(round(t0 * SR))
        if sig.ndim == 1:
            sig = to_stereo(sig, 0.0)
        n = sig.shape[1]
        a, b = max(i0, 0), min(i0 + n, self.x.shape[1])
        if b > a:
            self.x[:, a:b] += gain * sig[:, a - i0:b - i0]


# --------------------------------------------------------------------------------------
# wavetable oscillators (band-limited, phase-accumulating, crossfade between two spectra)
# --------------------------------------------------------------------------------------
def soft_lp(f, fc, order=2):
    return 1.0 / np.sqrt(1.0 + (np.asarray(f) / fc) ** (2 * order))


def make_table(amps, phases):
    """Band-limited single-cycle table, RMS-normalised like a unit sine.
    Returned as (values, forward differences) for fast linear interpolation."""
    amps = np.asarray(amps, dtype=float)
    ph = TWO_PI * np.arange(TABLE_N) / TABLE_N
    tab = np.zeros(TABLE_N)
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        if a > 1e-5:
            tab += a * np.sin(k * ph + p)
    tab /= np.sqrt(np.mean(tab ** 2)) * np.sqrt(2.0) + 1e-12
    return tab, np.append(tab[1:], tab[0]) - tab


def table_read(ta, tb, mix, ph):
    """Linear-interpolated lookup of phase (in cycles) in table ta, crossfaded to tb by mix."""
    ph = ph - np.floor(ph)
    idx = ph * TABLE_N
    i0 = idx.astype(np.int64)
    fr = idx - i0
    a = ta[0][i0] + ta[1][i0] * fr
    if tb is None:
        return a
    b = tb[0][i0] + tb[1][i0] * fr
    return a + (b - a) * mix


def osc_mix(ta, tb, mix, freq, n, phase0=0.0):
    """Read table ta (and tb, crossfaded by `mix`) at a fixed frequency."""
    return table_read(ta, tb, mix, phase0 + (freq / SR) * np.arange(n))


CTRL = 32   # pitch modulation is computed every 32 samples (0.67 ms) and its phase interpolated


def ensemble(f0, n, ta, tb, mix, rng, voices=5, detune=7.0, vib_rate=5.2, vib_depth=6.0,
             vib_delay=0.5, drift=2.0, width=0.6, pan=0.0):
    """Detuned ensemble of `voices` oscillators with delayed vibrato and slow pitch drift,
    each voice amplitude-panned across the stereo field.  Returns (2, n)."""
    out = np.zeros((2, n))
    m = n // CTRL + 2
    tc = np.arange(m) * (CTRL / SR)
    vib_env = ramp_shape(tc / max(2.0 * vib_delay, 1e-3)) if vib_delay > 0 else 1.0
    spread = np.linspace(-1.0, 1.0, voices) if voices > 1 else np.zeros(1)
    pans = pan + width * spread[rng.permutation(voices)]
    ia = np.arange(n)
    ic = ia // CTRL
    fc = (ia % CTRL) / CTRL
    for v in range(voices):
        cents = detune * spread[v] + rng.normal(0.0, 0.2 * detune + 0.3)
        cents = cents + drift * smooth_noise_at(tc, 0.25, rng)
        if vib_depth > 0:
            r = vib_rate * (1.0 + rng.normal(0.0, 0.06))
            cents = cents + vib_depth * vib_env * np.sin(TWO_PI * r * tc + rng.uniform(0, TWO_PI))
        f = f0 * np.exp(cents * (np.log(2.0) / 1200.0))
        phc = rng.uniform() + np.concatenate([[0.0], np.cumsum(f[:-1]) * (CTRL / SR)])
        dph = np.diff(phc)
        sig = table_read(ta, tb, mix, phc[ic] + dph[ic] * fc)
        gl, gr = pan_gains(pans[v])
        out[0] += gl * sig
        out[1] += gr * sig
    return out / np.sqrt(voices)


def modal(n, modes):
    """Sum of exponentially decaying sinusoids, each the impulse response of a two-pole
    resonator (A r^k sin(theta k + phi)); every mode is only computed until it has decayed
    by ~70 dB.  modes: iterable of (freq_hz, amp, t60_s, phase)."""
    out = np.zeros(n)
    for f, a, t60, ph in modes:
        if f <= 0 or f >= 0.45 * SR or a == 0:
            continue
        L = int(min(n, t60 * 1.18 * SR + 64))
        th = TWO_PI * f / SR
        r = 10.0 ** (-3.0 / (t60 * SR))
        imp = np.zeros(L)
        imp[0] = 1.0
        out[:L] += signal.lfilter([a * np.sin(ph), a * r * np.sin(th - ph)],
                                  [1.0, -2.0 * r * np.cos(th), r * r], imp)
    return out


# --------------------------------------------------------------------------------------
# time-varying spectral filter (STFT) and reverb
# --------------------------------------------------------------------------------------
def stft_filter(x, gain_fn, t0=0.0, nper=2048):
    """Filter mono x with a time-varying magnitude response gain_fn(f[:,None], t_abs[None,:])."""
    nov = nper * 3 // 4
    f, tt, Z = signal.stft(x, fs=SR, window="hann", nperseg=nper, noverlap=nov)
    G = gain_fn(f[:, None], t0 + tt[None, :])
    _, y = signal.istft(Z * G, fs=SR, window="hann", nperseg=nper, noverlap=nov)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def lp_response(f, fc, order=2):
    return soft_lp(f, np.maximum(fc, 1.0), order)


def make_ir(name, length=4.5, rt_low=3.0, rt_mid=3.6, rt_high=1.4, predelay=0.022,
            early=0.35, lp=7000.0, hp=60.0):
    """Synthetic stereo impulse response: decorrelated noise per channel, split into bands
    that each decay with their own RT60 (bright frequencies die first), a soft diffuse
    onset, a handful of early reflections, normalised to unit energy per channel."""
    rng = rng_for("ir:" + name)
    n = int(length * SR)
    t = np.arange(n) / SR
    edges = [30, 120, 300, 700, 1500, 3000, 6000, 11000]
    ir = np.zeros((2, n))
    for ch in range(2):
        w = rng.standard_normal(n)
        acc = np.zeros(n)
        for lo, hi in zip(edges[:-1], edges[1:]):
            fc = np.sqrt(lo * hi)
            lf = np.log10(fc)
            rt = np.interp(lf, [np.log10(100), np.log10(1000), np.log10(8000)], [rt_low, rt_mid, rt_high])
            band = signal.sosfilt(sos("bandpass", [lo, hi], 2), w)
            acc += band * 10.0 ** (-3.0 * t / rt)
        acc *= ramp_shape(t / 0.06) ** 0.7          # diffuse build-up
        # early reflections (different per channel)
        er = np.zeros(n)
        for k in range(10):
            d = rng.uniform(0.006, 0.075)
            g = early * (0.85 ** k) * rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            er[int(d * SR)] += g
        er = filt(er, "lowpass", 5000.0, 2)
        acc = acc / (np.sqrt(np.sum(acc ** 2)) + 1e-12) + er * 0.08
        acc = filt(acc, "lowpass", lp, 2)
        acc = filt(acc, "highpass", hp, 2)
        acc = acc * ramp_shape((length - t) / 0.25)  # no truncation click
        pd = int(predelay * SR)
        ir[ch, pd:] = acc[:n - pd]
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True)) + 1e-12
    return ir


def reverb_bus(bus, ir, send, t_from=0.0, t_to=None, send_hp=110.0, send_lp=9000.0):
    """Add a convolution-reverb return to `bus` (mono send -> decorrelated stereo IR).
    Only the active region [t_from, t_to] is convolved; the tail runs on after it."""
    n_total = bus.x.shape[1]
    a = max(0, int(t_from * SR))
    b = n_total if t_to is None else min(n_total, int(t_to * SR))
    if b <= a:
        return
    send_sig = 0.5 * (bus.x[0, a:b] + bus.x[1, a:b])
    if not np.any(send_sig):
        return
    send_sig = filt(send_sig, "highpass", send_hp, 2)
    send_sig = filt(send_sig, "lowpass", send_lp, 2)
    for ch in range(2):
        wet = signal.oaconvolve(send_sig, ir[ch])
        e = min(n_total, a + len(wet))
        bus.x[ch, a:e] += send * wet[:e - a]


# --------------------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------------------
_TAB_CACHE = {}


def _cached(key, build):
    if key not in _TAB_CACHE:
        _TAB_CACHE[key] = build()
    return _TAB_CACHE[key]


def _harm(f0, fmax):
    return np.arange(1, max(1, int(fmax / f0)) + 1)


def _phases(key, count):
    return rng_for("phase:" + key).uniform(0.0, TWO_PI, count)


def string_tables(midi):
    """Bowed-string ensemble spectrum (1/k, wooden body bumps), dark and bright versions."""
    def build():
        f0 = float(hz(midi))
        k = _harm(f0, 7000.0)
        f = k * f0
        body = 1.0 + 0.45 * np.exp(-np.log2(f / 290.0) ** 2 / 0.35) + 0.3 * np.exp(-np.log2(f / 1150.0) ** 2 / 0.5)
        ph = _phases("str%d" % midi, len(k))
        dark = (1.0 / k) * soft_lp(f, 850.0, 2) * body
        bright = (1.0 / k) * soft_lp(f, 4200.0, 2) * body
        return make_table(dark, ph), make_table(bright, ph)
    return _cached(("str", midi), build)


def organ_tables(midi):
    """Pipe-organ-like: soft flute stop (dark) and principal + mixture (bright)."""
    def build():
        f0 = float(hz(midi))
        k = _harm(f0, 6500.0)
        f = k * f0
        draw = {1: 1.0, 2: 0.6, 3: 0.32, 4: 0.3, 5: 0.07, 6: 0.15, 8: 0.11, 10: 0.04, 12: 0.05, 16: 0.03}
        flute = {1: 1.0, 2: 0.22, 3: 0.07, 4: 0.04}
        ph = _phases("org%d" % midi, len(k))
        a = np.array([draw.get(int(x), 0.0) for x in k]) * soft_lp(f, 3600.0, 2)
        b = np.array([flute.get(int(x), 0.0) for x in k])
        return make_table(b, ph), make_table(a, ph)
    return _cached(("org", midi), build)


FORMANTS = {
    "oo": [(320, 110, 1.0), (800, 140, 0.32), (2300, 240, 0.07), (3200, 400, 0.03)],
    "oh": [(500, 120, 1.0), (850, 140, 0.55), (2450, 250, 0.12), (3300, 400, 0.04)],
    "ah": [(720, 140, 1.0), (1100, 170, 0.55), (2500, 260, 0.2), (3400, 380, 0.06)],
}


def choir_tables(midi, v1="oo", v2="ah"):
    """Formant-shaped harmonic spectra for a wordless choir; crossfade = vowel morph."""
    def build():
        f0 = float(hz(midi))
        k = _harm(f0, 7000.0)
        f = k * f0
        ph = _phases("ch%d" % midi, len(k))
        out = []
        for v in (v1, v2):
            res = 0.03 + sum(a / (1.0 + ((f - F) / (bw / 2.0)) ** 2) for F, bw, a in FORMANTS[v])
            out.append(make_table(k ** -1.1 * res, ph))
        return tuple(out)
    return _cached(("ch", midi, v1, v2), build)


def horn_tables(midi):
    """Horn section: mellow (few harmonics) to brassy (brass formant near 1.1 kHz)."""
    def build():
        f0 = float(hz(midi))
        k = _harm(f0, 6000.0)
        f = k * f0
        ph = _phases("hn%d" % midi, len(k))
        dark = np.exp(-(k - 1) * 0.95) * soft_lp(f, 900.0, 2)
        bright = k ** -0.6 * soft_lp(f, 2900.0, 3) * (1.0 + 0.6 * np.exp(-np.log2(f / 1100.0) ** 2 / 0.4))
        return make_table(dark, ph), make_table(bright, ph)
    return _cached(("hn", midi), build)


def _mix_curve(bright, t_abs):
    return bright(t_abs) if callable(bright) else bright


def sustained(kind, midi, t0, dur, amp, rng, attack=0.5, release=1.0, bright=0.4, voices=5,
              pan=0.0, width=0.6, vib_depth=None, detune=None, dyn=None, tremolo=None,
              v1="oo", v2="ah"):
    """Sustained ensemble note.  kind: strings | organ | choir | horns.
    bright: scalar or f(t_abs) (spectral crossfade), dyn: optional f(t_abs) gain,
    tremolo: optional (rate_fn(t_abs), depth) amplitude tremolo."""
    f0 = float(hz(midi))
    n = int((dur + release) * SR)
    t_abs = t0 + np.arange(n) / SR
    if f0 < 90.0:
        amp = amp * (f0 / 90.0) ** 0.8          # keep the low end controlled
    if kind == "strings":
        ta, tb = string_tables(midi)
        vd, dt, vr = 7.0, 7.0, 5.3
    elif kind == "organ":
        ta, tb = organ_tables(midi)
        vd, dt, vr = 0.0, 3.0, 0.0
    elif kind == "choir":
        ta, tb = choir_tables(midi, v1, v2)
        vd, dt, vr = 14.0, 9.0, 5.0
    else:
        ta, tb = horn_tables(midi)
        vd, dt, vr = 3.0, 5.0, 4.6
    vd = vd if vib_depth is None else vib_depth
    dt = dt if detune is None else detune
    mix = _mix_curve(bright, t_abs)
    if kind == "horns" and not callable(bright):
        # brass brightens with loudness: tie the spectrum to the attack envelope
        mix = bright * (0.35 + 0.65 * note_env(n, attack * 0.7, release * 1.2))
    sig = ensemble(f0, n, ta, tb, mix, rng, voices, dt, vr if vr > 0 else 5.0, vd, 0.45, 1.6, width, pan)
    e = note_env(n, attack, release) * amp
    if dyn is not None:
        e = e * dyn(t_abs)
    if tremolo is not None:
        rate_fn, depth = tremolo
        ph = np.cumsum(rate_fn(t_abs)) / SR
        e = e * (1.0 - depth * (0.5 - 0.5 * np.cos(TWO_PI * ph)))
    return sig * e


def piano_note(midi, vel, rng, dur=6.0, bright=0.5):
    """Felt piano: stretched (inharmonic) partials, two slightly detuned strings per note,
    two-stage decay (modal resonators), soft felt-hammer thump.  Peak ~ vel."""
    f0 = float(hz(midi))
    n = int(dur * SR)
    B = 0.00008 * (f0 / 261.6) ** 0.8 + 0.00002
    soft = 0.8 - 0.45 * bright * vel
    modes = []
    for k in range(1, 16):
        fk = k * f0 * np.sqrt(1.0 + B * k * k)
        if fk > 6500.0:
            break
        a = np.exp(-(k - 1) * soft) / k
        t60 = 7.0 * (261.6 / f0) ** 0.45 / (1.0 + 0.5 * (k - 1))
        f2 = fk * 2.0 ** (rng.normal(0.0, 0.35) / 1200.0)
        p1, p2 = rng.uniform(0.0, TWO_PI, 2)
        for ff, aa, pp in ((fk, a, p1), (f2, 0.85 * a, p2)):
            modes.append((ff, 0.55 * aa, t60, pp))
            modes.append((ff, 0.45 * aa, 0.2 * t60, pp))
    out = modal(n, modes)
    nh = int(0.04 * SR)
    th = filt(rng.standard_normal(nh), "lowpass", 500.0 + 900.0 * vel, 2) * np.exp(-np.arange(nh) / (0.007 * SR))
    out /= np.max(np.abs(out)) + 1e-9
    out[:nh] += 0.06 * th / (np.max(np.abs(th)) + 1e-9)
    out *= note_env(n, 0.006, 0.12)
    return out * vel


def bell_note(midi, vel, rng, dur=4.5):
    """Glass/celesta bell: a few slightly inharmonic partials, each with its own decay."""
    f0 = float(hz(midi))
    n = int(dur * SR)
    scale = (523.0 / f0) ** 0.3
    modes = [(f0 * r, a, d * scale, rng.uniform(0, TWO_PI))
             for r, a, d in ((1.0, 1.0, 3.4), (2.0, 0.3, 1.7), (3.0, 0.08, 1.0), (4.07, 0.14, 0.75),
                             (5.19, 0.05, 0.5), (6.6, 0.025, 0.35)) if f0 * r <= 7000.0]
    out = modal(n, modes)
    out /= np.max(np.abs(out)) + 1e-9
    out *= note_env(n, 0.004, 0.1)
    return out * vel


def pluck_note(midi, vel, rng, decay=0.25, bright=0.5):
    """Spiccato/pizzicato-like ensemble pluck for the ostinato."""
    f0 = float(hz(midi))
    n = int(min(6.0 * decay + 0.05, 2.0) * SR)
    t = np.arange(n) / SR
    ta, tb = string_tables(midi)
    sig = np.zeros(n)
    for c in (-4.0, 3.5):
        sig += osc_mix(ta, tb, bright, f0 * 2.0 ** (c / 1200.0), n, rng.uniform())
    # the brightness itself decays quickly (string damping)
    sig_dark = osc_mix(ta, None, 0.0, f0, n, rng.uniform())
    damp = np.exp(-t / (decay * 0.35))
    sig = 0.5 * sig * (0.35 + 0.65 * damp) + 0.4 * sig_dark * (1.0 - 0.6 * damp)
    env = np.exp(-t / decay) * note_env(n, 0.003, 0.03)
    return sig * env * vel


def drone(midis, dur, amp, rng, harm=10, lp=700.0, attack=4.0, release=2.0, top=2500.0, sub=0.0):
    """Organ-like breathing drone; harmonics drift slowly in level.  The lowest harmonics are
    identical in both channels (mono-safe), upper ones breathe independently L/R."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    tc = np.arange(n // CTRL + 2) * (CTRL / SR)
    out = np.zeros((2, n))
    for m in midis:
        f0 = float(hz(m))
        for k in range(1, harm + 1):
            fk = k * f0
            if fk > top:
                break
            a = k ** -1.2 * soft_lp(fk, lp, 2)
            s = np.sin(TWO_PI * fk * t + rng.uniform(0, TWO_PI))
            ml = np.interp(t, tc, np.exp(0.3 * smooth_noise_at(tc, 0.1, rng)))
            mr = ml if k <= 3 else np.interp(t, tc, np.exp(0.3 * smooth_noise_at(tc, 0.1, rng)))
            out[0] += a * ml * s
            out[1] += a * mr * s
    if sub > 0:
        out += sub * np.sin(TWO_PI * 0.5 * float(hz(midis[0])) * t + rng.uniform(0, TWO_PI))
    out /= np.max(np.abs(out)) + 1e-9
    return out * note_env(n, attack, release) * amp


def boom(dur, f_hi, f_lo, glide, decay, rng, noise=0.3, noise_lp=180.0, attack=0.003):
    """Pitched sub impact: sine with exponential pitch drop plus a low noise thump."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f_lo + (f_hi - f_lo) * np.exp(-t / glide)
    body = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-t / decay)
    nz = norm_rms(filt(rng.standard_normal(n), "lowpass", noise_lp, 2)) * np.exp(-t / (0.3 * decay))
    out = body + noise * 0.5 * nz
    return out * note_env(n, attack, 0.05)


def taiko(vel, rng, f0=73.42, dur=1.6):
    """Big drum: membrane boom + skin slap."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = boom(dur, f0 * 2.0, f0, 0.035, 0.42, rng, noise=0.45, noise_lp=260.0)
    skin = filt(rng.standard_normal(n), "bandpass", [180.0, 1100.0], 2) * np.exp(-t / 0.045)
    out = out + 0.22 * skin / (np.max(np.abs(skin)) + 1e-9) * note_env(n, 0.002, 0.05)
    return out / (np.max(np.abs(out)) + 1e-9) * vel


def timpani_stroke(f0, vel, rng, dur=1.8):
    """Timpani: membrane modes (ratios of a kettle drum) + soft mallet noise."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = modal(n, [(f0 * r, a, 6.91 * d, rng.uniform(0, TWO_PI))
                    for r, a, d in ((1.0, 1.0, 1.3), (1.504, 0.5, 0.9), (1.742, 0.3, 0.75), (2.0, 0.22, 0.6),
                                    (2.245, 0.12, 0.45), (2.494, 0.08, 0.35), (0.6, 0.25, 0.25))])
    m = filt(rng.standard_normal(n), "lowpass", 700.0, 2) * np.exp(-t / 0.012)
    out = out / (np.max(np.abs(out)) + 1e-9) + 0.15 * m / (np.max(np.abs(m)) + 1e-9)
    return out * note_env(n, 0.003, 0.05) * vel


def tam(dur, rng, f_lo=90.0, f_hi=2400.0, count=48, bloom=0.5, decay=5.0):
    """Tam-tam / gong bloom: dense inharmonic partials, higher ones bloom later, die sooner.
    (1 - e^-t/att) e^-t/dec is the difference of two decaying modes."""
    n = int(dur * SR)
    out = np.zeros((2, n))
    for _ in range(count):
        f = np.exp(rng.uniform(np.log(f_lo), np.log(f_hi)))
        a = (f / 200.0) ** -0.55 * rng.uniform(0.4, 1.0)
        pos = np.log(f / f_lo) / np.log(f_hi / f_lo)
        att = bloom * (0.15 + 0.85 * pos)
        dec = decay * (200.0 / f) ** 0.3
        ph = rng.uniform(0, TWO_PI)
        s = modal(n, [(f, a, 6.91 * dec, ph), (f, -a, 6.91 / (1.0 / att + 1.0 / dec), ph)])
        gl, gr = pan_gains(rng.uniform(-0.6, 0.6))
        out[0] += gl * s
        out[1] += gr * s
    out /= np.max(np.abs(out)) + 1e-9
    return out * note_env(n, 0.005, 0.6)


def shimmer(dur, notes, rng, grain_rate=1.6, grain_notes=None, grain_level=0.5):
    """Faint high halo: slow-swelling sine partials plus sparse soft sparkle grains."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for m in notes:
        f = float(hz(m))
        for ratio, a in ((1.0, 1.0), (2.0, 0.18)):
            fk = f * ratio
            if fk > 7500.0:
                continue
            sw = 0.5 + 0.5 * np.sin(TWO_PI * rng.uniform(0.06, 0.16) * t + rng.uniform(0, TWO_PI))
            s = a * (0.25 + 0.75 * sw ** 2) * np.sin(TWO_PI * fk * t + rng.uniform(0, TWO_PI))
            gl, gr = pan_gains(rng.uniform(-0.7, 0.7))
            out[0] += gl * s
            out[1] += gr * s
    out /= len(notes)
    if grain_notes:
        count = int(grain_rate * dur)
        for _ in range(count):
            t0 = rng.uniform(0.0, max(dur - 1.0, 0.1))
            m = grain_notes[rng.integers(len(grain_notes))]
            g = bell_note(m, grain_level * rng.uniform(0.4, 1.0), rng, dur=1.6)
            i0 = int(t0 * SR)
            gl, gr = pan_gains(rng.uniform(-0.8, 0.8))
            e = min(n, i0 + len(g))
            out[0, i0:e] += gl * g[:e - i0]
            out[1, i0:e] += gr * g[:e - i0]
    return out


# --------------------------------------------------------------------------------------
# sound-design generators
# --------------------------------------------------------------------------------------
def crackle(n, rng, rate, density=None, alpha=2.3, decay_ms=1.0, band=(450.0, 5200.0), pos_frac=0.85):
    """Rocket crackle: an inhomogeneous Poisson train of heavy-tailed, mostly positive
    (shock-like) pressure spikes with ~1 ms exponential tails, band-limited."""
    if density is None:
        density = np.ones(n)
    dmax = float(np.max(density))
    count = rng.poisson(rate * dmax * n / SR)
    pos = rng.integers(0, n, count)
    pos = pos[rng.random(count) < density[pos] / dmax]
    amps = np.minimum(rng.pareto(alpha, len(pos)) + 1.0, 8.0)
    sign = np.where(rng.random(len(pos)) < pos_frac, 1.0, -0.6)
    imp = np.zeros(n)
    np.add.at(imp, pos, amps * sign)
    a = np.exp(-1.0 / (decay_ms * 1e-3 * SR))
    y = signal.lfilter([1.0 - a], [1.0, -a], imp)
    y = filt(y, "bandpass", list(band), 2)
    return norm_rms(y) if np.any(y) else y


def roar_layers(n, name, crackle_rate=110.0, pop_rate=16.0):
    """Layered rocket-engine noise.  Returns stereo (2, n) layers, each ~unit RMS:
    rumble (deep, slow-modulated), body (turbulent mid roar), crackle, hiss.
    Slow loudness modulation is common to both channels; fine structure is decorrelated."""
    rng = rng_for(name)
    m_slow = np.exp(0.16 * smooth_noise(n, 0.45, rng) + 0.07 * smooth_noise(n, 2.5, rng))
    m_turb = np.exp(0.14 * smooth_noise(n, 1.3, rng) + 0.09 * smooth_noise(n, 8.0, rng))
    dens = np.exp(1.1 * smooth_noise(n, 2.2, rng))

    def rumble1():
        return filt(filt(brown(n, rng), "lowpass", 120.0, 4), "highpass", 35.0, 2)

    def body1():
        return filt(pink(n, rng), "bandpass", [85.0, 1300.0], 2)

    def crack1():
        fine = crackle(n, rng, crackle_rate, dens, alpha=2.8, decay_ms=0.8, band=(650.0, 5000.0))
        pops = crackle(n, rng, pop_rate, dens, alpha=2.4, decay_ms=3.0, band=(180.0, 2400.0))
        return soft_clip(norm_rms(fine + 0.8 * pops), 3.0)

    def hiss1():
        return filt(filt(white(n, rng), "bandpass", [1800.0, 6000.0], 2), "lowpass", 6500.0, 2)

    out = {}
    for key, mk, rho, mod in (("rumble", rumble1, 0.92, m_slow), ("body", body1, 0.75, m_turb),
                              ("crackle", crack1, 0.5, m_slow), ("hiss", hiss1, 0.4, m_turb)):
        st = corr_stereo(mk(), mk(), mk(), rho) * mod
        out[key] = st / (rms(st) + 1e-12)
    return out


def thud(rng, f_hi=60.0, f_lo=37.0, decay=0.32, modes=(), lp=320.0, dur=1.8, noise=0.5):
    """Low, muffled, structure-borne impact."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = boom(dur, f_hi, f_lo, 0.06, decay, rng, noise=noise, noise_lp=170.0)
    for f, a, d in modes:
        out += a * np.exp(-t / d) * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
    out = filt(out * note_env(n, 0.003, 0.08), "lowpass", lp, 2)
    return out / (np.max(np.abs(out)) + 1e-9)


def clunk(rng, thump_hi=120.0, thump_lo=60.0, thump_decay=0.12, modes=(), lp=3200.0, dur=1.2,
          click=0.35):
    """Mechanical latch: thump + damped metal modes + a short click."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = boom(dur, thump_hi, thump_lo, 0.02, thump_decay, rng, noise=0.7, noise_lp=320.0, attack=0.002)
    for f, a, d in modes:
        out += a * np.exp(-t / d) * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
    ck = filt(rng.standard_normal(n), "bandpass", [900.0, 4000.0], 2) * np.exp(-t / 0.004)
    out += click * ck / (np.max(np.abs(ck)) + 1e-9)
    out = filt(out * note_env(n, 0.0015, 0.08), "lowpass", lp, 2)
    return out / (np.max(np.abs(out)) + 1e-9)


def noise_burst(dur, rng, band, attack, decay, release=0.08, flutter=0.12, low=0.3):
    """Gas release: band-passed hiss with a little low-frequency 'pff' and flutter."""
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    h = filt(filt(white(n, rng), "bandpass", list(band), 2), "lowpass", 6000.0, 2)
    lo = filt(white(n, rng), "bandpass", [220.0, 900.0], 2)
    sig = norm_rms(h) + low * norm_rms(lo)
    sig *= np.exp(flutter * smooth_noise(n, 22.0, rng))
    env = np.exp(-np.maximum(t - attack, 0.0) / decay)
    env *= ramp_shape(t / attack) * ramp_shape((dur + release - t) / release)
    return sig * env


def puff(dur, rng):
    return noise_burst(dur, rng, (900.0, 6000.0), 0.012, dur * 1.4, release=0.07, flutter=0.15, low=0.35)


def impact(rng, dur=3.0):
    """Touchdown: sub thump, ringing steel-deck modes, crush and a little debris rattle."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = 0.9 * boom(dur, 58.0, 36.0, 0.08, 0.5, rng, noise=0.6, noise_lp=150.0)
    for f, a, d in ((61, .6, .5), (97, .45, .4), (143, .4, .32), (211, .28, .24), (289, .2, .18),
                    (377, .14, .14), (520, .09, .1), (760, .06, .07)):
        out += a * np.exp(-t / d) * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI)) * note_env(n, 0.003, 0.1)
    crush = filt(rng.standard_normal(n), "bandpass", [250.0, 1800.0], 2) * np.exp(-t / 0.09)
    out += 0.45 * crush / (np.max(np.abs(crush)) + 1e-9) * note_env(n, 0.002, 0.1)
    rattle = crackle(n, rng, 60.0, np.exp(-t / 0.25), alpha=2.0, decay_ms=1.5, band=(700.0, 3500.0))
    out += 0.12 * rattle * np.exp(-t / 0.35)
    out = filt(out, "lowpass", 4000.0, 2)
    return out / (np.max(np.abs(out)) + 1e-9)


def steam(dur, rng, hold=1.1, tau=1.6):
    """Steam / vapour hiss after engine cutoff (gusty, slowly decaying)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    h = filt(filt(white(n, rng), "bandpass", [650.0, 5200.0], 2), "lowpass", 5500.0, 2)
    lo = filt(white(n, rng), "bandpass", [140.0, 600.0], 2)
    sig = norm_rms(h) * np.exp(0.35 * smooth_noise(n, 1.4, rng)) + 0.45 * norm_rms(lo) * np.exp(0.3 * smooth_noise(n, 0.8, rng))
    env = np.exp(-np.maximum(t - hold, 0.0) / tau) * note_env(n, 0.22, 0.6)
    return sig * env


def sea(dur, rng):
    """Gentle sea wash under the droneship shots (stereo, slow swells)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    chans = []
    base = filt(pink(n, rng), "lowpass", 1100.0, 2)
    for ch in range(2):
        w = 0.6 * base + 0.8 * filt(pink(n, rng), "lowpass", 1100.0, 2)
        w = filt(w, "highpass", 90.0, 2)
        sw = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(TWO_PI * 0.105 * t + ch * 1.3 + rng.uniform(0, 1)))
        chans.append(norm_rms(w) * sw * np.exp(0.2 * smooth_noise(n, 0.35, rng)))
    return np.vstack(chans)


# --------------------------------------------------------------------------------------
# composition helpers
# --------------------------------------------------------------------------------------
MOTIF_BEATS = (1.0, 1.0, 2.5, 0.5, 3.0)


def motif_line(root, last, beats=MOTIF_BEATS):
    """The motif '1 2 5 4 3': root, +2, +7, +5, +last (3 = minor third, 4 = major third)."""
    return list(zip([root, root + 2, root + 7, root + 5, root + last], beats))


def ifn(cues, lo_db, hi_db=0.0):
    """Gain curve from the cue intensity: intensity 0 -> lo_db, 1 -> hi_db."""
    return lambda t: undb(lo_db + (hi_db - lo_db) * cues.intensity(t))


def chord_pad(bus, kind, notes, t0, dur, amp, rng, **kw):
    for m in notes:
        bus.add(sustained(kind, m, t0, dur, amp, rng, **kw), t0)


def line_sus(bus, kind, line, t0, beat, amp, rng, legato=1.0, overlap=0.0, **kw):
    """Melody on a sustained instrument; line = [(midi, beats), ...]."""
    t = t0
    for m, b in line:
        d = b * beat
        if m is not None:
            bus.add(sustained(kind, m, t, d * legato + overlap, amp, rng, **kw), t)
        t += d
    return t


def line_keys(bus, fn, line, t0, beat, vel, rng, pan=0.0, ring=1.1, last_ring=4.0, **kw):
    """Melody on a struck instrument (piano/bell).  Each note is damped `ring` seconds after
    the next one starts (like a pedal change); the last note of the phrase rings on."""
    t = t0
    for i, (m, b) in enumerate(line):
        if m is not None:
            dur = b * beat + (last_ring if i == len(line) - 1 else ring)
            bus.add(to_stereo(fn(m, vel, rng, dur=min(dur, 6.0), **kw), pan), t)
        t += b * beat
    return t


def with_dyn(sig, t0, fn):
    t = t0 + np.arange(sig.shape[-1]) / SR
    return sig * fn(t)


def noise_riser(t0, t1, rng, f_lo=350.0, f_hi=5200.0, floor_db=-30.0, curve=2.2):
    """Pink noise through a rising one-octave band, swelling exponentially to its peak at t1."""
    n = int((t1 - t0) * SR)
    x = pink(n, rng)

    def g(f, t):
        p = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        fc = np.exp(np.log(f_lo) + (np.log(f_hi) - np.log(f_lo)) * p ** 1.4)
        band = np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / fc) / 0.55) ** 2)
        return band * undb(floor_db * (1.0 - p ** (1.0 / curve)))
    y = stft_filter(x, g, t0)
    return y * note_env(n, 0.3, 0.004)


# --------------------------------------------------------------------------------------
# the music
# --------------------------------------------------------------------------------------
def compose_act1(P, C, bus):
    rng = rng_for("act1")
    I = C.intensity
    a0, s02, s03, s04, s05, title, cut = P.fade_in, P.s02, P.s03, P.s04, P.s05, P.title, P.riser_cut
    tr = title + 0.45 * (cut - title)          # riser / dominant starts (~21 s)

    # drone: D pedal fades up from silence, hands over to A for the riser
    dyn = lambda t: 0.35 + 0.65 * np.minimum(I(t) / 0.6, 1.0)
    d1 = drone([nm("D2"), nm("A2")], tr + 1.2 - a0, 0.15, rng, harm=12, lp=650.0, attack=5.5, release=1.8,
               sub=0.25)
    bus.add(with_dyn(d1, a0, dyn), a0)
    d2 = drone([nm("A2"), nm("E3")], cut - tr + 0.1, 0.08, rng, harm=12, lp=800.0, attack=1.0, release=0.02,
               sub=0.25)
    bus.add(with_dyn(d2, tr, lambda t: 0.6 + 0.6 * I(t)), tr)

    # pads, entering one by one with the shots (soft choir 'oo', organ flute, strings)
    pad = ifn(C, -16.0, -1.0)
    chord_pad(bus, "choir", chord("D3", "A3", "F4"), s02, s03 - s02 + 0.6, 0.05, rng,
              attack=3.0, release=2.0, bright=0.0, dyn=pad, width=0.7)
    chord_pad(bus, "organ", chord("D3", "F3", "Bb3", "D4"), s03, s04 - s03 + 0.4, 0.045, rng,
              attack=2.0, release=1.8, bright=0.1, dyn=pad)
    chord_pad(bus, "strings", chord("Bb2", "F3"), s03, s04 - s03 + 0.4, 0.04, rng,
              attack=2.5, release=1.8, bright=0.15, dyn=pad)
    chord_pad(bus, "organ", chord("D3", "G3", "Bb3", "D4"), s04, s05 - s04 + 0.4, 0.05, rng,
              attack=1.5, release=1.6, bright=0.15, dyn=pad)
    chord_pad(bus, "strings", chord("G2", "D3", "G3"), s04, s05 - s04 + 0.4, 0.05, rng,
              attack=1.5, release=1.6, bright=0.2, dyn=pad)
    chord_pad(bus, "strings", chord("F2", "C3", "F3", "A3", "C4"), s05, title - s05 + 0.3, 0.055, rng,
              attack=1.2, release=1.0, bright=0.3, dyn=pad)
    chord_pad(bus, "choir", chord("F3", "A3", "C4", "F4"), s05, title - s05 + 0.3, 0.05, rng,
              attack=1.4, release=1.0, bright=0.3, dyn=pad)
    # title: Bb major, fuller
    chord_pad(bus, "strings", chord("Bb1", "Bb2", "F3", "Bb3", "D4", "F4"), title, tr - title + 0.25, 0.06, rng,
              attack=0.5, release=0.6, bright=0.45, dyn=pad)
    chord_pad(bus, "organ", chord("Bb2", "F3", "Bb3", "D4"), title, tr - title + 0.25, 0.05, rng,
              attack=0.6, release=0.6, bright=0.35, dyn=pad)
    chord_pad(bus, "choir", chord("D4", "F4", "Bb4"), title, tr - title + 0.25, 0.045, rng,
              attack=0.8, release=0.6, bright=0.6, dyn=pad)
    bus.add(to_stereo(timpani_stroke(float(hz(nm("Bb2"))), 0.3, rng), 0.0), title)
    bus.add(to_stereo(boom(3.0, 66.0, 44.0, 0.15, 0.8, rng, noise=0.4), 0.0), title, gain=0.12)

    # the riser: A sus4 tremolo strings + choir + organ, crescendo into the cut
    rise = lambda t: undb(-26.0 * (1.0 - np.clip((t - tr) / (cut - tr), 0.0, 1.0)) ** 0.9)
    trem_rate = lambda t: 8.0 + 8.0 * np.clip((t - tr) / (cut - tr), 0.0, 1.0)
    brt = lambda t: 0.25 + 0.75 * np.clip((t - tr) / (cut - tr), 0.0, 1.0)
    for m in chord("A3", "D4", "E4", "A4", "D5", "E5"):
        bus.add(sustained("strings", m, tr, cut - tr + 0.05, 0.06, rng, attack=0.4, release=0.02,
                          bright=brt, dyn=rise, tremolo=(trem_rate, 0.55), width=0.8), tr)
    for m in chord("A2", "E3"):
        bus.add(sustained("strings", m, tr, cut - tr + 0.05, 0.07, rng, attack=0.5, release=0.02,
                          bright=brt, dyn=rise), tr)
    for m in chord("A2", "E3", "A3", "D4", "E4"):
        bus.add(sustained("organ", m, tr, cut - tr + 0.05, 0.05, rng, attack=0.8, release=0.02,
                          bright=brt, dyn=rise), tr)
    for m in chord("A3", "D4", "E4", "A4"):
        bus.add(sustained("choir", m, tr, cut - tr + 0.05, 0.05, rng, attack=0.8, release=0.02,
                          bright=brt, dyn=rise), tr)
    # timpani roll on A crescendo (alternating sticks, ~14 strokes/s, slight timing jitter)
    t = cut - 2.6
    k = 0
    while t < cut - 0.03:
        p = (t - (cut - 2.6)) / 2.6
        v = 0.05 + 0.35 * p ** 2.0
        bus.add(to_stereo(timpani_stroke(float(hz(nm("A2"))), v * rng.uniform(0.85, 1.0), rng, dur=1.2),
                          -0.15 if k % 2 else 0.15), t)
        t += 1.0 / (11.0 + 6.0 * p) + rng.normal(0.0, 0.004)
        k += 1
    bus.add(to_stereo(noise_riser(tr, cut, rng_for("riser1")), 0.0), tr, gain=0.16)

    # the motif: felt piano with a glass-bell halo an octave above
    beat1 = (s03 - (a0 + 0.44 * (s02 - a0))) / 8.5
    st1 = a0 + 0.44 * (s02 - a0)
    line_keys(bus, piano_note, motif_line(nm("D4"), 3), st1, beat1, 0.17, rng, pan=-0.1)
    line_keys(bus, bell_note, motif_line(nm("D5"), 3), st1, beat1, 0.04, rng, pan=0.25)
    line_keys(bus, piano_note, motif_line(nm("F4"), 4), s03, beat1, 0.21, rng, pan=0.1)
    line_keys(bus, bell_note, motif_line(nm("F5"), 4), s03, beat1, 0.05, rng, pan=-0.25)
    # low echo of the motif head under the interstage orbit
    line_keys(bus, piano_note, [(nm("D3"), 1.0), (nm("E3"), 1.0), (nm("A3"), 2.0)], s04 + 0.5 * (s05 - s04),
              beat1 * 0.9, 0.14, rng, pan=-0.2)
    # third statement in octaves, its peak note (A) landing on the title
    beat3 = float(np.clip((title - s05) / 2.2, 0.3, 0.75))
    st3 = title - 2.0 * beat3
    line_keys(bus, piano_note, motif_line(nm("D4"), 3), st3, beat3, 0.3, rng, pan=-0.05)
    line_keys(bus, piano_note, motif_line(nm("D3"), 3), st3, beat3, 0.18, rng, pan=-0.05)
    line_keys(bus, bell_note, motif_line(nm("D5"), 3), st3, beat3, 0.09, rng, pan=0.2)
    line_sus(bus, "strings", motif_line(nm("D3"), 3)[:3], st3, beat3, 0.07, rng, overlap=0.25,
             attack=0.25, release=0.6, bright=0.35)


def compose_act2(P, C, bus, shim):
    rng = rng_for("act2")
    t0, tj, te = P.ascent, P.jelly, P.meco
    ba, bb = P.beat2a, P.beat2b
    hold = lambda t: 0.62 + 0.38 * np.exp(-np.maximum(t - t0, 0.0) / 0.5)

    # THE HIT
    bus.add(to_stereo(boom(4.5, 76.0, 38.0, 0.12, 0.9, rng, noise=0.55), 0.0), t0, gain=0.3)
    bus.add(to_stereo(taiko(0.42, rng), 0.0), t0)
    bus.add(to_stereo(timpani_stroke(float(hz(nm("D2"))), 0.38, rng, dur=2.5), 0.0), t0)
    bus.add(tam(6.0, rng_for("tam1"), bloom=0.35, decay=4.0), t0, gain=0.1)
    # Dm tutti, then Bb (half-way through the first shot)
    tb_ = t0 + 4.0 * ba
    chord_pad(bus, "strings", chord("D2", "A2", "D3", "F3", "A3", "D4", "F4", "A4"), t0, tb_ - t0 + 0.15, 0.075, rng,
              attack=0.03, release=0.35, bright=0.75, dyn=hold, width=0.8)
    chord_pad(bus, "organ", chord("D2", "D3", "A3", "D4", "F4", "A4"), t0, tb_ - t0 + 0.15, 0.05, rng,
              attack=0.05, release=0.35, bright=0.8, dyn=hold)
    chord_pad(bus, "strings", chord("Bb1", "Bb2", "F3", "Bb3", "D4", "F4", "Bb4"), tb_, tj - tb_ + 0.2, 0.07, rng,
              attack=0.15, release=0.4, bright=0.7, width=0.8)
    chord_pad(bus, "organ", chord("Bb1", "Bb2", "F3", "Bb3", "D4", "F4"), tb_, tj - tb_ + 0.2, 0.05, rng,
              attack=0.2, release=0.4, bright=0.75)
    chord_pad(bus, "choir", chord("D4", "F4", "Bb4", "D5"), tb_, tj - tb_ + 0.2, 0.045, rng,
              attack=0.3, release=0.4, bright=1.0)
    # heroic motif in the horns (two octaves) and celli
    line_sus(bus, "horns", motif_line(nm("D4"), 3), t0, ba, 0.16, rng, legato=0.96, overlap=0.08,
             attack=0.05, release=0.25, bright=0.95, width=0.4)
    line_sus(bus, "horns", motif_line(nm("D3"), 3), t0, ba, 0.14, rng, legato=0.96, overlap=0.08,
             attack=0.06, release=0.25, bright=0.8, width=0.4)
    line_sus(bus, "strings", motif_line(nm("D5"), 3), t0, ba, 0.05, rng, legato=0.98, overlap=0.1,
             attack=0.04, release=0.3, bright=0.6)
    # driving eighth-note pulse in the low strings
    step = ba / 2.0
    k = 0
    t = t0
    while t < tj - step * 0.5:
        root = nm("D2") if t < tb_ else nm("Bb1")
        m = root + (12 if k % 2 else 0)
        v = 0.22 if k % 4 == 0 else 0.14
        bus.add(to_stereo(pluck_note(m, v, rng, decay=0.16, bright=0.55), 0.0), t)
        t += step
        k += 1
    for i in range(1, 4):
        bus.add(to_stereo(taiko(0.28 + 0.06 * i, rng), 0.0), t0 + 2.0 * i * ba)
    # small roll into the jellyfish cut
    t = tj - 4.0 * step
    while t < tj - 0.02:
        p = (t - (tj - 4.0 * step)) / (4.0 * step)
        bus.add(to_stereo(timpani_stroke(float(hz(nm("F2"))), 0.08 + 0.22 * p, rng, dur=1.0), 0.0), t)
        t += 1.0 / 14.0

    # WONDER: the music opens up (F, C, Bb lydian); motif sequence high in the violins
    tc = tj + 2.5 / 6.0 * (te - tj)
    tbb = tj + 4.5 / 6.0 * (te - tj)
    bus.add(to_stereo(boom(4.0, 58.0, 40.0, 0.2, 1.2, rng, noise=0.3), 0.0), tj, gain=0.2)
    bus.add(tam(7.0, rng_for("tam2"), bloom=0.9, decay=5.0), tj, gain=0.07)
    for (t1, t2, low, mid, ch) in (
            (tj, tc, chord("F2", "C3"), chord("C3", "F3", "A3", "C4"), chord("F4", "A4", "C5")),
            (tc, tbb, chord("C2", "G2"), chord("G3", "C4", "E4", "G4"), chord("E4", "G4", "C5")),
            (tbb, te, chord("Bb1", "F2"), chord("F3", "A3", "D4", "F4"), chord("D4", "E4", "A4", "D5"))):
        chord_pad(bus, "strings", low, t1, t2 - t1 + 0.4, 0.07, rng, attack=0.6, release=0.6, bright=0.4)
        chord_pad(bus, "strings", mid, t1, t2 - t1 + 0.4, 0.05, rng, attack=0.8, release=0.6, bright=0.5, width=0.9)
        chord_pad(bus, "choir", ch, t1, t2 - t1 + 0.4, 0.06, rng, attack=1.0, release=0.6, bright=1.0, width=0.9)
        chord_pad(bus, "organ", mid, t1, t2 - t1 + 0.4, 0.035, rng, attack=1.0, release=0.6, bright=0.3)
    line_sus(bus, "strings", motif_line(nm("F5"), 4), tj, bb, 0.085, rng, legato=1.0, overlap=0.15,
             attack=0.2, release=0.5, bright=0.55, width=0.5)
    line_sus(bus, "strings", motif_line(nm("F4"), 4), tj, bb, 0.06, rng, legato=1.0, overlap=0.15,
             attack=0.25, release=0.5, bright=0.45, width=0.5)
    line_keys(bus, bell_note, motif_line(nm("F5"), 4), tj, bb, 0.07, rng, pan=0.3)

    # the high shimmer is born here and survives MECO into the silence of space
    sh = shimmer(P.ret + 2.0 - tj, chord("A5", "D6", "E6", "A6"), rng_for("shimmer"), grain_rate=1.2,
                 grain_notes=chord("D6", "E6", "F6", "A6", "D7"), grain_level=0.35)
    t = tj + np.arange(sh.shape[1]) / SR
    sh_env = ramp_shape((t - tj) / 2.5) * (0.55 + 0.45 * ramp_shape((t - P.mvac) / 6.0)) \
        * ramp_shape((P.ret + 2.0 - t) / 3.0)
    shim.add(sh * sh_env, tj, gain=0.05)


def compose_act3(P, C, bus):
    rng = rng_for("act3")
    t1 = P.mvac + 1.0                                   # the ethereal return
    tf, tr = P.fairing, P.ret
    ta = tr - min(1.6, 0.3 * (tr - tf))                 # dominant lead-in to Act IV
    swell = lambda t: undb(-12.0 * (1.0 - np.clip((t - t1) / max(tf - t1, 0.5), 0.0, 1.0)))
    chord_pad(bus, "choir", chord("Bb3", "D4", "F4", "A4"), t1, tf - t1 + 1.0, 0.05, rng,
              attack=3.5, release=2.5, bright=0.1, dyn=swell, width=0.9)
    chord_pad(bus, "organ", chord("F3", "A3", "D4"), t1, tf - t1 + 1.0, 0.03, rng,
              attack=4.0, release=2.5, bright=0.0, dyn=swell)
    chord_pad(bus, "strings", chord("Bb2", "F3"), t1 + 1.0, tf - t1, 0.03, rng,
              attack=3.0, release=2.0, bright=0.1, dyn=swell)
    chord_pad(bus, "choir", chord("G3", "Bb3", "D4", "F4", "A4"), tf, ta - tf + 0.8, 0.05, rng,
              attack=2.0, release=1.5, bright=0.3, width=0.9)
    chord_pad(bus, "strings", chord("G2", "D3"), tf, ta - tf + 0.8, 0.04, rng,
              attack=2.0, release=1.5, bright=0.2)
    chord_pad(bus, "organ", chord("D4", "F4", "A4"), tf, ta - tf + 0.8, 0.03, rng,
              attack=2.0, release=1.5, bright=0.1)
    chord_pad(bus, "strings", chord("A2", "E3", "A3", "D4", "E4"), ta, tr - ta + 0.3, 0.034, rng,
              attack=0.9, release=0.5, bright=0.3)
    chord_pad(bus, "choir", chord("A3", "D4", "E4"), ta, tr - ta + 0.3, 0.03, rng,
              attack=0.9, release=0.5, bright=0.4)
    # fragments of the motif floating in the void
    frag1 = [(nm("A5"), 1.0), (nm("G5"), 0.5), (nm("F5"), 2.0)]
    line_keys(bus, bell_note, frag1, P.mvac + 2.4, 0.9, 0.12, rng, pan=0.3)
    line_keys(bus, piano_note, [(m - 12, b) for m, b in frag1], P.mvac + 2.4, 0.9, 0.12, rng, pan=-0.2)
    frag2 = [(nm("D5"), 1.0), (nm("E5"), 1.0), (nm("A5"), 2.0)]
    line_keys(bus, bell_note, frag2, tf + 1.0, 0.85, 0.11, rng, pan=-0.3)
    line_keys(bus, piano_note, [(m - 12, b) for m, b in frag2], tf + 1.0, 0.85, 0.14, rng, pan=0.1)


def compose_act4(P, C, bus):
    rng = rng_for("act4")
    I = C.intensity
    b, beat, T = P.bars4, P.beat4, P.touch
    step = beat / 4.0
    span = T - P.ret
    harmony = [  # (ostinato tones, bass, pad)
        (chord("D3", "A3", "D4", "F4"), chord("D2"), chord("A3", "D4", "F4")),
        (chord("Bb2", "F3", "Bb3", "D4"), chord("Bb1"), chord("F3", "Bb3", "D4")),
        (chord("G2", "D3", "G3", "Bb3"), chord("G2"), chord("G3", "Bb3", "D4")),
        (chord("A2", "E3", "A3", "D4"), chord("A2"), chord("A3", "D4", "E4")),
    ]
    a_major = chord("A2", "E3", "A3", "C#4")
    mid3 = 0.5 * (b[3] + b[4])

    # --- ostinato (16ths), the pulse of the descent -------------------------------------
    IDX = [0, 2, 1, 2, 0, 2, 3, 2, 0, 2, 1, 2, 0, 3, 1, 2]
    ACC = {0, 3, 6, 8, 11, 14}
    for s in range(P.nbeats4 * 4):
        t = P.ret + s * step
        bar = int(np.clip(np.searchsorted(b, t + 1e-6, side="right") - 1, 0, 3))
        tones = a_major if (bar == 3 and t >= mid3) else harmony[bar][0]
        prog = (t - P.ret) / span
        acc = 1.0 if (s % 16) in ACC else 0.0
        vel = (0.55 + 0.45 * acc) * (0.26 + 0.14 * prog)
        m = tones[IDX[s % 16]]
        bus.add(to_stereo(pluck_note(m, vel, rng, decay=0.15 + 0.05 * prog, bright=0.15 + 0.75 * prog), -0.15), t)
        bus.add(to_stereo(pluck_note(m + 12, vel * (0.5 if bar >= 2 else 0.35), rng, decay=0.12,
                                     bright=0.35 + 0.4 * prog), 0.3), t)

    # --- sustained harmony, crescendo with the intensity curve ---------------------------
    cres = ifn(C, -7.0, 0.0)
    for i in range(4):
        t1, t2 = b[i], b[i + 1]
        ost, bass, pad = harmony[i]
        k = (1.5, 1.35, 1.15, 1.0)[i]           # early bars carry the tension on their own
        chord_pad(bus, "strings", bass, t1, t2 - t1 + 0.1, 0.08 * k, rng, attack=0.4, release=0.35, bright=0.3,
                  dyn=cres)
        if i < 3:
            chord_pad(bus, "strings", pad, t1, t2 - t1 + 0.15, 0.06 * k, rng, attack=0.6, release=0.4,
                      bright=0.3 + 0.15 * i, dyn=cres, width=0.8)
            chord_pad(bus, "organ", pad, t1, t2 - t1 + 0.15, 0.04 * k, rng, attack=0.8, release=0.4, bright=0.25,
                      dyn=cres)
            chord_pad(bus, "choir", pad[1:], t1, t2 - t1 + 0.15, 0.035 * k, rng, attack=0.9, release=0.4,
                      bright=0.2 + 0.2 * i, dyn=cres, width=0.9)
        else:
            for notes, u1, u2 in ((chord("A3", "D4", "E4"), t1, mid3), (chord("A3", "C#4", "E4", "G4"), mid3, t2)):
                chord_pad(bus, "strings", notes, u1, u2 - u1 + 0.1, 0.05, rng, attack=0.3, release=0.12,
                          bright=0.7, dyn=cres, width=0.8)
                chord_pad(bus, "organ", notes, u1, u2 - u1 + 0.1, 0.035, rng, attack=0.3, release=0.12,
                          bright=0.5, dyn=cres)
                chord_pad(bus, "choir", notes[1:], u1, u2 - u1 + 0.1, 0.035, rng, attack=0.4, release=0.12,
                          bright=0.8, dyn=cres)

    # --- the motif returns in the horns and celli (minor), cadencing on the dominant ------
    bm = (b[3] - b[2]) / 4.0
    head = [(nm("D4"), 1.0), (nm("E4"), 1.0), (nm("A4"), 2.0)]
    tail = [(nm("G4"), 1.0), (nm("F4"), 1.0), (nm("E4"), 1.0), (nm("D4"), 0.5), (nm("E4"), 0.5)]
    for line, t1 in ((head, b[2]), (tail, b[3])):
        line_sus(bus, "horns", line, t1, bm, 0.12, rng, legato=0.95, overlap=0.06, attack=0.06,
                 release=0.18, bright=0.9, width=0.4)
        line_sus(bus, "horns", [(m - 12, d) for m, d in line], t1, bm, 0.1, rng, legato=0.95, overlap=0.06,
                 attack=0.07, release=0.18, bright=0.75, width=0.4)
        line_sus(bus, "strings", [(m - 12, d) for m, d in line], t1, bm, 0.05, rng, legato=1.0, overlap=0.08,
                 attack=0.06, release=0.2, bright=0.5)

    # --- percussion: pulse, then toms building into touchdown -----------------------------
    for k in range(P.nbeats4):
        t = P.ret + k * beat
        if t < b[1] - 1e-6:
            continue
        prog = (t - P.ret) / span
        bus.add(to_stereo(taiko(0.12 + 0.2 * prog, rng), 0.0), t)
    t = b[3]
    while t < T - 0.02:
        p = (t - b[3]) / (T - b[3])
        sub = step if p >= 0.5 else 2.0 * step
        f0 = 110.0 if int(round((t - b[3]) / step)) % 2 else 146.83
        bus.add(to_stereo(taiko(0.08 + 0.32 * p ** 1.5, rng, f0=f0, dur=0.9), 0.3 if f0 > 100 else -0.3), t)
        t += sub
    bus.add(to_stereo(noise_riser(T - 2.0 * beat, T, rng_for("riser2"), f_lo=600.0, f_hi=6000.0,
                                  floor_db=-26.0), 0.0), T - 2.0 * beat, gain=0.05)

    # --- TOUCHDOWN: D major --------------------------------------------------------------
    rel_t = P.s12 if P.s12 > T + 0.5 else T + 1.4
    hold = lambda t: (0.45 + 0.55 * np.exp(-np.maximum(t - T, 0.0) / 0.8)) * cres(t)
    bus.add(to_stereo(boom(5.0, 74.0, 38.0, 0.12, 1.1, rng, noise=0.5), 0.0), T, gain=0.28)
    bus.add(to_stereo(taiko(0.38, rng), 0.0), T)
    bus.add(to_stereo(timpani_stroke(float(hz(nm("D2"))), 0.35, rng, dur=3.0), 0.0), T)
    bus.add(tam(8.0, rng_for("tam3"), bloom=0.6, decay=5.5), T, gain=0.11)
    chord_pad(bus, "strings", chord("D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4", "D5"), T, rel_t - T + 0.6, 0.07,
              rng, attack=0.03, release=1.4, bright=0.75, dyn=hold, width=0.85)
    chord_pad(bus, "organ", chord("D2", "D3", "A3", "D4", "F#4", "A4"), T, rel_t - T + 0.6, 0.05, rng,
              attack=0.05, release=1.4, bright=0.8, dyn=hold)
    chord_pad(bus, "choir", chord("D4", "F#4", "A4", "D5"), T, rel_t - T + 0.6, 0.05, rng,
              attack=0.15, release=1.4, bright=1.0, dyn=hold, width=0.9)
    peak = [(nm("A4"), 1.5), (nm("G4"), 0.5), (nm("F#4"), 2.0)]
    line_sus(bus, "horns", peak, T, beat, 0.15, rng, overlap=0.1, attack=0.04, release=1.2, bright=1.0, width=0.4,
             dyn=hold)
    line_sus(bus, "horns", [(m - 12, d) for m, d in peak], T, beat, 0.12, rng, overlap=0.1, attack=0.05,
             release=1.2, bright=0.85, width=0.4, dyn=hold)
    line_sus(bus, "strings", [(m + 12, d) for m, d in peak], T, beat, 0.07, rng, overlap=0.15, attack=0.05,
             release=1.4, bright=0.6, width=0.5, dyn=hold)

    # --- resolution: plagal (G/D) to D, the motif in D major on piano and bell ------------
    calm = ifn(C, -20.0, -4.0)
    beat_e = min(0.68, max(0.45, (P.fade_out - P.end_titles) / 8.0))
    tG = max(rel_t + 0.5, P.end_titles)          # plagal IV as the end titles open
    tD = P.end_titles + 4.5 * beat_e             # home to I as the end motif sings G -> F#
    chord_pad(bus, "strings", chord("D2", "D3", "G3", "B3", "D4", "G4"), tG, tD - tG + 0.8, 0.09, rng,
              attack=1.2, release=1.4, bright=0.25, dyn=calm, width=0.8)
    chord_pad(bus, "choir", chord("B3", "D4", "G4"), tG, tD - tG + 0.8, 0.07, rng,
              attack=1.4, release=1.4, bright=0.2, dyn=calm, v1="oh", v2="ah")
    chord_pad(bus, "organ", chord("D3", "G3", "B3"), tG, tD - tG + 0.8, 0.045, rng,
              attack=1.4, release=1.4, bright=0.0, dyn=calm)
    end = P.dur + 0.5
    chord_pad(bus, "strings", chord("D2", "A2", "D3", "A3", "E4", "F#4"), tD, end - tD, 0.05, rng,
              attack=1.5, release=0.5, bright=0.2, dyn=calm, width=0.8)
    chord_pad(bus, "choir", chord("A3", "D4", "F#4"), tD, end - tD, 0.045, rng,
              attack=1.8, release=0.5, bright=0.1, dyn=calm, v1="oo", v2="oh")
    chord_pad(bus, "organ", chord("D3", "A3", "D4"), tD, end - tD, 0.03, rng,
              attack=2.0, release=0.5, bright=0.0, dyn=calm)
    line_keys(bus, piano_note, motif_line(nm("D5"), 4), P.end_titles, beat_e, 0.2, rng, pan=-0.1)
    line_keys(bus, piano_note, motif_line(nm("D4"), 4), P.end_titles, beat_e, 0.1, rng, pan=-0.1)
    line_keys(bus, bell_note, motif_line(nm("D6"), 4), P.end_titles, beat_e, 0.04, rng, pan=0.25)
    bus.add(to_stereo(piano_note(nm("D2"), 0.16, rng, dur=6.0), -0.05), P.end_titles + 4.5 * beat_e)
    bus.add(to_stereo(piano_note(nm("A2"), 0.1, rng, dur=6.0), -0.05), P.end_titles + 4.5 * beat_e)


# --------------------------------------------------------------------------------------
# the sound design
# --------------------------------------------------------------------------------------
def compose_sfx(P, C, buses):
    """Returns markers {event_id: (cue_time, isolated mono signal starting at cue_time)}
    used afterwards to verify that each event lands on its cue."""
    markers = {}

    # --- Act I: the light sweeping across the engine bells (airy, barely there) ---------
    rng = rng_for("sfx:sweep")
    t0 = P.fade_in + 0.8
    t1 = max(t0 + 1.5, min(P.s02, P.fade_in + 5.0) - 0.3)
    nn = int((t1 - t0) * SR)
    tt = np.arange(nn) / SR
    x = filt(filt(white(nn, rng), "bandpass", [1400.0, 5200.0], 2), "lowpass", 6000.0, 2)
    x = norm_rms(x) * np.sin(np.pi * tt / (t1 - t0)) ** 2
    gl, gr = pan_gains(-0.6 + 1.2 * tt / (t1 - t0))
    buses["fx1"].add(np.vstack([x * gl, x * gr]), t0, gain=0.01)

    # --- Act II: the ascent roar -------------------------------------------------------
    t0, tj, te = P.ascent, P.jelly, P.meco
    nn = int((te - t0 + 0.1) * SR)
    L = roar_layers(nn, "roar:ascent", crackle_rate=130.0, pop_rate=18.0)
    tt = t0 + np.arange(nn) / SR
    near = 0.48 * L["rumble"] + 0.8 * L["body"] + 0.7 * L["crackle"] + 0.1 * L["hiss"]
    near = soft_clip(near, 2.3 * rms(near))
    near = peaking_eq(near, 700.0, -4.0, 0.7)

    def g_near(f, t):   # brightest as it thunders past the camera, darker as it climbs away
        p = np.clip((t - t0) / (tj - t0), 0.0, 1.0)
        return lp_response(f, np.exp(np.log(8500.0) + (np.log(4000.0) - np.log(8500.0)) * p), 2)
    near = np.vstack([stft_filter(near[ch], g_near, t0) for ch in range(2)])
    lvl = undb(1.5 * np.exp(-((tt - (t0 + 0.7)) / 0.7) ** 2) - 3.0 * np.clip((tt - t0) / (tj - t0), 0.0, 1.0))
    far = (0.6 * filt(L["rumble"], "lowpass", 170.0, 2) + 0.6 * filt(L["body"], "lowpass", 420.0, 2)
           + 0.16 * filt(L["crackle"], "lowpass", 1200.0, 2)) * undb(-9.0)
    w = ramp_shape((tt - (tj - 0.02)) / 0.04)
    roar = (near * lvl * (1.0 - w) + far * w) * ramp_shape((tt - t0) / 0.004)
    buses["roar"].add(roar, t0, gain=0.22)
    markers["ascent_cut"] = (t0, roar[0, :int(0.5 * SR)])

    # --- Act III: space (muffled, felt more than heard) --------------------------------
    sp = buses["space"]
    rng = rng_for("sfx:space")
    th = thud(rng, 62.0, 38.0, 0.25, ((70, .5, .18), (112, .35, .15), (158, .22, .12), (236, .12, .08)), lp=330.0)
    sp.add(to_stereo(th, 0.0), P.sep, gain=0.3)
    markers["separation"] = (P.sep, th)

    dur = max(P.ret - P.mvac, 2.0)
    nn = int(dur * SR)
    tt = np.arange(nn) / SR
    wh = norm_rms(filt(white(nn, rng), "lowpass", 220.0, 2)) * ramp_shape(tt / 0.035) \
        * np.exp(-np.maximum(tt - 0.035, 0.0) / 0.4)
    wh = 0.5 * wh / (np.max(np.abs(wh)) + 1e-9) + boom(dur, 50.0, 34.0, 0.1, 0.5, rng, noise=0.2)[:nn]
    wh = wh / (np.max(np.abs(wh)) + 1e-9)
    rb = filt(brown(nn, rng), "lowpass", 90.0, 4) + 0.35 * filt(pink(nn, rng), "bandpass", [70.0, 240.0], 2)
    rb = norm_rms(rb) * np.exp(0.15 * smooth_noise(nn, 0.8, rng)) * ramp_shape((tt - 0.05) / 0.6) \
        * ramp_shape((dur - tt) / 2.5)
    sp.add(to_stereo(0.26 * wh + 0.018 * rb, 0.0), P.mvac)
    markers["mvac_ignition"] = (P.mvac, wh)

    pops = []
    for dt, p in ((0.0, -0.35), (P.fairing_gap, 0.35)):
        nn = int(0.6 * SR)
        tt = np.arange(nn) / SR
        pop = filt(white(nn, rng), "bandpass", [280.0, 1300.0], 2) * np.exp(-tt / 0.012)
        pop = pop / (np.max(np.abs(pop)) + 1e-9) + 0.8 * boom(0.6, 130.0, 72.0, 0.015, 0.07, rng, noise=0.3)
        pop = filt(pop * note_env(nn, 0.0015, 0.05), "lowpass", 1600.0, 2)
        pop /= np.max(np.abs(pop)) + 1e-9
        sp.add(to_stereo(pop, p), P.fairing + dt, gain=0.22)
        pops.append(pop)
    markers["fairing_sep"] = (P.fairing, pops[0])
    markers["fairing_sep_2"] = (P.fairing + P.fairing_gap, pops[1])

    # --- Act IV: the return ------------------------------------------------------------
    lb = buses["landing"]
    rng = rng_for("sfx:landing")
    durs = [0.2, 0.28, 0.16, 0.32, 0.22]
    for i, t in enumerate(P.cold):
        pf = filt(puff(durs[i % len(durs)], rng), "lowpass", 4500.0, 2)
        lb.add(to_stereo(pf, 0.45 if i % 2 else -0.45), t, gain=0.05)
        markers["cold_gas_%d" % (i + 1)] = (t, pf)

    wd = 0.32                                   # hydraulic whir as the fins swing out...
    nn = int(wd * SR)
    tt = np.arange(nn) / SR
    ph = TWO_PI * np.cumsum(160.0 + 90.0 * tt / wd) / SR
    whir = sum(k ** -1.3 * np.sin(k * ph) for k in range(1, 7))
    whir = whir + 0.3 * norm_rms(filt(white(nn, rng), "bandpass", [400.0, 2000.0], 2))
    whir *= ramp_shape(tt / 0.08) * ramp_shape((wd - tt) / 0.02)
    lb.add(to_stereo(whir / (np.max(np.abs(whir)) + 1e-9), 0.1), P.grid - wd, gain=0.025)
    ck = clunk(rng, 130.0, 62.0, 0.11, ((310, .5, .12), (487, .4, .09), (733, .3, .07), (1120, .2, .05),
                                       (1610, .12, .035)), lp=3000.0)
    lb.add(to_stereo(ck, 0.1), P.grid, gain=0.3)    # ...and lock
    markers["gridfins"] = (P.grid, ck)

    t0, T, tc = P.landing, P.touch, P.cutoff     # single-engine landing burn, approaching
    nn = int((tc - t0 + 0.4) * SR)
    tt = t0 + np.arange(nn) / SR
    L = roar_layers(nn, "roar:landing", crackle_rate=100.0, pop_rate=14.0)
    x = np.clip((tt - t0) / (T - t0), 0.0, 1.0)
    burn = 0.45 * L["rumble"] + 0.8 * L["body"] + 0.55 * L["crackle"] * (0.3 + 0.7 * x) + 0.06 * L["hiss"]
    burn = soft_clip(burn, 2.3 * rms(burn))

    def g_burn(f, t):
        xx = np.clip((t - t0) / (T - t0), 0.0, 1.0)
        return lp_response(f, np.exp(np.log(300.0) + (np.log(6500.0) - np.log(300.0)) * xx ** 1.3), 2)
    burn = np.vstack([stft_filter(burn[ch], g_burn, t0) for ch in range(2)])
    burn *= undb(-24.0 * (1.0 - x) ** 1.5) * ramp_shape((tt - t0) / 0.02) * ramp_shape((tc + 0.12 - tt) / 0.12)
    burn *= 1.0 - 0.3 * ramp_shape((tt - T) / 0.012)
    lb.add(burn, t0, gain=0.2)
    markers["landing_cut"] = (t0, burn[0, :int(0.5 * SR)])
    i_c = int(round((tc - t0) * SR))
    markers["engine_cutoff"] = (tc, burn[0, i_c - int(0.3 * SR):], "offset", tc - 0.3)

    hd = max(P.latch - P.legs - 0.05, 0.4)      # legs: pneumatic hiss, swing, heavy latch
    hs = noise_burst(hd, rng, (700.0, 5200.0), 0.03, 0.7, release=0.15, flutter=0.1, low=0.25)
    lb.add(to_stereo(hs, -0.2), P.legs, gain=0.06)
    markers["legs_deploy"] = (P.legs, hs)
    swg = noise_burst(hd, rng, (250.0, 1400.0), hd * 0.6, hd * 0.5, release=0.2, flutter=0.05, low=0.5)
    lb.add(to_stereo(swg, 0.2), P.legs, gain=0.03)
    lt = clunk(rng, 95.0, 46.0, 0.2, ((160, .6, .25), (240, .45, .2), (395, .35, .14), (585, .25, .1),
                                     (860, .15, .07)), lp=3500.0, dur=1.4, click=0.45)
    lb.add(to_stereo(lt, -0.05), P.latch, gain=0.45)
    markers["legs_latch"] = (P.latch, lt)

    im = impact(rng, 3.5)
    lb.add(to_stereo(im, 0.0), T, gain=0.6)
    markers["touchdown"] = (T, im)

    am = buses["amb"]                            # steam after cutoff, the sea at dusk
    sd = max(1.0, min(7.0, P.dur - tc))
    stl = corr_stereo(steam(sd, rng_for("steam0")), steam(sd, rng_for("steam1")), steam(sd, rng_for("steam2")), 0.5)
    am.add(stl, tc, gain=0.045)
    markers["steam"] = (tc, stl[0])
    sdur = P.dur - t0
    se = sea(sdur, rng)
    tt = t0 + np.arange(se.shape[1]) / SR
    am.add(se * ramp_shape((tt - t0) / 2.0) * (0.45 + 0.55 * ramp_shape((tt - tc) / 2.5)), t0, gain=0.012)
    return markers


# --------------------------------------------------------------------------------------
# loudness (ITU-R BS.1770-4), true peak, limiter
# --------------------------------------------------------------------------------------
def k_weight(x):
    """BS.1770 K-weighting at 48 kHz: high-shelf pre-filter + RLB high-pass."""
    y = signal.lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285],
                       [1.0, -1.69065929318241, 0.73248077421585], x, axis=-1)
    return signal.lfilter([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621], y, axis=-1)


def _block_powers(x, block, hop):
    y = k_weight(np.atleast_2d(x))
    p = np.sum(y ** 2, axis=0)                     # channel weights G = 1 (L, R)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    nb, nh = int(round(block * SR)), int(round(hop * SR))
    starts = np.arange(0, len(p) - nb + 1, nh)
    return (cs[starts + nb] - cs[starts]) / nb, starts


def _lufs(z):
    return -0.691 + 10.0 * np.log10(np.maximum(z, 1e-20))


def integrated_lufs(x):
    z, _ = _block_powers(x, 0.4, 0.1)
    l = _lufs(z)
    zg = z[l > -70.0]
    if len(zg) == 0:
        return -np.inf
    rel = _lufs(np.mean(zg)) - 10.0
    zg = z[(l > -70.0) & (l > rel)]
    return float(_lufs(np.mean(zg)))


def loudness_range(x):
    z, _ = _block_powers(x, 3.0, 0.1)
    l = _lufs(z)
    l = l[l > -70.0]
    if len(l) == 0:
        return 0.0
    l = l[l > _lufs(np.mean(10.0 ** ((l + 0.691) / 10.0))) - 20.0]
    return float(np.percentile(l, 95) - np.percentile(l, 10))


def short_term_curve(x):
    z, starts = _block_powers(x, 3.0, 0.1)
    return _lufs(z), (starts + int(1.5 * SR)) / SR


def momentary_max(x):
    z, _ = _block_powers(x, 0.4, 0.1)
    return float(np.max(_lufs(z)))


def true_peak(x):
    up = signal.resample_poly(np.atleast_2d(x), 4, 1, axis=-1)
    return float(np.max(np.abs(up)))


def peak_envelope(x):
    """Per-sample true-peak estimate: max over the 4x-oversampled values (both channels)."""
    n = x.shape[1]
    up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(axis=0)[:4 * n]
    return np.maximum(up.reshape(n, 4).max(axis=1), np.abs(x).max(axis=0))


def limiter_gain(x=None, ceiling_db=-1.3, lookahead=0.004, release=0.15, pk=None, scale=1.0):
    """Look-ahead brickwall limiter gain curve from a 4x-oversampled peak estimate.
    A centered min-filter followed by a centered box filter of the same width never exceeds
    the required gain; a control-rate (1 ms) release smoother keeps the recovery gentle, and
    linear interpolation of block minima keeps that guarantee at audio rate."""
    if pk is None:
        pk = peak_envelope(x)
    n = len(pk)
    req = np.minimum(1.0, undb(ceiling_db) / np.maximum(pk * scale, 1e-12))
    if req.min() >= 1.0:
        return np.ones(n)
    w = 2 * int(lookahead * SR) + 1
    g = uniform_filter1d(minimum_filter1d(req, size=w, mode="nearest"), size=w, mode="nearest")
    R = 48
    nblk = (n + R - 1) // R
    gp = np.pad(g, (0, nblk * R - n), mode="edge").reshape(nblk, R).min(axis=1)
    gp = minimum_filter1d(gp, size=3, mode="nearest")
    a = 1.0 - np.exp(-1.0 / (release * 1000.0))
    out = np.empty(nblk)
    cur = 1.0
    for k in range(nblk):
        cur = cur + (1.0 - cur) * a
        if gp[k] < cur:
            cur = gp[k]
        out[k] = cur
    return np.interp(np.arange(n), np.arange(nblk) * R, out)


def glue_gain(x, thr_db, ratio=2.5, knee=6.0, attack=0.08, release=0.7):
    """Slow 'glue' compressor gain: K-weighted 300 ms loudness detector at a 10 ms control
    rate, soft knee, attack/release smoothing of the gain reduction (in dB)."""
    p = np.sum(k_weight(x) ** 2, axis=0)
    R = 480
    nb = len(p) // R + 1
    pp = np.pad(p, (0, nb * R - len(p))).reshape(nb, R).mean(axis=1)
    lv = _lufs(uniform_filter1d(pp, size=30, mode="nearest"))
    over = lv - thr_db
    gr = np.where(over <= -knee / 2.0, 0.0,
                  np.where(over >= knee / 2.0, over * (1.0 - 1.0 / ratio),
                           (1.0 - 1.0 / ratio) * (over + knee / 2.0) ** 2 / (2.0 * knee)))
    a_att, a_rel = np.exp(-0.01 / attack), np.exp(-0.01 / release)
    out = np.empty(nb)
    cur = 0.0
    for k in range(nb):
        a = a_att if gr[k] > cur else a_rel
        cur = a * cur + (1.0 - a) * gr[k]
        out[k] = cur
    return np.interp(np.arange(len(p)), np.arange(nb) * R + R / 2.0, undb(-out))


def master(music, sfx, target=-14.0, ceiling_db=-1.3):
    """Glue compression, then the make-up gain that hits the integrated-loudness target after
    true-peak limiting (iterated).  Returns (make-up gain, total gain curve, glue curve)."""
    mix = music + sfx
    g_glue = glue_gain(mix, integrated_lufs(mix) + 2.0)
    mixg = mix * g_glue
    pk = peak_envelope(mixg)
    gain = undb(target - integrated_lufs(mixg))
    g_lim = np.ones(mix.shape[1])
    for _ in range(8):
        g_lim = limiter_gain(pk=pk, scale=gain, ceiling_db=ceiling_db)
        l = integrated_lufs(mixg * (gain * g_lim))
        if abs(l - target) < 0.02:
            break
        gain *= undb(target - l)
    return gain, g_lim * g_glue, g_glue


# --------------------------------------------------------------------------------------
# analysis / verification
# --------------------------------------------------------------------------------------
def band_fractions(x):
    mono = 0.5 * (x[0] + x[1])
    f, p = signal.welch(mono, fs=SR, nperseg=16384)
    tot = np.sum(p) + 1e-30
    edges = [0, 30, 60, 120, 250, 500, 2000, 8000, 12000, SR / 2]
    return {"%d-%d Hz" % (lo, hi): round(float(np.sum(p[(f >= lo) & (f < hi)]) / tot * 100.0), 3)
            for lo, hi in zip(edges[:-1], edges[1:])}


def stereo_stats(x):
    L, R = x[0], x[1]
    corr = float(np.sum(L * R) / np.sqrt(np.sum(L * L) * np.sum(R * R) + 1e-30))
    m = 0.5 * (L + R)
    mono_l = integrated_lufs(np.vstack([m, m]))
    st_l = integrated_lufs(x)
    return {"lr_correlation": round(corr, 3), "mono_minus_stereo_lufs": round(mono_l - st_l, 2)}


def onset_time(sig, t_cue, frac=0.1, pre=0.0):
    """First sample whose 1 ms envelope reaches frac of the event's peak (isolated signal)."""
    env = maximum_filter1d(np.abs(sig), size=48)
    lim = int(0.6 * SR)
    pk = np.max(env[:lim]) if len(env) else 0.0
    if pk <= 0:
        return None
    i = int(np.argmax(env[:lim] >= frac * pk))
    return t_cue + pre + i / SR


def onset_in_stem(stem, t_cue, win=(-0.1, 0.4)):
    """Onset of an event inside a full stem: first time the 5 ms envelope rises 10 dB above
    the level that precedes the cue.  Returns None if the event is masked by other sound."""
    mono = np.abs(0.5 * (stem[0] + stem[1]))
    a = max(0, int(round((t_cue + win[0]) * SR)))
    b = min(len(mono), int(round((t_cue + win[1]) * SR)))
    env = maximum_filter1d(mono[a:b], size=240)
    i_cue = int(round(t_cue * SR)) - a
    pre = float(np.percentile(env[:i_cue - 480], 90)) if i_cue > 960 else 0.0
    thr = max(pre * undb(10.0), 1e-5)
    i0 = max(0, i_cue - int(0.05 * SR))
    if np.max(env[i0:]) < thr * 1.5:
        return None
    return (a + i0 + int(np.argmax(env[i0:] >= thr))) / SR


def level_db_at(x, t, win=0.05):
    a, b = int((t - win / 2) * SR), int((t + win / 2) * SR)
    seg = x[:, max(a, 0):max(b, 1)]
    return float(db(np.sqrt(np.mean(seg ** 2) + 1e-20)))


def analyze(mix, music, sfx, P, markers, extra):
    rep = {"samples": int(mix.shape[1]), "duration_s": mix.shape[1] / SR, "sample_rate": SR}
    for name, x in (("mix", mix), ("music", music), ("sfx", sfx)):
        rep[name] = {
            "integrated_lufs": round(integrated_lufs(x), 2),
            "true_peak_dbtp": round(float(db(true_peak(x))), 2),
            "sample_peak_dbfs": round(float(db(np.max(np.abs(x)))), 2),
            "max_momentary_lufs": round(momentary_max(x), 2),
            "max_short_term_lufs": round(float(np.max(short_term_curve(x)[0])), 2),
            "loudness_range_lu": round(loudness_range(x), 2),
            "dc_offset": [float("%.2e" % np.mean(x[0])), float("%.2e" % np.mean(x[1]))],
            "clipped_samples": int(np.sum(np.abs(x) >= 0.99999)),
        }
    rep["mix"]["band_energy_percent"] = band_fractions(mix)
    rep["mix"]["stereo"] = stereo_stats(mix)
    hf = filt(0.5 * (mix[0] + mix[1]), "highpass", 8000.0, 4)
    rep["mix"]["hf_above_8k_rms_dbfs"] = round(float(db(rms(hf))), 1)
    rep["mix"]["hf_above_8k_peak_dbfs"] = round(float(db(np.max(np.abs(hf)))), 1)
    # loudness over time (short-term), coarse
    st, tt = short_term_curve(mix)
    rep["mix"]["short_term_lufs_every_2s"] = {"%.0f" % t: round(float(v), 1) for t, v in zip(tt[::20], st[::20])}
    # sections
    secs = {}
    for name, a, b in (("act1", P.fade_in, P.riser_cut), ("act2_near", P.ascent, P.jelly),
                       ("act2_far", P.jelly, P.meco), ("act3", P.meco, P.ret), ("act4", P.ret, P.touch),
                       ("resolution", P.touch, P.dur)):
        seg = mix[:, int(a * SR):int(b * SR)]
        secs[name] = {"lufs": round(integrated_lufs(seg), 1),
                      "music_lufs": round(integrated_lufs(music[:, int(a * SR):int(b * SR)]), 1),
                      "sfx_lufs": round(integrated_lufs(sfx[:, int(a * SR):int(b * SR)]), 1),
                      "lr_corr": stereo_stats(seg)["lr_correlation"] if np.any(seg) else None}
    rep["sections"] = secs
    # events
    ev = {}
    sfx_hp = filt(sfx, "highpass", 200.0, 4)
    for key, mk in sorted(markers.items(), key=lambda kv: kv[1][0]):
        t_cue, sig = mk[0], mk[1]
        if len(mk) > 2 and mk[2] == "offset":
            # envelope of the band above 500 Hz: wide-band noise gives a far steadier RMS than
            # the rumble-dominated full band, so the threshold crossings are not noise-driven
            env = np.sqrt(np.maximum(uniform_filter1d(filt(sig, "highpass", 500.0, 4) ** 2, size=960), 0.0))
            t_s = mk[3]
            i_cue = int(round((t_cue - t_s) * SR))
            ref = float(np.mean(env[max(0, i_cue - int(0.15 * SR)):i_cue - int(0.03 * SR)]))
            i1 = i_cue - int(0.05 * SR) + int(np.argmax(env[i_cue - int(0.05 * SR):] < ref * undb(-6.0)))
            i20 = i_cue - int(0.05 * SR) + int(np.argmax(env[i_cue - int(0.05 * SR):] < ref * undb(-20.0)))
            ev[key] = {"cue": round(t_cue, 4), "kind": "offset",
                       "down_6dB_after_ms": round((t_s + i1 / SR - t_cue) * 1000.0, 1),
                       "down_20dB_after_ms": round((t_s + i20 / SR - t_cue) * 1000.0, 1)}
            continue
        o = onset_time(sig, t_cue)
        spec_ = np.abs(np.fft.rfft(sig[:int(0.5 * SR)]))
        fq = np.fft.rfftfreq(min(len(sig), int(0.5 * SR)), 1.0 / SR)
        centroid = float(np.sum(fq * spec_) / (np.sum(spec_) + 1e-12))
        o2 = onset_in_stem(sfx_hp if centroid > 400.0 else sfx, t_cue)
        ev[key] = {"cue": round(t_cue, 4),
                   "isolated_onset_err_ms": None if o is None else round((o - t_cue) * 1000.0, 1),
                   "in_stem_onset_err_ms": "masked" if o2 is None else round((o2 - t_cue) * 1000.0, 1)}
    rep["events"] = ev
    rep["events_note"] = ("isolated: onset of the event's own layer (10% of its peak); in_stem: onset "
                          "inside sfx.wav (10 dB rise over the preceding 100 ms; events above 400 Hz "
                          "are judged above 200 Hz); 'masked' = buried under louder sound in the stem; offset events (engine cutoff): time after the cue at which the layer's >500 Hz envelope is 6 / 20 dB below its level just before the cue")
    # the cuts
    pre = level_db_at(mix, P.riser_cut - 0.06, 0.05)
    post = level_db_at(mix, P.riser_cut + 0.08, 0.05)
    gap = level_db_at(mix, 0.5 * (P.riser_cut + P.ascent), 0.1)
    pw = np.concatenate([[0.0], np.cumsum(0.5 * (music[0] ** 2 + music[1] ** 2))])
    env = np.sqrt(np.maximum(pw[2400:] - pw[:-2400], 0.0) / 2400.0)    # RMS of the preceding 50 ms
    a, b = int((P.riser - 1.5) * SR), int(P.ascent * SR)
    rep["riser"] = {"peak_time_s": round((a + 2400 + int(np.argmax(env[a:b]))) / SR, 3),
                    "level_before_cut_db": round(pre, 1), "level_80ms_after_cut_db": round(post, 1),
                    "silence_level_before_ascent_db": round(gap, 1)}
    mm = 0.5 * np.abs(mix[0] + mix[1])
    a = int((P.riser_cut + 0.05) * SR)
    i = a + int(np.argmax(mm[a:] > undb(-40.0)))
    rep["ascent_hit_first_sample_above_-40dBFS_s"] = round(i / SR, 4)
    rep["meco"] = {"level_100ms_before_db": round(level_db_at(mix, P.meco - 0.1, 0.05), 1),
                   "level_50ms_after_db": round(level_db_at(mix, P.meco + 0.05, 0.05), 1),
                   "level_500ms_after_db": round(level_db_at(mix, P.meco + 0.5, 0.2), 1)}
    rep["end"] = {"level_last_50ms_db": round(level_db_at(mix, P.dur - 0.025, 0.05), 1),
                  "level_at_fade_out_cue_db": round(level_db_at(mix, P.fade_out, 0.2), 1),
                  "level_at_0.2s_db": round(level_db_at(mix, 0.2, 0.2), 1)}
    rep.update(extra)
    return rep


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def write_wav(path, x, bits=24):
    x = np.ascontiguousarray(np.clip(x, -1.0, 1.0).T)
    if bits == 24:
        q = np.round(x * 8388607.0).astype("<i4")
        data = q.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        width = 3
    else:
        data = np.round(x * 32767.0).astype("<i2").tobytes()
        width = 2
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(width)
        w.setframerate(SR)
        w.writeframes(data)


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, width = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = np.frombuffer(w.readframes(n), dtype=np.uint8)
    if width == 3:
        b = raw.reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v) / 8388607.0
    else:
        v = raw.view("<i2") / 32767.0
    return v.reshape(-1, ch).T


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def finish(bus, hp=28.0, on=None, off=None, fade_in=0.003, fade_out=0.012):
    """Subsonic/DC high-pass, then the bus gate (applied after reverb, so cuts are dead)."""
    bus.x = filt(bus.x, "highpass", hp, 4)
    if on is not None or off is not None:
        bus.x *= gate(bus.x.shape[1], on, off, fade_in, fade_out)
    return bus.x


def main(argv=None):
    ap = argparse.ArgumentParser(description="Synthesize the FALCON 9 score and sound design.")
    ap.add_argument("--cues", default=os.path.join("film", "cues.json"))
    ap.add_argument("--outdir", default=os.path.join("film", "audio"))
    ap.add_argument("--lufs", type=float, default=-14.0, help="integrated loudness target of mix.wav")
    ap.add_argument("--ceiling", type=float, default=-1.3, help="limiter ceiling, dBTP")
    ap.add_argument("--bits", type=int, default=24, choices=(16, 24))
    args = ap.parse_args(argv)

    C = Cues(args.cues)
    P = Plan(C)
    n = P.n
    os.makedirs(args.outdir, exist_ok=True)
    log("cues: %.3f s, %d samples; plan %s" % (C.duration, n, json.dumps(P.summary())))

    # impulse responses
    hall = make_ir("hall", length=5.0, rt_low=2.4, rt_mid=3.4, rt_high=1.4, predelay=0.024)
    void = make_ir("void", length=7.0, rt_low=3.0, rt_mid=5.5, rt_high=2.2, predelay=0.04, lp=7500.0)
    outdoor = make_ir("outdoor", length=2.6, rt_low=1.8, rt_mid=1.5, rt_high=0.6, predelay=0.05,
                      early=0.6, lp=3500.0, hp=35.0)
    muffled = make_ir("muffled", length=1.8, rt_low=1.2, rt_mid=0.9, rt_high=0.4, predelay=0.015,
                      lp=1800.0, hp=30.0)
    sea_ir = make_ir("sea", length=2.2, rt_low=1.4, rt_mid=1.2, rt_high=0.5, predelay=0.06,
                     early=0.5, lp=4000.0, hp=40.0)
    log("impulse responses ready")

    # ---- music ----
    mb = {k: Bus(n, k) for k in ("act1", "act2", "shimmer", "act3", "act4")}
    compose_act1(P, C, mb["act1"])
    log("act I composed")
    compose_act2(P, C, mb["act2"], mb["shimmer"])
    log("act II composed")
    compose_act3(P, C, mb["act3"])
    compose_act4(P, C, mb["act4"])
    log("acts III-IV composed")
    reverb_bus(mb["act1"], hall, 0.5, 0.0, P.riser_cut + 0.05)
    reverb_bus(mb["act2"], hall, 0.32, P.ascent, P.meco + 0.05)
    reverb_bus(mb["shimmer"], void, 0.9, P.jelly, None, send_hp=300.0)
    reverb_bus(mb["act3"], void, 0.75, P.meco, None)
    reverb_bus(mb["act4"], hall, 0.4, P.ret - 0.5, None)
    music = finish(mb["act1"], off=P.riser_cut) + finish(mb["act2"], on=P.ascent, off=P.meco, fade_in=0.002,
                                                          fade_out=0.015)
    music += finish(mb["shimmer"]) + finish(mb["act3"]) + finish(mb["act4"])
    music = high_shelf(music, 3200.0, 2.5)
    log("music reverbs and gates done")

    # ---- sound design ----
    sb = {k: Bus(n, k) for k in ("fx1", "roar", "space", "landing", "amb")}
    markers = compose_sfx(P, C, sb)
    log("sound design composed")
    reverb_bus(sb["fx1"], hall, 0.4, 0.0, P.s02)
    reverb_bus(sb["roar"], outdoor, 0.22, P.ascent, P.meco + 0.05, send_hp=50.0)
    reverb_bus(sb["space"], muffled, 0.35, P.sep - 0.1, None, send_hp=35.0)
    reverb_bus(sb["landing"], sea_ir, 0.16, P.ret, None, send_hp=60.0)
    sfx = finish(sb["fx1"], off=P.riser_cut) + finish(sb["roar"], on=P.ascent, off=P.meco, fade_in=0.002,
                                                       fade_out=0.015)
    sfx += finish(sb["space"]) + finish(sb["landing"]) + finish(sb["amb"])

    # ---- global envelope: from silence, to silence by the end ----
    t = np.arange(n) / SR
    t_end = P.dur - 0.03
    genv = ramp_shape((t - P.fade_in) / 0.01) * ramp_shape((t_end - t) / max(t_end - P.fade_out, 0.05))
    genv[t >= t_end] = 0.0
    music *= genv
    sfx *= genv

    # ---- master: loudness target + true-peak limiting (same gain applied to the stems) ----
    raw = {"music_lufs_raw": round(integrated_lufs(music), 2), "sfx_lufs_raw": round(integrated_lufs(sfx), 2)}
    gain, g_tot, g_glue = master(music, sfx, args.lufs, args.ceiling)
    music_o = music * gain * g_tot
    sfx_o = sfx * gain * g_tot
    mix_o = music_o + sfx_o
    gr = db(g_tot / g_glue)
    extra = {"master": {"makeup_gain_db": round(float(db(gain)), 2),
                        "glue_max_gain_reduction_db": round(float(-np.min(db(g_glue))), 2),
                        "glue_mean_gain_reduction_db": round(float(-np.mean(db(g_glue))), 2),
                        "limiter_max_gain_reduction_db": round(float(-np.min(gr)), 2),
                        "limiter_time_above_1dB_reduction_s": round(float(np.sum(gr < -1.0) / SR), 3),
                        "limiter_time_above_3dB_reduction_s": round(float(np.sum(gr < -3.0) / SR), 3),
                        **raw},
             "plan": P.summary()}
    nsec = int(np.ceil(n / SR))
    grs = np.pad(gr, (0, nsec * SR - n)).reshape(nsec, SR).min(axis=1)
    extra["master"]["limiter_gr_db_by_second"] = [round(float(-v), 1) for v in grs]
    # stems may not exceed the ceiling either (they normally sit below the mix)
    for name in ("music", "sfx"):
        x = music_o if name == "music" else sfx_o
        if db(true_peak(x)) > -1.0:
            g_stem = limiter_gain(x, args.ceiling)
            x *= g_stem
            extra["master"][name + "_stem_extra_limiting_max_db"] = round(float(-np.min(db(g_stem))), 2)
    log("mastered: makeup %.2f dB, max limiter reduction %.2f dB" % (db(gain), -np.min(gr)))

    paths = {}
    for name, x in (("mix", mix_o), ("music", music_o), ("sfx", sfx_o)):
        p = os.path.join(args.outdir, name + ".wav")
        write_wav(p, x, args.bits)
        paths[name] = p
    log("wrote %s" % ", ".join(paths.values()))

    # ---- verify from the files on disk ----
    mix_r, mus_r, sfx_r = (read_wav(paths[k]) for k in ("mix", "music", "sfx"))
    rep = analyze(mix_r, mus_r, sfx_r, P, markers, extra)
    rep["files"] = {k: os.path.abspath(v) for k, v in paths.items()}
    rep["bits"] = args.bits
    with open(os.path.join(args.outdir, "score_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    m = rep["mix"]
    log("mix: %.2f LUFS integrated, true peak %.2f dBTP, LRA %.1f LU, corr %.2f, >8k %.2f%%" % (
        m["integrated_lufs"], m["true_peak_dbtp"], m["loudness_range_lu"], m["stereo"]["lr_correlation"],
        m["band_energy_percent"]["8000-12000 Hz"] + m["band_energy_percent"]["12000-24000 Hz"]))
    log("done in %.1f s" % (time.time() - T_START))
    return 0


if __name__ == "__main__":
    sys.exit(main())
