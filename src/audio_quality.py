"""Signal-level quality measurements used to *skip* bad ground-truth candidates in the samplers (the audio itself is
never modified). All numbers come from simple statistics, so they are fast and reproducible.

    from audio_quality import measure, passes, DEFAULT_THRESHOLDS
    m = measure("clip.mp3")            # or bytes / (array, sr)
    ok, reasons = passes(m)            # reasons lists the failed checks

Metrics: loudness_lufs (ITU-R BS.1770 integrated, pyloudnorm), peak_dbfs, noise_floor_dbfs (10th percentile of frame RMS),
speech_dbfs (90th percentile), snr_db (speech - noise floor), lead_silence_s / trail_silence_s (frames more than
SILENCE_DROP dB below the speech level at either edge), speech_ratio (share of frames above that gate), clip_ratio
(share of samples at full scale), duration_s, sample_rate.
"""
import io
import numpy as np, soundfile as sf, pyloudnorm as pyln

FRAME_S, HOP_S, SILENCE_DROP = 0.025, 0.010, 35.0
# Calibrated 2026-11-21 on the Hausa/Yoruba/Swahili pools (Common Voice, FLEURS test, Open Bible test): pass rates 44-94%.
DEFAULT_THRESHOLDS = dict(min_loudness_lufs=-38.0, min_peak_dbfs=-20.0, min_snr_db=22.0, max_noise_floor_dbfs=-48.0,
                          max_edge_silence_s=1.0, max_total_edge_s=1.5, min_speech_ratio=0.45, max_clip_ratio=0.001,
                          max_band_floor_db=-35.0,                      # stationary background in any band vs speech level
                          min_dnsmos_bak=None, min_dnsmos_ovrl=None)    # set per language when DNSMOS is enabled
METRIC_COLUMNS = ["loudness_lufs", "peak_dbfs", "snr_db", "noise_floor_dbfs", "lead_silence_s", "trail_silence_s", "speech_ratio", "clip_ratio",
                  "floor_low_db", "floor_mid_db", "floor_high_db"]
DNSMOS_COLUMNS = ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"]


def _load(src):
    if isinstance(src, tuple): y, sr = src
    elif isinstance(src, (bytes, bytearray)): y, sr = sf.read(io.BytesIO(src), dtype="float32")
    else: y, sr = sf.read(src, dtype="float32")
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1: y = y.mean(axis=1)
    return y, int(sr)


def _db(x): return 20 * np.log10(np.maximum(x, 1e-9))


def measure(src):
    y, sr = _load(src)
    n = len(y); dur = n / sr
    frame, hop = max(int(FRAME_S * sr), 1), max(int(HOP_S * sr), 1)
    if n < frame: y = np.pad(y, (0, frame - n))
    idx = np.arange(0, len(y) - frame + 1, hop)
    rms = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(y, frame)[idx] ** 2, axis=1))
    rms_db = _db(rms)
    speech_db, floor_db = np.percentile(rms_db, 90), np.percentile(rms_db, 10)
    gate = speech_db - SILENCE_DROP
    active = rms_db > gate
    lead = int(np.argmax(active)) if active.any() else len(active)
    trail = int(np.argmax(active[::-1])) if active.any() else len(active)
    try: lufs = float(pyln.Meter(sr).integrated_loudness(y)) if dur >= 0.4 else float("nan")
    except Exception: lufs = float("nan")
    return dict(duration_s=round(dur, 3), sample_rate=sr, loudness_lufs=round(lufs, 2), peak_dbfs=round(float(_db(np.abs(y).max())), 2),
                speech_dbfs=round(float(speech_db), 2), noise_floor_dbfs=round(float(floor_db), 2), snr_db=round(float(speech_db - floor_db), 2),
                lead_silence_s=round(lead * hop / sr, 3), trail_silence_s=round(trail * hop / sr, 3),
                speech_ratio=round(float(active.mean()), 3), clip_ratio=round(float(np.mean(np.abs(y) >= 0.999)), 5),
                **band_floors(y, sr))


def band_floors(y, sr):
    """Stationary background per band: 5th-percentile power over time in 0-1 kHz, 1-4 kHz and 4-8 kHz, relative to the
    clip's speech level (90th percentile in 500-2000 Hz). Catches hum, rumble and continuous ambience under pause-less
    speech, where the frame-RMS noise floor cannot see it. Clean recordings sit around -45 dB and below; the noisy FLEURS
    clips found in calibration sit at -16 to -30 dB in the low band."""
    import librosa
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    S = 10 * np.log10(np.abs(librosa.stft(y, n_fft=512, hop_length=160)) ** 2 + 1e-12)
    f = np.linspace(0, sr / 2, S.shape[0]); ref = np.percentile(S[(f >= 500) & (f < 2000)], 90)
    def fl(a, b): return round(float(np.percentile(S[(f >= a) & (f < b)], 5, axis=1).mean() - ref), 1)
    return dict(floor_low_db=fl(0, 1000), floor_mid_db=fl(1000, 4000), floor_high_db=fl(4000, 8000))


def passes(m, t=DEFAULT_THRESHOLDS):
    """(ok, [failed checks]) for a metrics dict from `measure`."""
    r = []
    if not np.isnan(m["loudness_lufs"]) and m["loudness_lufs"] < t["min_loudness_lufs"]: r.append(f"quiet {m['loudness_lufs']} LUFS")
    if m["peak_dbfs"] < t["min_peak_dbfs"]: r.append(f"low peak {m['peak_dbfs']} dBFS")
    if m["snr_db"] < t["min_snr_db"]: r.append(f"snr {m['snr_db']} dB")
    if m["noise_floor_dbfs"] > t["max_noise_floor_dbfs"]: r.append(f"noise floor {m['noise_floor_dbfs']} dBFS")
    if m["lead_silence_s"] > t["max_edge_silence_s"]: r.append(f"lead silence {m['lead_silence_s']} s")
    if m["trail_silence_s"] > t["max_edge_silence_s"]: r.append(f"trail silence {m['trail_silence_s']} s")
    if m["lead_silence_s"] + m["trail_silence_s"] > t.get("max_total_edge_s", 1e9): r.append(f"edge silence total {m['lead_silence_s'] + m['trail_silence_s']:.2f} s")
    if m["speech_ratio"] < t["min_speech_ratio"]: r.append(f"speech ratio {m['speech_ratio']}")
    if m["clip_ratio"] > t["max_clip_ratio"]: r.append(f"clipping {m['clip_ratio']}")
    bf = max(m.get("floor_low_db", -99), m.get("floor_mid_db", -99), m.get("floor_high_db", -99))
    if t.get("max_band_floor_db") is not None and bf > t["max_band_floor_db"]: r.append(f"stationary noise {bf} dB")
    if t.get("min_dnsmos_bak") is not None and m.get("dnsmos_bak", 5) < t["min_dnsmos_bak"]: r.append(f"DNSMOS BAK {m['dnsmos_bak']}")
    if t.get("min_dnsmos_ovrl") is not None and m.get("dnsmos_ovrl", 5) < t["min_dnsmos_ovrl"]: r.append(f"DNSMOS OVRL {m['dnsmos_ovrl']}")
    return (not r), r
