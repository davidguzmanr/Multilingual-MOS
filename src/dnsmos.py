"""DNSMOS P.835 (Microsoft DNS Challenge) scorer: SIG / BAK / OVRL on a 1-5 scale, plus the P.808 MOS, from the two
ONNX models in checkpoints/DNSMOS/ (sig_bak_ovr.onnx, model_v8.onnx; download with:
  curl -sSL -o checkpoints/DNSMOS/<f> https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/<f>).
Port of DNSMOS/dnsmos_local.py: 16 kHz mono, 9.01 s windows with 1 s hop, segment scores averaged, non-personalised
polynomial mapping. CPU only, ~0.2 s per clip. BAK is the background-noise score used by the samplers' quality gate.

    from dnsmos import DNSMOS; d = DNSMOS(); d.score("clip.wav") -> {"dnsmos_sig": ..., "dnsmos_bak": ..., "dnsmos_ovrl": ..., "dnsmos_p808": ...}
"""
import io, os
import numpy as np, librosa, onnxruntime as ort, soundfile as sf

REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"
SR, INPUT_LENGTH = 16000, 9.01
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


class DNSMOS:
    def __init__(self, model_dir=f"{REPO}/checkpoints/DNSMOS"):
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(f"{model_dir}/sig_bak_ovr.onnx", opts, providers=["CPUExecutionProvider"])
        self.p808 = ort.InferenceSession(f"{model_dir}/model_v8.onnx", opts, providers=["CPUExecutionProvider"])

    @staticmethod
    def _melspec(audio):
        m = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(m, ref=np.max) + 40) / 40).T

    def score(self, src):
        if isinstance(src, tuple): y, sr = src
        elif isinstance(src, (bytes, bytearray)): y, sr = sf.read(io.BytesIO(src), dtype="float32")
        else: y, sr = sf.read(src, dtype="float32")
        y = np.asarray(y, dtype=np.float32); y = y.mean(axis=1) if y.ndim > 1 else y
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        need = int(INPUT_LENGTH * SR)
        while len(y) < need: y = np.concatenate([y, y])                       # dnsmos_local tiles short clips
        n_hops = int(np.floor(len(y) / SR) - INPUT_LENGTH) + 1
        sig, bak, ovr, p808 = [], [], [], []
        for h in range(n_hops):
            seg = y[h * SR: h * SR + need]
            if len(seg) < need: break
            mos_sig, mos_bak, mos_ovr = self.sess.run(None, {"input_1": seg[np.newaxis, :].astype(np.float32)})[0][0]
            p808.append(float(self.p808.run(None, {"input_1": self._melspec(seg[:-160])[np.newaxis, :, :].astype(np.float32)})[0][0][0]))
            sig.append(float(P_SIG(mos_sig))); bak.append(float(P_BAK(mos_bak))); ovr.append(float(P_OVR(mos_ovr)))
        return dict(dnsmos_sig=round(np.mean(sig), 3), dnsmos_bak=round(np.mean(bak), 3), dnsmos_ovrl=round(np.mean(ovr), 3), dnsmos_p808=round(np.mean(p808), 3))
