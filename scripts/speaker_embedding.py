"""Speaker embeddings ("voiceprints") with WeSpeaker ResNet34 on MLX.

The model is pyannote/wespeaker-voxceleb-resnet34-LM (MIT), 6.6M parameters,
256-dim output, trained for speaker verification on VoxCeleb. Weights come
from mlx-community's conversion (pinned in config/mlx-models.lock.json); the
network is defined here rather than by that repo's script, which gives its
convolutions a bias the checkpoint does not have and pools in a different
order from PyTorch. This definition matches Wespeaker's own ONNX export —
see docs/voiceprints.md for the check.

Features are Kaldi fbank exactly as WeSpeaker computes them: 80 mel bins,
25 ms Hamming windows every 10 ms on 16-bit-scaled audio, no dither, log
floored at float32 epsilon, then mean-normalised over time.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SAMPLE_RATE = 16000
_LOG_FLOOR = float(np.log(np.finfo(np.float32).eps))


def decode(path) -> np.ndarray:
    """Any file ffmpeg reads -> 16 kHz mono float32."""
    import subprocess

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def fbank(audio: np.ndarray) -> mx.array:
    """(samples,) float waveform in [-1, 1] at 16 kHz -> (frames, 80), CMN applied."""
    from mlx_audio.dsp import compute_fbank_kaldi

    feats = compute_fbank_kaldi(
        mx.array(np.asarray(audio, dtype=np.float32) * 32768.0),
        sample_rate=SAMPLE_RATE, win_len=400, win_inc=160, num_mels=80,
        win_type="hamming", dither=0.0,
    )
    # mlx-audio floors the log at 1e-8, torchaudio (and so WeSpeaker) at eps.
    feats = mx.maximum(feats, _LOG_FLOOR)
    return feats - mx.mean(feats, axis=0, keepdims=True)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(cout)
        if stride != 1 or cin != cout:
            self.shortcut = [nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                             nn.BatchNorm(cout)]

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if "shortcut" in self:
            x = self.shortcut[1](self.shortcut[0](x))
        return nn.relu(out + x)


class ResNet34(nn.Module):
    def __init__(self, feat_dim: int = 80, embed_dim: int = 256, channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(channels)
        widths = [channels, channels * 2, channels * 4, channels * 8]
        cin = channels
        for index, (blocks, width) in enumerate(zip((3, 4, 6, 3), widths), 1):
            stride = 1 if index == 1 else 2
            layer = [_Block(cin, width, stride)] + [_Block(width, width, 1) for _ in range(blocks - 1)]
            setattr(self, f"layer{index}", layer)
            cin = width
        pooled = widths[-1] * (feat_dim // 8) * 2
        self.seg_1 = nn.Linear(pooled, embed_dim)

    def __call__(self, feats: mx.array) -> mx.array:
        """(batch, frames, 80) -> (batch, 256), not normalised."""
        # PyTorch runs NCHW with H = frequency, W = time; MLX is NHWC.
        x = mx.transpose(feats, (0, 2, 1))[..., None]
        x = nn.relu(self.bn1(self.conv1(x)))
        for index in range(1, 5):
            for block in getattr(self, f"layer{index}"):
                x = block(x)
        # Temporal statistics pooling, flattened channel-major as torch does
        # for (batch, channels, freq): mean block, then std block.
        x = mx.transpose(x, (0, 3, 1, 2))            # (batch, channels, freq, time)
        mean = mx.mean(x, axis=-1)
        std = mx.sqrt(mx.var(x, axis=-1, ddof=1) + 1e-7)  # torch.var is unbiased
        batch = x.shape[0]
        stats = mx.concatenate([mean.reshape(batch, -1), std.reshape(batch, -1)], axis=-1)
        return self.seg_1(stats)


class SpeakerEmbedder:
    """Loads once; `embed(audio)` returns a unit-length 256-dim numpy vector."""

    def __init__(self, model_dir: str):
        from pathlib import Path

        weights = mx.load(str(Path(model_dir) / "weights.npz"))
        model = ResNet34()
        model.load_weights([(k.removeprefix("resnet."), v) for k, v in weights.items()],
                           strict=True)
        model.eval()
        mx.eval(model.parameters())
        self.model = model

    def embed(self, audio: np.ndarray) -> np.ndarray:
        vec = np.array(self.model(fbank(audio)[None])[0], dtype=np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-12)
