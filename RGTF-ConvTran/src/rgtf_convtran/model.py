"""RGTF-ConvTran network definition.

The implementation is extracted from the two supplied experiment sources and
contains architecture code only. It intentionally excludes loss functions,
optimizers, schedulers, training loops, and checkpoint-selection logic.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class TAPositionalEncoding(nn.Module):
    """Time absolute positional encoding (tAPE) for ``[B, L, D]`` tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal tAPE")
        self.d_model = d_model

    def forward(
        self,
        length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        k = torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
        omega = torch.pow(
            torch.tensor(10000.0, device=device, dtype=dtype),
            -k / self.d_model,
        )
        omega = omega * (self.d_model / max(length, 1))
        encoding = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * omega)
        encoding[:, 1::2] = torch.cos(position * omega)
        return encoding.unsqueeze(0)


class SEBlock2D(nn.Module):
    """Squeeze-and-excitation block used by the frequency branch."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ConvBNGELU2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RawECGEncoder(nn.Module):
    """Temporal and spatial convolutions mapping ECG to temporal tokens."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        temporal_filters: int = 64,
        temporal_kernel: int = 8,
        temporal_stride: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride
        padding = temporal_kernel // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel),
                stride=(1, temporal_stride),
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.GELU(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                d_model,
                kernel_size=(in_channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def output_length(self, seq_len: int) -> int:
        kernel = self.temporal_kernel
        stride = self.temporal_stride
        padding = self.temporal_kernel // 2
        return (seq_len + 2 * padding - kernel) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x=[B,C,L], got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels={x.shape[1]}, expected {self.in_channels}"
            )
        features = self.temporal_conv(x.unsqueeze(1))
        features = self.spatial_conv(features)
        return features.squeeze(2).transpose(1, 2)


class RRRhythmEncoder(nn.Module):
    """Map the 14-dimensional RR/HRV vector to one rhythm token."""

    def __init__(self, hrv_dim: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hrv_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, hrv_features: torch.Tensor) -> torch.Tensor:
        return self.net(hrv_features).unsqueeze(1)


class TokenAttentionPooling(nn.Module):
    """Attention pooling across the rhythm token and all ECG tokens."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model // 2, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(tokens), dim=1)
        pooled = torch.sum(tokens * weights, dim=1)
        return pooled, weights.squeeze(-1)


class ERPEMultiheadAttention(nn.Module):
    """Multi-head self-attention with enhanced relative position encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        post_softmax: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.post_softmax = post_softmax
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rel = nn.Parameter(torch.zeros(num_heads, 2 * max_seq_len - 1))
        nn.init.trunc_normal_(self.rel, std=0.02)
        index = torch.arange(max_seq_len)
        relative_index = index[None, :] - index[:, None] + max_seq_len - 1
        self.register_buffer("rel_index", relative_index, persistent=False)

    def _relative_bias(self, length: int) -> torch.Tensor:
        if length > self.max_seq_len:
            raise ValueError(
                f"Token length {length} exceeds max_seq_len={self.max_seq_len}"
            )
        index = self.rel_index[:length, :length]
        return self.rel[:, index]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, dimensions = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        logits = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask == 0, float("-inf"))
        bias = self._relative_bias(length).unsqueeze(0)
        if self.post_softmax:
            attention = torch.softmax(logits, dim=-1) + bias
        else:
            attention = torch.softmax(logits + bias, dim=-1)
        attention = self.attn_drop(attention)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch, length, dimensions)
        return self.proj_drop(self.proj(output))


class ConvTranEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
        erpe_post_softmax: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ERPEMultiheadAttention(
            d_model,
            num_heads,
            max_seq_len,
            dropout,
            erpe_post_softmax,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class RhythmAwareConvTranBranch(nn.Module):
    """Raw ECG tokens plus an RR/HRV rhythm token and ConvTran encoder."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        hrv_dim: int,
        d_model: int = 128,
        out_dim: int = 128,
        temporal_filters: int = 64,
        temporal_kernel: int = 8,
        temporal_stride: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        erpe_post_softmax: bool = True,
    ):
        super().__init__()
        self.raw_encoder = RawECGEncoder(
            in_channels,
            d_model,
            temporal_filters,
            temporal_kernel,
            temporal_stride,
        )
        max_seq_len = 1 + self.raw_encoder.output_length(seq_len)
        self.tape = TAPositionalEncoding(d_model)
        self.rhythm_encoder = RRRhythmEncoder(hrv_dim, d_model, dropout)
        self.rhythm_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.rhythm_pos, std=0.02)
        self.layers = nn.ModuleList(
            [
                ConvTranEncoderLayer(
                    d_model,
                    num_heads,
                    ffn_dim,
                    max_seq_len,
                    dropout,
                    erpe_post_softmax,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_attention_pool = TokenAttentionPooling(
            d_model,
            max(d_model // 2, 32),
            dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(4 * d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, hrv_features: torch.Tensor) -> torch.Tensor:
        ecg_tokens = self.raw_encoder(x)
        ecg_tokens = ecg_tokens + self.tape(
            ecg_tokens.shape[1], ecg_tokens.device, ecg_tokens.dtype
        )
        rhythm_token = self.rhythm_encoder(hrv_features) + self.rhythm_pos
        tokens = torch.cat([rhythm_token, ecg_tokens], dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        rhythm_output = tokens[:, 0, :]
        ecg_output = tokens[:, 1:, :]
        average_pool = ecg_output.mean(dim=1)
        maximum_pool = ecg_output.max(dim=1).values
        attention_pool, _ = self.token_attention_pool(tokens)
        combined = torch.cat(
            [rhythm_output, average_pool, maximum_pool, attention_pool], dim=-1
        )
        return self.proj(combined)


class FrequencyCNNBranch(nn.Module):
    """STFT log-spectrogram, three CNN blocks, SE, and global pooling."""

    def __init__(
        self,
        in_channels: int,
        out_dim: int = 128,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int | None = None,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.1,
        use_se: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.use_se = use_se
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )
        first, second, third = cnn_channels
        self.cnn = nn.Sequential(
            ConvBNGELU2D(in_channels, first),
            ConvBNGELU2D(first, second),
            ConvBNGELU2D(second, third),
            SEBlock2D(third) if use_se else nn.Identity(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(third, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _stft_log_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        flattened = x.reshape(batch * channels, length)
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=x.device, dtype=x.dtype),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitude = torch.sqrt(
            spectrum.real.pow(2) + spectrum.imag.pow(2) + 1e-8
        )
        spectrum = torch.log1p(magnitude)
        spectrum = spectrum.reshape(
            batch, channels, spectrum.shape[-2], spectrum.shape[-1]
        )
        mean = spectrum.mean(dim=(-2, -1), keepdim=True)
        std = spectrum.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (spectrum - mean) / std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = self._stft_log_spectrogram(x)
        return self.proj(self.cnn(spectrum)), spectrum


class GatedFusion(nn.Module):
    """Learned softmax gate fusing temporal and frequency embeddings."""

    def __init__(self, dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, z_time: torch.Tensor, z_frequency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.softmax(self.gate(torch.cat([z_time, z_frequency], dim=-1)), dim=-1)
        fused = alpha[:, 0:1] * z_time + alpha[:, 1:2] * z_frequency
        return fused, alpha


class RGTFConvTran(nn.Module):
    """Rhythm-Guided Time-Frequency ConvTran.

    Inputs:
        ``x``: ECG tensor shaped ``[batch, channels, length]``.
        ``hrv_features``: RR/HRV tensor shaped ``[batch, 14]``.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        hrv_dim: int,
        num_classes: int = 2,
        d_model: int = 128,
        out_dim: int = 128,
        temporal_filters: int = 64,
        temporal_kernel: int = 8,
        temporal_stride: int = 4,
        transformer_layers: int = 2,
        heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        n_fft: int = 256,
        hop_length: int = 64,
        freq_cnn_channels: tuple[int, int, int] = (32, 64, 128),
        use_se: bool = True,
        aux_heads: bool = True,
        erpe_post_softmax: bool = True,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        self.aux_heads = aux_heads
        self.time_branch = RhythmAwareConvTranBranch(
            in_channels,
            seq_len,
            hrv_dim,
            d_model,
            out_dim,
            temporal_filters,
            temporal_kernel,
            temporal_stride,
            transformer_layers,
            heads,
            ffn_dim,
            dropout,
            erpe_post_softmax,
        )
        self.freq_branch = FrequencyCNNBranch(
            in_channels,
            out_dim,
            n_fft,
            hop_length,
            None,
            freq_cnn_channels,
            dropout,
            use_se,
        )
        self.fusion = GatedFusion(out_dim, out_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes),
        )
        if aux_heads:
            self.time_classifier = nn.Linear(out_dim, num_classes)
            self.freq_classifier = nn.Linear(out_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        hrv_features: torch.Tensor,
        return_features: bool = False,
    ) -> dict[str, Any]:
        z_time = self.time_branch(x, hrv_features)
        z_frequency, spectrum = self.freq_branch(x)
        z_fused, gate = self.fusion(z_time, z_frequency)
        output: dict[str, Any] = {
            "logits": self.classifier(z_fused),
            "gate": gate,
        }
        if self.aux_heads:
            output["logits_time"] = self.time_classifier(z_time)
            output["logits_freq"] = self.freq_classifier(z_frequency)
        if return_features:
            output.update(
                {
                    "z_time": z_time,
                    "z_freq": z_frequency,
                    "z_fused": z_fused,
                    "spectrogram": spectrum,
                }
            )
        return output
