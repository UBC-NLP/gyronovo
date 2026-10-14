"""Minimal DepthCharge-compatible components for LIPNovo."""

from __future__ import annotations

import math
import re
from typing import Sequence

import einops
import numpy as np
import torch

from .sequence import normalize_lipnovo_sequence


def listify(values):
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


class ModelMixin:
    """Small subset of the DepthCharge model mixin API."""


class PeptideMass:
    """Peptide mass calculator compatible with the existing LipNovo usage."""

    canonical = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C+57.021": 103.009184505 + 57.02146,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
    }
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        if residues == "canonical":
            self.masses = dict(self.canonical)
        else:
            self.masses = dict(residues)

    def __len__(self):
        return len(self.masses)

    def mass(self, seq: str | Sequence[str], charge: int | None = None) -> float:
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z])", seq)
        calc_mass = sum(self.masses[aa] for aa in seq) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton
        return calc_mass


class MassEncoder(torch.nn.Module):
    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        super().__init__()
        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (torch.arange(0, n_sin).float() / max(n_sin - 1, 1))
        cos_term = base * scale ** (torch.arange(0, n_cos).float() / max(n_cos - 1, 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, values):
        sin_mz = torch.sin(values / self.sin_term)
        cos_mz = torch.cos(values / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class PeakEncoder(MassEncoder):
    def __init__(
        self,
        dim_model,
        dim_intensity=None,
        min_wavelength=0.001,
        max_wavelength=10000,
    ):
        self.dim_intensity = dim_intensity
        self.dim_mz = dim_model if dim_intensity is None else dim_model - dim_intensity
        super().__init__(
            dim_model=self.dim_mz,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )
        if self.dim_intensity is None:
            self.int_encoder = torch.nn.Linear(1, dim_model, bias=False)
        else:
            self.int_encoder = MassEncoder(
                dim_model=dim_intensity,
                min_wavelength=0,
                max_wavelength=1,
            )

    def forward(self, spectra):
        encoded = super().forward(spectra[:, :, [0]])
        intensity = self.int_encoder(spectra[:, :, [1]])
        return encoded + intensity


class PositionalEncoder(torch.nn.Module):
    def __init__(self, dim_model, max_wavelength=10000):
        super().__init__()
        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        scale = max_wavelength / (2 * np.pi)
        sin_term = scale ** (torch.arange(0, n_sin).float() / max(n_sin - 1, 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / max(n_cos - 1, 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, values):
        position = torch.arange(values.shape[1], device=values.device, dtype=self.sin_term.dtype)
        position = einops.repeat(position, "n -> b n", b=values.shape[0])
        sin_in = einops.repeat(position, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(position, "b n -> b n f", f=len(self.cos_term))
        encoded = torch.cat(
            [
                torch.sin(sin_in / self.sin_term),
                torch.cos(cos_in / self.cos_term),
            ],
            axis=2,
        )
        return encoded + values


class MzRotaryEmbedding(torch.nn.Module):
    """Continuous m/z-based rotary embedding for spectrum self-attention."""

    def __init__(
        self,
        head_dim: int,
        min_wavelength: float = 1.0,
        max_wavelength: float = 10000.0,
    ):
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if min_wavelength <= 0 or max_wavelength <= 0:
            raise ValueError("m/z-RoPE wavelengths must be positive")
        if min_wavelength > max_wavelength:
            raise ValueError("min_wavelength must be <= max_wavelength")
        self.head_dim = int(head_dim)
        self.rotary_dim = (self.head_dim // 2) * 2
        if self.rotary_dim > 0:
            n_pairs = self.rotary_dim // 2
            exponents = torch.arange(n_pairs).float() / max(n_pairs - 1, 1)
            wavelengths = float(min_wavelength) * (
                float(max_wavelength) / float(min_wavelength)
            ) ** exponents
        else:
            wavelengths = torch.empty(0)
        self.register_buffer("wavelengths", wavelengths)

    def angles(self, mz_values: torch.Tensor) -> torch.Tensor:
        return (2.0 * math.pi * mz_values.unsqueeze(-1)) / self.wavelengths.to(
            device=mz_values.device,
            dtype=mz_values.dtype,
        )

    def rotate(self, values: torch.Tensor, mz_values: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim == 0:
            return values
        rotary = values[..., : self.rotary_dim]
        tail = values[..., self.rotary_dim :]
        pairs = rotary.view(*rotary.shape[:-1], -1, 2)
        angles = self.angles(mz_values).unsqueeze(1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        first = pairs[..., 0]
        second = pairs[..., 1]
        rotated = torch.stack(
            [
                first * cos - second * sin,
                first * sin + second * cos,
            ],
            dim=-1,
        ).flatten(-2)
        if tail.shape[-1] == 0:
            return rotated
        return torch.cat([rotated, tail], dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mz_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotate(query, mz_values), self.rotate(key, mz_values)


class MzRopeSelfAttention(torch.nn.Module):
    """Self-attention that rotates query/key vectors by continuous peak m/z."""

    def __init__(
        self,
        dim_model: int,
        n_head: int,
        dropout: float,
        min_wavelength: float = 1.0,
        max_wavelength: float = 10000.0,
    ):
        super().__init__()
        if dim_model % n_head != 0:
            raise ValueError("dim_model must be divisible by n_head")
        self.dim_model = dim_model
        self.n_head = n_head
        self.head_dim = dim_model // n_head
        self.qkv = torch.nn.Linear(dim_model, dim_model * 3)
        self.out_proj = torch.nn.Linear(dim_model, dim_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.rope = MzRotaryEmbedding(
            head_dim=self.head_dim,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mz_values: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = inputs.shape
        qkv = self.qkv(inputs)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        query, key = self.rope(query, key, mz_values)

        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            logits = logits.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(logits.dtype).min,
            )
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        outputs = torch.matmul(attention, value)
        outputs = outputs.transpose(1, 2).contiguous().view(batch, seq_len, self.dim_model)
        return self.out_proj(outputs)


class MzRopeTransformerLayer(torch.nn.Module):
    """Transformer encoder layer with m/z-RoPE in self-attention."""

    def __init__(
        self,
        dim_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        min_wavelength: float = 1.0,
        max_wavelength: float = 10000.0,
    ):
        super().__init__()
        self.self_attn = MzRopeSelfAttention(
            dim_model=dim_model,
            n_head=n_head,
            dropout=dropout,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )
        self.linear1 = torch.nn.Linear(dim_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, dim_model)
        self.norm1 = torch.nn.LayerNorm(dim_model)
        self.norm2 = torch.nn.LayerNorm(dim_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = torch.nn.functional.relu

    def forward(
        self,
        inputs: torch.Tensor,
        mz_values: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.self_attn(
            inputs,
            mz_values=mz_values,
            key_padding_mask=key_padding_mask,
        )
        outputs = self.norm1(inputs + self.dropout1(attended))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(outputs))))
        return self.norm2(outputs + self.dropout2(feedforward))


class SpectrumEncoder(torch.nn.Module):
    """Transformer encoder for input spectra."""

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0.0,
        peak_encoder=True,
        dim_intensity=None,
        mz_rope=None,
    ):
        super().__init__()
        mz_rope = dict(mz_rope or {})
        self.use_mz_rope = bool(mz_rope.get("enabled", False))
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        self.peak_encoder = PeakEncoder(dim_model, dim_intensity=dim_intensity) if peak_encoder else torch.nn.Linear(2, dim_model)
        if self.use_mz_rope:
            self.transformer_encoder = torch.nn.ModuleList(
                [
                    MzRopeTransformerLayer(
                        dim_model=dim_model,
                        n_head=n_head,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        min_wavelength=float(mz_rope.get("min_wavelength", 1.0)),
                        max_wavelength=float(mz_rope.get("max_wavelength", 10000.0)),
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, spectra, precursors=None, complement_augmented: bool = False):
        zeros = spectra[:, :, 1] == 0
        mask = torch.cat(
            [
                torch.zeros((spectra.shape[0], 1), device=spectra.device, dtype=torch.bool),
                zeros,
            ],
            dim=1,
        )
        peaks = self.peak_encoder(spectra)
        latent = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
        peaks = torch.cat([latent, peaks], dim=1)
        if self.use_mz_rope:
            mz_values = spectra[:, :, 0].masked_fill(zeros, 0.0)
            latent_mz = mz_values.new_zeros((mz_values.shape[0], 1))
            mz_values = torch.cat([latent_mz, mz_values], dim=1)
            encoded = peaks
            for block in self.transformer_encoder:
                encoded = block(
                    encoded,
                    mz_values=mz_values,
                    key_padding_mask=mask,
                )
            return encoded, mask
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    @property
    def device(self):
        return next(self.parameters()).device


class _PeptideTransformer(torch.nn.Module):
    def __init__(self, dim_model, pos_encoder, residues, max_charge):
        super().__init__()
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}
        self.pos_encoder = PositionalEncoder(dim_model) if pos_encoder else torch.nn.Identity()
        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_model,
            padding_idx=0,
        )

    def tokenize(self, sequence, partial=False):
        if not isinstance(sequence, str):
            return sequence
        sequence = normalize_lipnovo_sequence(sequence, self._aa2idx)
        sequence = sequence.replace("I", "L")
        tokens = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse:
            tokens = list(reversed(tokens))
        if not partial:
            tokens += ["$"]
        return torch.tensor([self._aa2idx[aa] for aa in tokens], device=self.device)

    def detokenize(self, tokens):
        sequence = [self._idx2aa.get(token.item(), "") for token in tokens]
        if "$" in sequence:
            sequence = sequence[: sequence.index("$")]
        if self.reverse:
            sequence = list(reversed(sequence))
        return sequence

    @property
    def vocab_size(self):
        return len(self._aa2idx)

    @property
    def device(self):
        return next(self.parameters()).device


class PeptideDecoder(_PeptideTransformer):
    """Transformer decoder for peptide generation."""

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0.0,
        pos_encoder=True,
        reverse=True,
        residues="canonical",
        max_charge=5,
    ):
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse
        self.mass_encoder = MassEncoder(dim_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(layer, num_layers=n_layers)
        self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)

    def forward(self, sequences, precursors, memory, memory_key_padding_mask):
        if sequences is not None:
            if torch.is_tensor(sequences):
                tokens = sequences.to(self.device, dtype=torch.long)
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
            else:
                sequences = listify(sequences)
                if sequences and all(torch.is_tensor(sequence) for sequence in sequences):
                    tokens = [sequence.to(self.device, dtype=torch.long) for sequence in sequences]
                    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
                else:
                    tokens = [self.tokenize(sequence) for sequence in sequences]
                    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.empty((memory.shape[0], 0), device=self.device, dtype=torch.long)

        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursor_tokens = masses + charges[:, None, :]
        if sequences is None:
            target = precursor_tokens
        else:
            target = torch.cat([precursor_tokens, self.aa_encoder(tokens)], dim=1)

        tgt_key_padding_mask = target.sum(axis=2) == 0
        target = self.pos_encoder(target)
        tgt_mask = generate_tgt_mask(target.shape[1]).type_as(precursor_tokens)
        predictions = self.transformer_decoder(
            tgt=target,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )
        return self.final(predictions), tokens


def generate_tgt_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask
