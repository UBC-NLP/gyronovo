"""Training-only augmentation and linear curriculum utilities for GyroNovo."""

from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np
import torch

from .sequence import normalize_lipnovo_sequence


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _config_section(value: Any, *, name: str) -> dict[str, Any]:
    """Return a validated configuration subsection."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class DifficultyVector:
    """Explicit magnitudes for GyroNovo's spectrum augmentation operators."""

    peak_missingness: float = 0.0
    b_ion_dropout: float = 0.0
    y_ion_dropout: float = 0.0
    fragmentation_ladder_gap: float = 0.0
    intensity_jitter: float = 0.0
    mz_jitter: float = 0.0
    structured_noise_count: float = 0.0
    ptm_visibility_reduction: float = 0.0

    def scaled(self, scale: float) -> "DifficultyVector":
        scale = max(_finite_float(scale, default=0.0), 0.0)
        return DifficultyVector(
            peak_missingness=self.peak_missingness * scale,
            b_ion_dropout=self.b_ion_dropout * scale,
            y_ion_dropout=self.y_ion_dropout * scale,
            fragmentation_ladder_gap=self.fragmentation_ladder_gap * scale,
            intensity_jitter=self.intensity_jitter * scale,
            mz_jitter=self.mz_jitter * scale,
            structured_noise_count=self.structured_noise_count * scale,
            ptm_visibility_reduction=self.ptm_visibility_reduction * scale,
        )

    def clamp(self, maxima: "DifficultyVector | None" = None) -> "DifficultyVector":
        if maxima is None:
            maxima = DifficultyVector(
                peak_missingness=1.0,
                b_ion_dropout=1.0,
                y_ion_dropout=1.0,
                fragmentation_ladder_gap=1.0,
                intensity_jitter=1.0,
                mz_jitter=1.0,
                structured_noise_count=max(
                    _finite_float(self.structured_noise_count, default=0.0),
                    0.0,
                ),
                ptm_visibility_reduction=1.0,
            )
        return DifficultyVector(
            **{
                field: min(
                    max(_finite_float(getattr(self, field), default=0.0), 0.0),
                    max(_finite_float(getattr(maxima, field), default=0.0), 0.0),
                )
                for field in asdict(self)
            }
        )

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
        *,
        fallback: "DifficultyVector | None" = None,
    ) -> "DifficultyVector":
        fallback = fallback or cls()
        if values is None:
            return fallback
        merged = fallback.to_dict()
        for field in merged:
            if field in values and values[field] is not None:
                merged[field] = _finite_float(values[field], default=merged[field])
        return cls(**merged)

class PeptideTokenizer:
    """Tokenize peptides with PTM-aware longest-match parsing."""

    def __init__(self, residues: Mapping[str, float]):
        self.residues = {str(key): float(value) for key, value in residues.items()}
        self.tokens = sorted(self.residues.keys(), key=len, reverse=True)

    def split(self, sequence: str | Sequence[str]) -> list[str]:
        if not isinstance(sequence, str):
            return list(sequence)
        text = str(sequence).strip()
        if not text or text.lower() in {"nan", "none"}:
            return []
        text = str(normalize_lipnovo_sequence(text, self.residues))
        parts: list[str] = []
        index = 0
        while index < len(text):
            matched = None
            for token in self.tokens:
                if text.startswith(token, index):
                    matched = token
                    break
            if matched is None:
                raise ValueError(f"Unable to tokenize peptide sequence: {sequence}")
            parts.append(matched)
            index += len(matched)
        return parts

    @staticmethod
    def is_modified(token: str) -> bool:
        return "(" in token or ")" in token or token.startswith(("+", "-"))


class TheoreticalSpectrumBuilder:
    """Rule-based b/y ion builder matching the current LIPNovo target path."""

    H_MASS = 1.0078
    H2O_MASS = 18.01056

    def __init__(
        self,
        residues: Mapping[str, float],
        *,
        config: Mapping[str, Any] | None = None,
    ):
        self.residues = {str(key): float(value) for key, value in residues.items()}
        self.tokenizer = PeptideTokenizer(self.residues)
        self.config = dict(config or {})
        self.position_baseline = float(self.config.get("position_baseline", 0.75))
        self.position_center_boost = float(self.config.get("position_center_boost", 0.35))
        self.b_ion_intensity_scale = float(self.config.get("b_ion_intensity_scale", 0.78))
        self.y_ion_intensity_scale = float(self.config.get("y_ion_intensity_scale", 1.0))
        self.ptm_intensity_boost = float(self.config.get("ptm_intensity_boost", 1.08))

    def build(
        self,
        sequence: str | Sequence[str],
        *,
        intensity_mode: str = "heuristic",
        intensity_scale: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer.split(sequence)
        length = len(tokens)
        dtype = dtype or torch.float32
        if length < 2:
            empty = torch.zeros(0, device=device, dtype=dtype)
            return {
                "mz": empty,
                "intensity": empty,
                "cleavage_index": torch.zeros(0, device=device, dtype=torch.long),
                "ion_family": torch.zeros(0, device=device, dtype=torch.long),
                "is_ptm_related": torch.zeros(0, device=device, dtype=torch.bool),
            }

        masses = [self.residues[token] for token in tokens]
        prefix_running = 0.0
        suffix_running = 0.0
        mz_values: list[float] = []
        intensity_values: list[float] = []
        cleavage_indices: list[int] = []
        ion_families: list[int] = []
        ptm_flags: list[bool] = []

        modified_prefix = False
        modified_suffix = False
        suffix_modified_flags = [self.tokenizer.is_modified(token) for token in tokens[::-1]]

        for cleavage_idx in range(length - 1):
            prefix_running += masses[cleavage_idx]
            modified_prefix = modified_prefix or self.tokenizer.is_modified(tokens[cleavage_idx])
            b_mass = prefix_running + self.H_MASS
            b_ptm = modified_prefix

            suffix_running += masses[-(cleavage_idx + 1)]
            modified_suffix = modified_suffix or suffix_modified_flags[cleavage_idx]
            y_mass = suffix_running + self.H2O_MASS + self.H_MASS
            y_ptm = modified_suffix

            rel_pos = float(cleavage_idx + 1) / float(max(length - 1, 1))
            position_shape = self.position_baseline + self.position_center_boost * (
                1.0 - abs(rel_pos - 0.5) * 2.0
            )
            b_intensity = (
                intensity_scale
                * self.b_ion_intensity_scale
                * position_shape
                * (self.ptm_intensity_boost if b_ptm else 1.0)
            )
            y_intensity = (
                intensity_scale
                * self.y_ion_intensity_scale
                * position_shape
                * (self.ptm_intensity_boost if y_ptm else 1.0)
            )

            mz_values.extend([b_mass, y_mass])
            cleavage_indices.extend([cleavage_idx, cleavage_idx])
            ion_families.extend([0, 1])
            ptm_flags.extend([b_ptm, y_ptm])
            if intensity_mode == "flat":
                intensity_values.extend([intensity_scale, intensity_scale])
            else:
                intensity_values.extend([b_intensity, y_intensity])

        order = np.argsort(np.asarray(mz_values, dtype=np.float32))
        mz_tensor = torch.tensor(np.asarray(mz_values)[order], device=device, dtype=dtype)
        intensity_tensor = torch.tensor(np.asarray(intensity_values)[order], device=device, dtype=dtype)
        cleavage_tensor = torch.tensor(np.asarray(cleavage_indices)[order], device=device, dtype=torch.long)
        family_tensor = torch.tensor(np.asarray(ion_families)[order], device=device, dtype=torch.long)
        ptm_tensor = torch.tensor(np.asarray(ptm_flags)[order], device=device, dtype=torch.bool)
        return {
            "mz": mz_tensor,
            "intensity": intensity_tensor,
            "cleavage_index": cleavage_tensor,
            "ion_family": family_tensor,
            "is_ptm_related": ptm_tensor,
        }

    def build_batch(
        self,
        sequences: Sequence[str],
        *,
        intensity_mode: str = "heuristic",
        intensity_scales: Sequence[float] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        if intensity_scales is None:
            intensity_scales = [1.0 for _ in sequences]
        items = [
            self.build(
                sequence,
                intensity_mode=intensity_mode,
                intensity_scale=float(scale),
                device=device,
                dtype=dtype,
            )
            for sequence, scale in zip(sequences, intensity_scales)
        ]
        max_len = max((item["mz"].shape[0] for item in items), default=0)
        spectra = torch.zeros(len(items), max_len, 2, device=device, dtype=dtype or torch.float32)
        cleavage_index = torch.full((len(items), max_len), -1, device=device, dtype=torch.long)
        ion_family = torch.full((len(items), max_len), -1, device=device, dtype=torch.long)
        is_ptm_related = torch.zeros(len(items), max_len, device=device, dtype=torch.bool)
        peak_mask = torch.ones(len(items), max_len, device=device, dtype=torch.bool)
        for batch_idx, item in enumerate(items):
            n_peaks = item["mz"].shape[0]
            if n_peaks == 0:
                continue
            spectra[batch_idx, :n_peaks, 0] = item["mz"]
            spectra[batch_idx, :n_peaks, 1] = item["intensity"]
            cleavage_index[batch_idx, :n_peaks] = item["cleavage_index"]
            ion_family[batch_idx, :n_peaks] = item["ion_family"]
            is_ptm_related[batch_idx, :n_peaks] = item["is_ptm_related"]
            peak_mask[batch_idx, :n_peaks] = False
        return {
            "spectra": spectra,
            "cleavage_index": cleavage_index,
            "ion_family": ion_family,
            "is_ptm_related": is_ptm_related,
            "peak_mask": peak_mask,
        }


class PeptideConditionedSpectrumAugmentor:
    """Peptide-conditioned observed-spectrum corruption for GyroNovo training."""

    def __init__(
        self,
        residues: Mapping[str, float],
        *,
        config: Mapping[str, Any] | None = None,
        n_peaks: int = 150,
    ):
        self.config = dict(config or {})
        theoretical_config = _config_section(
            self.config.get("theoretical_spectrum"),
            name="augmentation.theoretical_spectrum",
        )
        self.builder = TheoreticalSpectrumBuilder(residues, config=theoretical_config)
        self.min_mz = float(self.config.get("min_mz", 50.0))
        self.max_rendered_peaks = int(self.config.get("max_rendered_peaks", n_peaks))

        anchor_config = _config_section(
            self.config.get("observed_anchor"),
            name="augmentation.observed_anchor",
        )
        easy_focus_config = _config_section(
            anchor_config.get("easy_focus"),
            name="augmentation.observed_anchor.easy_focus",
        )
        hard_focus_config = _config_section(
            anchor_config.get("hard_focus"),
            name="augmentation.observed_anchor.hard_focus",
        )
        keep_penalty_config = _config_section(
            anchor_config.get("keep_penalties"),
            name="augmentation.observed_anchor.keep_penalties",
        )
        add_penalty_config = _config_section(
            anchor_config.get("add_penalties"),
            name="augmentation.observed_anchor.add_penalties",
        )
        probability_bounds_config = _config_section(
            anchor_config.get("probability_bounds"),
            name="augmentation.observed_anchor.probability_bounds",
        )
        fallback_config = _config_section(
            self.config.get("fallback_theoretical"),
            name="augmentation.fallback_theoretical",
        )
        ladder_gap_config = _config_section(
            self.config.get("ladder_gap"),
            name="augmentation.ladder_gap",
        )
        structured_noise_config = _config_section(
            self.config.get("structured_noise"),
            name="augmentation.structured_noise",
        )

        self.observed_anchor_enabled = bool(
            anchor_config.get(
                "enabled",
                self.config.get("observed_anchor_enabled", False),
            )
        )
        self.observed_anchor_mz_tolerance = float(
            anchor_config.get(
                "mz_tolerance",
                self.config.get("observed_anchor_mz_tolerance", 0.5),
            )
        )
        self.observed_anchor_keep_prob = float(
            anchor_config.get(
                "fragment_keep_probability",
                self.config.get("observed_anchor_keep_prob", 0.88),
            )
        )
        self.observed_anchor_noise_keep_prob = float(
            anchor_config.get(
                "unannotated_keep_probability",
                self.config.get("observed_anchor_noise_keep_prob", 0.72),
            )
        )
        self.observed_anchor_theory_add_prob = float(
            anchor_config.get(
                "missing_fragment_add_probability",
                self.config.get("observed_anchor_theory_add_prob", 0.22),
            )
        )
        legacy_easy_keep_scale = self.config.get(
            "observed_anchor_easy_focus_keep_scale",
            self.config.get(
                "observed_anchor_focus_keep_scale",
                self.config.get("observed_anchor_focus_drop_scale", 0.28),
            ),
        )
        legacy_easy_add_scale = self.config.get(
            "observed_anchor_easy_focus_add_scale",
            self.config.get("observed_anchor_focus_add_scale", 0.34),
        )
        self.observed_anchor_easy_focus_keep_scale = float(
            easy_focus_config.get("keep_probability_boost", legacy_easy_keep_scale)
        )
        self.observed_anchor_easy_focus_add_scale = float(
            easy_focus_config.get("add_probability_boost", legacy_easy_add_scale)
        )
        self.observed_anchor_easy_focus_intensity_scale = float(
            easy_focus_config.get(
                "intensity_boost",
                self.config.get("observed_anchor_easy_focus_intensity_scale", 0.20),
            )
        )
        self.observed_anchor_hard_focus_drop_scale = float(
            hard_focus_config.get(
                "keep_probability_penalty",
                self.config.get("observed_anchor_hard_focus_drop_scale", 0.55),
            )
        )
        self.observed_anchor_hard_focus_add_suppression = float(
            hard_focus_config.get(
                "add_probability_penalty",
                self.config.get("observed_anchor_hard_focus_add_suppression", 0.35),
            )
        )
        self.observed_anchor_hard_focus_force_drop_prob = float(
            hard_focus_config.get(
                "extra_drop_probability",
                self.config.get("observed_anchor_hard_focus_force_drop_prob", 0.35),
            )
        )
        self.observed_anchor_hard_focus_intensity_suppression = float(
            hard_focus_config.get(
                "intensity_penalty",
                self.config.get("observed_anchor_hard_focus_intensity_suppression", 0.45),
            )
        )
        # The generic focus attributes are retained as compatibility aliases.
        self.observed_anchor_focus_keep_scale = self.observed_anchor_easy_focus_keep_scale
        self.observed_anchor_focus_add_scale = self.observed_anchor_easy_focus_add_scale

        self.observed_anchor_peak_missingness_keep_penalty = float(
            keep_penalty_config.get("peak_missingness", 0.22)
        )
        self.observed_anchor_ion_dropout_keep_penalty = float(
            keep_penalty_config.get("ion_dropout", 0.22)
        )
        self.observed_anchor_ptm_visibility_keep_penalty = float(
            keep_penalty_config.get("ptm_visibility", 0.12)
        )
        self.observed_anchor_peak_missingness_add_penalty = float(
            add_penalty_config.get("peak_missingness", 0.15)
        )
        self.observed_anchor_ptm_visibility_add_penalty = float(
            add_penalty_config.get("ptm_visibility", 0.15)
        )
        self.observed_anchor_keep_probability_min = float(
            probability_bounds_config.get("keep_min", 0.05)
        )
        self.observed_anchor_keep_probability_max = float(
            probability_bounds_config.get("keep_max", 1.0)
        )
        self.observed_anchor_add_probability_min = float(
            probability_bounds_config.get("add_min", 0.02)
        )
        self.observed_anchor_hard_add_probability_min = float(
            probability_bounds_config.get("hard_add_min", 0.0)
        )
        self.observed_anchor_add_probability_max = float(
            probability_bounds_config.get("add_max", 0.92)
        )
        self.observed_anchor_easy_ladder_focus_threshold = float(
            anchor_config.get("easy_ladder_focus_threshold", 0.5)
        )

        self.fallback_ptm_dropout_scale = float(fallback_config.get("ptm_dropout_scale", 0.5))
        self.fallback_max_drop_probability = float(
            fallback_config.get("maximum_drop_probability", 0.95)
        )
        self.ladder_gap_max_run_scale = float(ladder_gap_config.get("max_run_scale", 3.0))

        neutral_losses = np.asarray(
            structured_noise_config.get("neutral_loss_masses", [18.010565, 17.026549]),
            dtype=np.float32,
        ).reshape(-1)
        if neutral_losses.size == 0 or not np.isfinite(neutral_losses).all():
            raise ValueError("augmentation.structured_noise.neutral_loss_masses must be finite and non-empty")
        self.structured_noise_neutral_loss_masses = neutral_losses
        self.structured_noise_mz_jitter_std = float(
            structured_noise_config.get("mz_jitter_std", 0.01)
        )
        intensity_range = structured_noise_config.get("intensity_fraction_range", [0.08, 0.35])
        if not isinstance(intensity_range, Sequence) or len(intensity_range) != 2:
            raise ValueError("augmentation.structured_noise.intensity_fraction_range must have two values")
        intensity_low, intensity_high = sorted(float(value) for value in intensity_range)
        self.structured_noise_intensity_fraction_min = intensity_low
        self.structured_noise_intensity_fraction_max = intensity_high
        self.structured_noise_reference_intensity_cap = float(
            structured_noise_config.get("reference_intensity_cap", 0.35)
        )
        self.structured_noise_observed_anchor_count_scale = float(
            structured_noise_config.get("observed_anchor_count_scale", 0.35)
        )
        self._validate_configuration()
        self.last_activation_stats = self._empty_activation_stats()

    def _validate_configuration(self) -> None:
        probabilities = {
            "observed_anchor.fragment_keep_probability": self.observed_anchor_keep_prob,
            "observed_anchor.unannotated_keep_probability": self.observed_anchor_noise_keep_prob,
            "observed_anchor.missing_fragment_add_probability": self.observed_anchor_theory_add_prob,
            "observed_anchor.easy_focus.keep_probability_boost": self.observed_anchor_easy_focus_keep_scale,
            "observed_anchor.easy_focus.add_probability_boost": self.observed_anchor_easy_focus_add_scale,
            "observed_anchor.hard_focus.keep_probability_penalty": self.observed_anchor_hard_focus_drop_scale,
            "observed_anchor.hard_focus.add_probability_penalty": self.observed_anchor_hard_focus_add_suppression,
            "observed_anchor.hard_focus.extra_drop_probability": self.observed_anchor_hard_focus_force_drop_prob,
            "observed_anchor.easy_ladder_focus_threshold": self.observed_anchor_easy_ladder_focus_threshold,
            "observed_anchor.probability_bounds.keep_min": self.observed_anchor_keep_probability_min,
            "observed_anchor.probability_bounds.keep_max": self.observed_anchor_keep_probability_max,
            "observed_anchor.probability_bounds.add_min": self.observed_anchor_add_probability_min,
            "observed_anchor.probability_bounds.hard_add_min": self.observed_anchor_hard_add_probability_min,
            "observed_anchor.probability_bounds.add_max": self.observed_anchor_add_probability_max,
            "fallback_theoretical.maximum_drop_probability": self.fallback_max_drop_probability,
        }
        for name, value in probabilities.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"augmentation.{name} must be between 0 and 1")

        nonnegative_values = {
            "observed_anchor.mz_tolerance": self.observed_anchor_mz_tolerance,
            "observed_anchor.easy_focus.intensity_boost": self.observed_anchor_easy_focus_intensity_scale,
            "observed_anchor.hard_focus.intensity_penalty": self.observed_anchor_hard_focus_intensity_suppression,
            "observed_anchor.keep_penalties.peak_missingness": self.observed_anchor_peak_missingness_keep_penalty,
            "observed_anchor.keep_penalties.ion_dropout": self.observed_anchor_ion_dropout_keep_penalty,
            "observed_anchor.keep_penalties.ptm_visibility": self.observed_anchor_ptm_visibility_keep_penalty,
            "observed_anchor.add_penalties.peak_missingness": self.observed_anchor_peak_missingness_add_penalty,
            "observed_anchor.add_penalties.ptm_visibility": self.observed_anchor_ptm_visibility_add_penalty,
            "fallback_theoretical.ptm_dropout_scale": self.fallback_ptm_dropout_scale,
            "ladder_gap.max_run_scale": self.ladder_gap_max_run_scale,
            "structured_noise.mz_jitter_std": self.structured_noise_mz_jitter_std,
            "structured_noise.intensity_fraction_range[0]": self.structured_noise_intensity_fraction_min,
            "structured_noise.intensity_fraction_range[1]": self.structured_noise_intensity_fraction_max,
            "structured_noise.reference_intensity_cap": self.structured_noise_reference_intensity_cap,
            "structured_noise.observed_anchor_count_scale": self.structured_noise_observed_anchor_count_scale,
        }
        for name, value in nonnegative_values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"augmentation.{name} must be finite and nonnegative")

        if self.observed_anchor_keep_probability_min > self.observed_anchor_keep_probability_max:
            raise ValueError("augmentation.observed_anchor keep_min cannot exceed keep_max")
        if self.observed_anchor_add_probability_min > self.observed_anchor_add_probability_max:
            raise ValueError("augmentation.observed_anchor add_min cannot exceed add_max")
        if self.observed_anchor_hard_add_probability_min > self.observed_anchor_add_probability_max:
            raise ValueError("augmentation.observed_anchor hard_add_min cannot exceed add_max")

    def render_batch(
        self,
        sequences: Sequence[str],
        precursors: torch.Tensor,
        reference_spectra: torch.Tensor,
        difficulties: Sequence[DifficultyVector],
        *,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        target_focus_weights: torch.Tensor | None = None,
        view_policy: str | None = None,
    ) -> torch.Tensor:
        spectra: list[torch.Tensor] = []
        activation_sums = self._empty_activation_stats()
        for index, (sequence, precursor, difficulty) in enumerate(zip(sequences, precursors, difficulties)):
            rng = np.random.default_rng(seed + index)
            reference = reference_spectra[index].detach().cpu()
            precursor_mass = float(precursor[0].detach().cpu())
            focus_weights = None
            if target_focus_weights is not None:
                focus_weights = target_focus_weights[index].detach().cpu()
            augmented, stats = self._render_single(
                sequence=sequence,
                precursor_mass=precursor_mass,
                reference_spectrum=reference,
                difficulty=difficulty,
                rng=rng,
                target_focus_weights=focus_weights,
                view_policy=view_policy,
            )
            spectra.append(augmented.to(device=device, dtype=dtype))
            for key, value in stats.items():
                activation_sums[key] += float(value)
        denom = max(len(spectra), 1)
        self.last_activation_stats = {
            key: float(value) / float(denom) for key, value in activation_sums.items()
        }
        return torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)

    def _render_single(
        self,
        *,
        sequence: str,
        precursor_mass: float,
        reference_spectrum: torch.Tensor,
        difficulty: DifficultyVector,
        rng: np.random.Generator,
        target_focus_weights: torch.Tensor | np.ndarray | None = None,
        view_policy: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        difficulty = difficulty.clamp()
        reference_intensity = (
            _finite_float(reference_spectrum[:, 1].max().item(), default=1.0)
            if reference_spectrum.numel()
            else 1.0
        )
        annotated = self.builder.build(
            sequence,
            intensity_mode="heuristic",
            intensity_scale=max(reference_intensity, 1.0),
        )
        mz = annotated["mz"].detach().cpu().numpy().astype(np.float32)
        intensity = annotated["intensity"].detach().cpu().numpy().astype(np.float32)
        cleavage_index = annotated["cleavage_index"].detach().cpu().numpy().astype(np.int64)
        ion_family = annotated["ion_family"].detach().cpu().numpy().astype(np.int64)
        ptm_mask = annotated["is_ptm_related"].detach().cpu().numpy().astype(bool)
        stats = self._empty_activation_stats()

        if mz.size == 0:
            return torch.zeros(1, 2, dtype=torch.float32), stats

        if self.observed_anchor_enabled:
            anchored = self._render_observed_anchored_single(
                mz=mz,
                intensity=intensity,
                cleavage_index=cleavage_index,
                ion_family=ion_family,
                ptm_mask=ptm_mask,
                precursor_mass=precursor_mass,
                reference_spectrum=reference_spectrum,
                reference_intensity=reference_intensity,
                difficulty=difficulty,
                rng=rng,
                target_focus_weights=target_focus_weights,
                view_policy=view_policy,
            )
            if anchored is not None:
                return anchored

        focus = self._normalise_focus_weights(target_focus_weights, mz.shape[0])
        policy = str(view_policy or "").lower()
        drop_prob = np.full(mz.shape, difficulty.peak_missingness, dtype=np.float32)
        drop_prob += (ion_family == 0).astype(np.float32) * difficulty.b_ion_dropout
        drop_prob += (ion_family == 1).astype(np.float32) * difficulty.y_ion_dropout
        drop_prob += ptm_mask.astype(np.float32) * (
            self.fallback_ptm_dropout_scale * difficulty.ptm_visibility_reduction
        )
        if policy == "easy_decoder_focused":
            drop_prob -= focus * self.observed_anchor_easy_focus_keep_scale
        elif policy == "hard_decoder_focused":
            drop_prob += focus * self.observed_anchor_hard_focus_drop_scale
        ladder_gap_mask = self._sample_ladder_gap_mask(
            cleavage_index=cleavage_index,
            strength=difficulty.fragmentation_ladder_gap,
            rng=rng,
        )

        drop_prob = np.clip(drop_prob, 0.0, self.fallback_max_drop_probability)
        keep_mask = rng.random(mz.shape[0]) >= drop_prob
        if ladder_gap_mask.any():
            keep_mask[ladder_gap_mask] = False
        if not keep_mask.any():
            keep_mask[np.argmax(intensity)] = True
        removed = ~keep_mask
        stats["peak_missingness"] = float(removed.mean())
        stats["b_ion_dropout"] = float(removed[ion_family == 0].mean()) if (ion_family == 0).any() else 0.0
        stats["y_ion_dropout"] = float(removed[ion_family == 1].mean()) if (ion_family == 1).any() else 0.0
        stats["fragmentation_ladder_gap"] = float(ladder_gap_mask.any())

        mz = mz[keep_mask]
        intensity = intensity[keep_mask]
        cleavage_index = cleavage_index[keep_mask]
        ion_family = ion_family[keep_mask]
        ptm_mask = ptm_mask[keep_mask]

        if intensity.size:
            distortion = 1.0 + rng.normal(0.0, difficulty.intensity_jitter, size=intensity.shape[0])
            intensity = intensity * np.clip(distortion.astype(np.float32), 0.05, None)
            stats["intensity_jitter"] = float(np.abs(distortion - 1.0).mean())
            if difficulty.ptm_visibility_reduction > 0.0 and ptm_mask.any():
                ptm_scale = 1.0 - rng.uniform(
                    0.0,
                    difficulty.ptm_visibility_reduction,
                    size=ptm_mask.sum(),
                ).astype(np.float32)
                intensity[ptm_mask] *= np.clip(ptm_scale, 0.05, 1.0)
                stats["ptm_visibility_reduction"] = float(1.0 - np.clip(ptm_scale, 0.05, 1.0).mean())
            mz_delta = rng.normal(0.0, difficulty.mz_jitter, size=mz.shape[0]).astype(np.float32)
            mz = mz + mz_delta
            stats["mz_jitter"] = float(np.abs(mz_delta).mean())

        noise_lambda = max(difficulty.structured_noise_count, 0.0)
        noise_count = int(rng.poisson(noise_lambda))
        added_noise_count = 0
        if noise_count > 0:
            noise_mz, noise_intensity = self._sample_structured_noise(
                parent_mz=mz,
                parent_intensity=intensity,
                count=noise_count,
                rng=rng,
                reference_intensity=reference_intensity,
            )
            if noise_mz.size:
                mz = np.concatenate([mz, noise_mz], axis=0)
                intensity = np.concatenate([intensity, noise_intensity], axis=0)
                added_noise_count = int(noise_mz.size)
        stats["structured_noise_count"] = float(added_noise_count)

        mz = np.clip(mz, 1e-3, None)
        intensity = np.clip(intensity, 1e-4, None)

        if mz.shape[0] > self.max_rendered_peaks:
            keep = np.argsort(intensity)[-self.max_rendered_peaks :]
            mz = mz[keep]
            intensity = intensity[keep]

        order = np.argsort(mz)
        spectrum = np.stack([mz[order], intensity[order]], axis=-1).astype(np.float32)
        return torch.tensor(spectrum, dtype=torch.float32), stats

    def _render_observed_anchored_single(
        self,
        *,
        mz: np.ndarray,
        intensity: np.ndarray,
        cleavage_index: np.ndarray,
        ion_family: np.ndarray,
        ptm_mask: np.ndarray,
        precursor_mass: float,
        reference_spectrum: torch.Tensor,
        reference_intensity: float,
        difficulty: DifficultyVector,
        rng: np.random.Generator,
        target_focus_weights: torch.Tensor | np.ndarray | None,
        view_policy: str | None,
    ) -> tuple[torch.Tensor, dict[str, float]] | None:
        reference = reference_spectrum.detach().cpu().numpy().astype(np.float32)
        if reference.ndim != 2 or reference.shape[-1] != 2:
            return None
        observed_mask = (
            np.isfinite(reference[:, 0])
            & np.isfinite(reference[:, 1])
            & (reference[:, 0] >= self.min_mz)
            & (reference[:, 1] > 0)
        )
        if not observed_mask.any():
            return None

        obs_mz = reference[observed_mask, 0].copy()
        obs_intensity = reference[observed_mask, 1].copy()
        focus = self._normalise_focus_weights(target_focus_weights, mz.shape[0])
        policy = str(view_policy or "").lower()
        easy_decoder_focused = policy == "easy_decoder_focused"
        hard_decoder_focused = policy == "hard_decoder_focused"

        delta = np.abs(mz[:, None] - obs_mz[None, :])
        nearest_obs = np.argmin(delta, axis=1)
        nearest_delta = delta[np.arange(mz.shape[0]), nearest_obs]
        matched = nearest_delta <= self.observed_anchor_mz_tolerance

        obs_focus = np.zeros(obs_mz.shape[0], dtype=np.float32)
        obs_is_fragment = np.zeros(obs_mz.shape[0], dtype=bool)
        obs_theoretical_idx = np.full(obs_mz.shape[0], -1, dtype=np.int64)
        for theo_idx in np.where(matched)[0]:
            obs_idx = int(nearest_obs[theo_idx])
            obs_is_fragment[obs_idx] = True
            obs_focus[obs_idx] = max(float(obs_focus[obs_idx]), float(focus[theo_idx]))
            obs_theoretical_idx[obs_idx] = int(theo_idx)

        ladder_gap_mask = self._sample_ladder_gap_mask(
            cleavage_index=cleavage_index,
            strength=difficulty.fragmentation_ladder_gap,
            rng=rng,
        )
        obs_b = np.zeros(obs_mz.shape[0], dtype=np.float32)
        obs_y = np.zeros(obs_mz.shape[0], dtype=np.float32)
        obs_ptm = np.zeros(obs_mz.shape[0], dtype=np.float32)
        obs_ladder = np.zeros(obs_mz.shape[0], dtype=bool)
        matched_obs = obs_theoretical_idx >= 0
        if matched_obs.any():
            parent_idx = obs_theoretical_idx[matched_obs]
            obs_b[matched_obs] = (ion_family[parent_idx] == 0).astype(np.float32)
            obs_y[matched_obs] = (ion_family[parent_idx] == 1).astype(np.float32)
            obs_ptm[matched_obs] = ptm_mask[parent_idx].astype(np.float32)
            obs_ladder[matched_obs] = ladder_gap_mask[parent_idx]

        keep_prob = np.full(obs_mz.shape, self.observed_anchor_noise_keep_prob, dtype=np.float32)
        keep_prob[obs_is_fragment] = self.observed_anchor_keep_prob
        keep_prob -= self.observed_anchor_peak_missingness_keep_penalty * difficulty.peak_missingness
        keep_prob -= self.observed_anchor_ion_dropout_keep_penalty * obs_b * difficulty.b_ion_dropout
        keep_prob -= self.observed_anchor_ion_dropout_keep_penalty * obs_y * difficulty.y_ion_dropout
        keep_prob -= (
            self.observed_anchor_ptm_visibility_keep_penalty
            * obs_ptm
            * difficulty.ptm_visibility_reduction
        )
        if hard_decoder_focused:
            keep_prob -= obs_focus * self.observed_anchor_hard_focus_drop_scale
        elif easy_decoder_focused:
            keep_prob += obs_focus * self.observed_anchor_easy_focus_keep_scale
        else:
            keep_prob += obs_focus * self.observed_anchor_focus_keep_scale
        keep_prob = np.clip(
            keep_prob,
            self.observed_anchor_keep_probability_min,
            self.observed_anchor_keep_probability_max,
        )
        observed_keep = rng.random(obs_mz.shape[0]) < keep_prob
        if hard_decoder_focused and self.observed_anchor_hard_focus_force_drop_prob > 0.0:
            forced_drop_prob = np.clip(
                obs_focus * self.observed_anchor_hard_focus_force_drop_prob,
                0.0,
                self.observed_anchor_keep_probability_max,
            )
            forced_drop = obs_is_fragment & (rng.random(obs_mz.shape[0]) < forced_drop_prob)
            observed_keep[forced_drop] = False
        if obs_ladder.any():
            ladder_drop = obs_ladder.copy()
            if easy_decoder_focused:
                ladder_drop &= obs_focus < self.observed_anchor_easy_ladder_focus_threshold
            observed_keep[ladder_drop] = False
        if easy_decoder_focused and self.observed_anchor_easy_focus_intensity_scale > 0.0:
            obs_intensity *= 1.0 + obs_focus * self.observed_anchor_easy_focus_intensity_scale
        elif hard_decoder_focused and self.observed_anchor_hard_focus_intensity_suppression > 0.0:
            obs_intensity *= np.clip(
                1.0 - obs_focus * self.observed_anchor_hard_focus_intensity_suppression,
                0.05,
                1.0,
            )

        matched_and_kept = matched & observed_keep[nearest_obs]
        missing_or_dropped = ~matched_and_kept
        add_prob = np.full(mz.shape, self.observed_anchor_theory_add_prob, dtype=np.float32)
        if hard_decoder_focused:
            add_prob -= focus * self.observed_anchor_hard_focus_add_suppression
        elif easy_decoder_focused:
            add_prob += focus * self.observed_anchor_easy_focus_add_scale
        else:
            add_prob += focus * self.observed_anchor_focus_add_scale
        add_prob -= self.observed_anchor_peak_missingness_add_penalty * difficulty.peak_missingness
        add_prob -= (
            ptm_mask.astype(np.float32)
            * self.observed_anchor_ptm_visibility_add_penalty
            * difficulty.ptm_visibility_reduction
        )
        add_prob = np.clip(
            add_prob,
            self.observed_anchor_hard_add_probability_min
            if hard_decoder_focused
            else self.observed_anchor_add_probability_min,
            self.observed_anchor_add_probability_max,
        )
        if ladder_gap_mask.any():
            if easy_decoder_focused:
                add_prob[
                    ladder_gap_mask
                    & (focus <= self.observed_anchor_easy_ladder_focus_threshold)
                ] = 0.0
            else:
                add_prob[ladder_gap_mask] = 0.0
        add_theoretical = missing_or_dropped & (rng.random(mz.shape[0]) < add_prob)

        rendered_mz = [obs_mz[observed_keep]]
        rendered_intensity = [obs_intensity[observed_keep]]
        if add_theoretical.any():
            add_mz = mz[add_theoretical].copy()
            add_intensity = intensity[add_theoretical].copy()
            distortion = 1.0 + rng.normal(
                0.0,
                difficulty.intensity_jitter,
                size=add_intensity.shape[0],
            )
            add_intensity *= np.clip(distortion.astype(np.float32), 0.05, None)
            add_mz += rng.normal(0.0, difficulty.mz_jitter, size=add_mz.shape[0]).astype(np.float32)
            rendered_mz.append(add_mz)
            rendered_intensity.append(add_intensity)

        noise_lambda = max(
            self.structured_noise_observed_anchor_count_scale
            * difficulty.structured_noise_count,
            0.0,
        )
        noise_count = int(rng.poisson(noise_lambda))
        added_noise_count = 0
        if noise_count > 0:
            noise_mz, noise_intensity = self._sample_structured_noise(
                parent_mz=mz,
                parent_intensity=intensity,
                count=noise_count,
                rng=rng,
                reference_intensity=reference_intensity,
            )
            if noise_mz.size:
                rendered_mz.append(noise_mz)
                rendered_intensity.append(noise_intensity)
                added_noise_count = int(noise_mz.size)

        out_mz = np.concatenate(rendered_mz, axis=0) if rendered_mz else np.zeros(0, dtype=np.float32)
        out_intensity = (
            np.concatenate(rendered_intensity, axis=0)
            if rendered_intensity
            else np.zeros(0, dtype=np.float32)
        )
        valid = np.isfinite(out_mz) & np.isfinite(out_intensity) & (out_mz > 0) & (out_intensity > 0)
        out_mz = out_mz[valid]
        out_intensity = out_intensity[valid]
        if out_mz.size == 0:
            strongest = int(np.argmax(intensity))
            out_mz = mz[[strongest]]
            out_intensity = intensity[[strongest]]

        if out_mz.shape[0] > self.max_rendered_peaks:
            keep = np.argsort(out_intensity)[-self.max_rendered_peaks :]
            out_mz = out_mz[keep]
            out_intensity = out_intensity[keep]

        out_mz = np.clip(out_mz, 1e-3, None)
        out_intensity = np.clip(out_intensity, 1e-4, None)
        order = np.argsort(out_mz)
        spectrum = np.stack([out_mz[order], out_intensity[order]], axis=-1).astype(np.float32)
        stats = self._empty_activation_stats()
        removed_fragment_obs = obs_is_fragment & ~observed_keep
        stats["peak_missingness"] = (
            float(removed_fragment_obs.sum()) / float(max(obs_is_fragment.sum(), 1))
        )
        stats["b_ion_dropout"] = (
            float((removed_fragment_obs & (obs_b > 0)).sum()) / float(max((obs_b > 0).sum(), 1))
        )
        stats["y_ion_dropout"] = (
            float((removed_fragment_obs & (obs_y > 0)).sum()) / float(max((obs_y > 0).sum(), 1))
        )
        stats["fragmentation_ladder_gap"] = float(ladder_gap_mask.any())
        stats["intensity_jitter"] = float(difficulty.intensity_jitter if add_theoretical.any() else 0.0)
        stats["mz_jitter"] = float(difficulty.mz_jitter if add_theoretical.any() else 0.0)
        stats["structured_noise_count"] = float(added_noise_count)
        stats["ptm_visibility_reduction"] = float(difficulty.ptm_visibility_reduction if obs_ptm.any() else 0.0)
        return torch.tensor(spectrum, dtype=torch.float32), stats

    @staticmethod
    def _empty_activation_stats() -> dict[str, float]:
        return {
            "peak_missingness": 0.0,
            "b_ion_dropout": 0.0,
            "y_ion_dropout": 0.0,
            "fragmentation_ladder_gap": 0.0,
            "intensity_jitter": 0.0,
            "mz_jitter": 0.0,
            "structured_noise_count": 0.0,
            "ptm_visibility_reduction": 0.0,
        }

    def _sample_ladder_gap_mask(
        self,
        *,
        cleavage_index: np.ndarray,
        strength: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = np.zeros(cleavage_index.shape, dtype=bool)
        unique_cleavages = np.unique(cleavage_index[cleavage_index >= 0])
        strength = min(max(_finite_float(strength, default=0.0), 0.0), 1.0)
        if unique_cleavages.size == 0 or strength <= 0.0 or rng.random() > strength:
            return mask
        max_run = min(
            unique_cleavages.size,
            max(1, 1 + int(round(self.ladder_gap_max_run_scale * strength))),
        )
        run_length = int(rng.integers(1, max_run + 1))
        start = int(rng.integers(0, unique_cleavages.size - run_length + 1))
        selected = unique_cleavages[start : start + run_length]
        return np.isin(cleavage_index, selected)

    def _sample_structured_noise(
        self,
        *,
        parent_mz: np.ndarray,
        parent_intensity: np.ndarray,
        count: int,
        rng: np.random.Generator,
        reference_intensity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if count <= 0 or parent_mz.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        offsets = self.structured_noise_neutral_loss_masses
        eligible = np.where(parent_mz > (self.min_mz + float(offsets.min())))[0]
        if eligible.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        parent_idx = rng.choice(eligible, size=count, replace=True)
        chosen_offsets = rng.choice(offsets, size=count, replace=True)
        noise_mz = parent_mz[parent_idx] - chosen_offsets
        noise_mz += rng.normal(
            0.0,
            self.structured_noise_mz_jitter_std,
            size=count,
        ).astype(np.float32)
        parent_scale = np.clip(parent_intensity[parent_idx], 1e-4, None)
        fallback_scale = max(_finite_float(reference_intensity, default=1.0), 1.0)
        noise_intensity = np.minimum(
            parent_scale
            * rng.uniform(
                self.structured_noise_intensity_fraction_min,
                self.structured_noise_intensity_fraction_max,
                size=count,
            ).astype(np.float32),
            fallback_scale * self.structured_noise_reference_intensity_cap,
        )
        valid = noise_mz >= self.min_mz
        return noise_mz[valid].astype(np.float32), noise_intensity[valid].astype(np.float32)

    @staticmethod
    def _normalise_focus_weights(
        target_focus_weights: torch.Tensor | np.ndarray | None,
        length: int,
    ) -> np.ndarray:
        if target_focus_weights is None or length <= 0:
            return np.zeros(max(length, 0), dtype=np.float32)
        raw = np.asarray(target_focus_weights, dtype=np.float32).reshape(-1)
        if raw.shape[0] < length:
            raw = np.pad(raw, (0, length - raw.shape[0]), constant_values=1.0)
        raw = raw[:length]
        raw = np.nan_to_num(raw, nan=1.0, posinf=1.0, neginf=1.0)
        focus = np.clip(raw - 1.0, 0.0, None)
        max_focus = float(focus.max()) if focus.size else 0.0
        if not math.isfinite(max_focus) or max_focus <= 1e-8:
            return np.zeros(length, dtype=np.float32)
        return (focus / max_focus).astype(np.float32)

class DifficultyScheduler:
    """Static or linear easy-to-hard curriculum used by the retained workflows."""

    def __init__(
        self,
        residues: Mapping[str, float],
        *,
        config: Mapping[str, Any] | None = None,
    ):
        self.config = dict(config or {})
        self.mode = str(self.config.get("mode", "static")).lower()
        if self.mode not in {"static", "linear"}:
            raise ValueError(f"Unsupported curriculum mode: {self.mode}")
        self.initial = DifficultyVector.from_mapping(
            self.config.get("initial_difficulty"),
            fallback=DifficultyVector(
                peak_missingness=0.03,
                b_ion_dropout=0.02,
                y_ion_dropout=0.02,
                fragmentation_ladder_gap=0.0,
                intensity_jitter=0.04,
                mz_jitter=0.01,
                structured_noise_count=2.0,
                ptm_visibility_reduction=0.02,
            ),
        )
        self.maximum = DifficultyVector.from_mapping(
            self.config.get("max_difficulty"),
            fallback=DifficultyVector(
                peak_missingness=0.45,
                b_ion_dropout=0.35,
                y_ion_dropout=0.42,
                fragmentation_ladder_gap=0.8,
                intensity_jitter=0.4,
                mz_jitter=0.18,
                structured_noise_count=24.0,
                ptm_visibility_reduction=0.75,
            ),
        )
        self.ptm_visibility_multiplier = float(
            self.config.get("ptm_visibility_multiplier", 1.15)
        )
        if not math.isfinite(self.ptm_visibility_multiplier) or self.ptm_visibility_multiplier < 0.0:
            raise ValueError("curriculum.ptm_visibility_multiplier must be finite and nonnegative")
        self.tokenizer = PeptideTokenizer(residues)

    def difficulty_for_sample(
        self,
        sequence: str,
        *,
        progress: float,
        view_scale: float = 1.0,
    ) -> DifficultyVector:
        if self.mode == "linear":
            base = self._interpolate(self.initial, self.maximum, min(max(progress, 0.0), 1.0))
        else:
            base = self.initial

        difficulty = base
        tokens = self.tokenizer.split(sequence)
        has_ptm = any(self.tokenizer.is_modified(token) for token in tokens)

        difficulty = difficulty.scaled(view_scale)
        difficulty = DifficultyVector(
            peak_missingness=difficulty.peak_missingness,
            b_ion_dropout=difficulty.b_ion_dropout,
            y_ion_dropout=difficulty.y_ion_dropout,
            fragmentation_ladder_gap=difficulty.fragmentation_ladder_gap,
            intensity_jitter=difficulty.intensity_jitter,
            mz_jitter=difficulty.mz_jitter,
            structured_noise_count=difficulty.structured_noise_count,
            ptm_visibility_reduction=difficulty.ptm_visibility_reduction
            * (self.ptm_visibility_multiplier if has_ptm else 1.0),
        )
        return difficulty.clamp(self.maximum)

    def batch_difficulties(
        self,
        sequences: Sequence[str],
        *,
        progress: float,
        view_scale: float = 1.0,
    ) -> list[DifficultyVector]:
        return [
            self.difficulty_for_sample(sequence, progress=progress, view_scale=view_scale)
            for sequence in sequences
        ]

    def current_logging_difficulty(self, progress: float) -> DifficultyVector:
        return (
            self._interpolate(self.initial, self.maximum, min(max(progress, 0.0), 1.0))
            if self.mode == "linear"
            else self.initial
        )

    @staticmethod
    def _interpolate(
        start: DifficultyVector,
        end: DifficultyVector,
        amount: float | DifficultyVector,
    ) -> DifficultyVector:
        if isinstance(amount, DifficultyVector):
            ratios = amount
        else:
            ratios = DifficultyVector(
                peak_missingness=float(amount),
                b_ion_dropout=float(amount),
                y_ion_dropout=float(amount),
                fragmentation_ladder_gap=float(amount),
                intensity_jitter=float(amount),
                mz_jitter=float(amount),
                structured_noise_count=float(amount),
                ptm_visibility_reduction=float(amount),
            )
        start_values = start.to_dict()
        end_values = end.to_dict()
        ratio_values = ratios.to_dict()
        interpolated = {}
        for key in start_values:
            interpolated[key] = start_values[key] + (end_values[key] - start_values[key]) * ratio_values[key]
        return DifficultyVector(**interpolated)
