"""
Extended from https://github.com/Noble-Lab/casanovo
Reference
----------
Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo mass spectrometry peptide sequencing with a transformer model. 
in Proceedings of the 39th International Conference on Machine Learning - ICML '22 vol. 162 25514–25522 (PMLR, 2022). https://proceedings.mlr.press/v162/yilmaz22a.html

"""

import collections
import heapq
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import einops
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .augmentation import DifficultyVector
from .augmentation import DifficultyScheduler
from .augmentation import PeptideConditionedSpectrumAugmentor
from .augmentation import TheoreticalSpectrumBuilder
from .depthcharge_compat import ModelMixin
from .depthcharge_compat import PeptideDecoder
from .depthcharge_compat import PeptideMass
from .depthcharge_compat import SpectrumEncoder
from .failure_profile import build_failure_profile
from .failure_profile import flatten_failure_profile_for_logging
from .sequence import normalize_lipnovo_sequence

logger = logging.getLogger("lipnovo")



class MLP(pl.LightningModule, ModelMixin):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class Imputator(pl.LightningModule, ModelMixin):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        n_enc_layers=3,
        n_dec_layers=3
        ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_enc_layers
        ) if n_enc_layers > 0 else None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout)

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_dec_layers
        )

        self.pred_head = MLP(
            input_dim=dim_model, 
            hidden_dim=dim_model*4, 
            output_dim=dim_model,
            num_layers=3)
        
        self.cls_head = MLP(
            input_dim=dim_model, 
            hidden_dim=dim_model*4, 
            output_dim=1,
            num_layers=3)
    
    def forward(self, src, masks, query_embed):
        """
        memories: extracted memories from exp peaks
        masks: masks
        """
        bs, l, dim = src.shape

        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        memories = self.encoder(src, src_key_padding_mask=masks) if self.encoder is not None else src

        out = self.decoder(
            tgt=query_embed,
            memory=memories,
            memory_key_padding_mask=masks,
        )

        pred = self.pred_head(out)
        cls_logit = self.cls_head(out)

        return pred, cls_logit


class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    max_length : int
        The maximum peptide length to decode.
    residues: Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams: int
        Number of beams used during beam search decoding.
    top_match: int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    train_label_smoothing: float
        Smoothing factor when calculating the training loss.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 1,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        saved_path: str = "",
        validation_decode_fraction: float = 0.1,
        gen_num = 50,
        gen_enc_layers=3,
        gen_dec_layers=3,
        gen_threshold=0.8,
        n_peaks: int = 150,
        random_seed: int = 0,
        augmentation: Optional[Dict[str, Any]] = None,
        curriculum: Optional[Dict[str, Any]] = None,
        auxiliary_losses: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        mz_rope: Optional[Dict[str, Any]] = None,
        training_code: str = "extended",
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gen_num = gen_num
        self.gen_enc_layers = gen_enc_layers
        self.gen_dec_layers = gen_dec_layers
        self.gen_threshold = gen_threshold
        self.validation_decode_fraction = min(max(float(validation_decode_fraction), 0.0), 1.0)
        self._validation_records: list[dict[str, float | str]] = []
        self.random_seed = int(random_seed)
        self.n_peaks = int(n_peaks)
        self.augmentation_config = dict(augmentation or {})
        self.curriculum_config = dict(curriculum or {})
        self.auxiliary_config = dict(auxiliary_losses or {})
        self.analysis_config = dict(analysis or {})
        self.mz_rope_config = dict(mz_rope or {})
        self.training_code = str(training_code or "extended").lower()
        self.use_original_lipnovo_training = self.training_code == "original_lipnovo"
        self.decoder_guidance_config = dict(
            self.curriculum_config.get("decoder_guidance", {}) or {}
        )
        self.decoder_guidance_enabled = bool(self.decoder_guidance_config.get("enabled", False))
        self.decoder_guidance_mode = str(
            self.decoder_guidance_config.get("mode", "vanilla_aa_ce")
        ).lower()
        if self.decoder_guidance_enabled and self.decoder_guidance_mode != "vanilla_aa_ce":
            raise ValueError(
                "GyroNovo supports only decoder_guidance.mode='vanilla_aa_ce'"
            )
        self.decoder_conditioning_enabled = self.decoder_guidance_enabled
        self.decoder_conditioning_strength = float(self.decoder_guidance_config.get("strength", 0.0))
        self.decoder_conditioning_loss_temperature = float(
            self.decoder_guidance_config.get("loss_temperature", 2.0)
        )
        self.decoder_conditioning_peak_focus_strength = float(
            self.decoder_guidance_config.get("peak_focus_strength", 0.0)
        )
        self.decoder_conditioning_peak_focus_radius = int(
            self.decoder_guidance_config.get("peak_focus_radius", 1)
        )
        self.decoder_conditioning_hard_threshold = float(
            self.decoder_guidance_config.get("hard_threshold", 0.65)
        )
        self.decoder_sample_scale_min = float(self.decoder_guidance_config.get("sample_scale_min", 1.0))
        self.decoder_sample_scale_max = float(
            self.decoder_guidance_config.get("sample_scale_max", 1.5)
        )
        default_target_weight_max = 1.0 + max(
            self.decoder_conditioning_peak_focus_strength,
            1.0,
        )
        self.decoder_conditioning_target_weight_min = float(
            self.decoder_guidance_config.get("target_weight_min", 1.0)
        )
        self.decoder_conditioning_target_weight_max = float(
            self.decoder_guidance_config.get(
                "target_weight_max",
                default_target_weight_max,
            )
        )
        self.augmentation_enabled = bool(self.augmentation_config.get("enabled", False))
        views_config = self.augmentation_config.get("views", {}) or {}
        if not isinstance(views_config, dict):
            raise TypeError("augmentation.views must be a mapping")
        easy_view_config = views_config.get("easy", {}) or {}
        hard_view_config = views_config.get("hard", {}) or {}
        if not isinstance(easy_view_config, dict) or not isinstance(hard_view_config, dict):
            raise TypeError("augmentation.views.easy and augmentation.views.hard must be mappings")

        if "decode" in views_config:
            self.decode_augmented_views = self.augmentation_enabled and bool(views_config["decode"])
            self.no_decoder_for_augmented_views = not bool(views_config["decode"])
        else:
            self.no_decoder_for_augmented_views = bool(
                self.augmentation_config.get("no_decoder_for_augmented_views", True)
            )
            self.decode_augmented_views = (
                self.augmentation_enabled and not self.no_decoder_for_augmented_views
            )

        configured_view_types = views_config.get(
            "types",
            self.augmentation_config.get("view_types"),
        )
        configured_num_views = int(self.augmentation_config.get("num_views", 2))
        self.num_aug_views = configured_num_views if self.augmentation_enabled else 0
        if configured_view_types is None:
            self.augmentation_view_types = ("easy", "hard")[: max(self.num_aug_views, 0)]
        else:
            if not isinstance(configured_view_types, (list, tuple)):
                raise TypeError("augmentation.view_types must be a list of view names")
            self.augmentation_view_types = tuple(
                str(view_type).strip().lower() for view_type in configured_view_types
            )
            invalid_view_types = set(self.augmentation_view_types) - {"easy", "hard"}
            if invalid_view_types:
                invalid_names = ", ".join(sorted(invalid_view_types))
                raise ValueError(
                    f"augmentation.view_types contains unsupported view(s): {invalid_names}"
                )
            if len(set(self.augmentation_view_types)) != len(self.augmentation_view_types):
                raise ValueError("augmentation.view_types cannot contain duplicate views")
            self.num_aug_views = len(self.augmentation_view_types) if self.augmentation_enabled else 0
        self.easy_view_scale = float(
            easy_view_config.get(
                "difficulty_scale",
                self.augmentation_config.get("easy_view_scale", 0.55),
            )
        )
        self.hard_view_scale = float(
            hard_view_config.get(
                "difficulty_scale",
                self.augmentation_config.get("hard_view_scale", 1.0),
            )
        )
        self.easy_view_weight = float(
            easy_view_config.get(
                "loss_weight",
                self.augmentation_config.get("easy_view_weight", 1.0),
            )
        )
        self.hard_view_weight = float(
            hard_view_config.get(
                "loss_weight",
                self.augmentation_config.get("hard_view_weight", 1.0),
            )
        )
        self.lambda_theory = float(self.auxiliary_config.get("lambda_theory", 1.0))
        self.lambda_imp_real = float(self.auxiliary_config.get("lambda_imp_real", 1.0))
        self.lambda_imp_aug = float(self.auxiliary_config.get("lambda_imp_aug", 0.0))
        self.lambda_decode_aug = float(self.auxiliary_config.get("lambda_decode_aug", 0.0))
        self.analysis_high_confidence_threshold = float(
            self.analysis_config.get("high_confidence_threshold", 0.98)
        )
        self.saved_path = saved_path
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
            mz_rope=self.mz_rope_config,
        )

        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
        )

        num_queries = self.gen_num
        self.query_embed = nn.Embedding(num_queries, dim_model)
        self.num_queries = num_queries
        self.theoretical_builder = TheoreticalSpectrumBuilder(residues)
        self.curriculum_scheduler = DifficultyScheduler(
            residues,
            config=self.curriculum_config,
        )
        self.augmentor = (
            PeptideConditionedSpectrumAugmentor(
                residues,
                config=self.augmentation_config,
                n_peaks=self.n_peaks,
            )
            if self.augmentation_enabled
            else None
        )
        self.imputator = Imputator(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_enc_layers=self.gen_enc_layers,
            n_dec_layers=self.gen_dec_layers
        )

        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        self.gen_mseloss = torch.nn.MSELoss()
        self.gen_bceloss = torch.nn.BCEWithLogitsLoss()

        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.peptide_mass_calculator = PeptideMass(self.residues)
        self.stop_token = self.decoder._aa2idx["$"]

        # Logging.
        self.n_log = n_log
        self._history = []



    def hungarian_matching(self, AA, logits, BB, B_mask):
        from scipy.optimize import linear_sum_assignment

        def mse_distance(A, B):
            diff = A.unsqueeze(1) - B.unsqueeze(0)
            mse = torch.mean(diff ** 2, dim=-1)
            return mse
        
        cls_score = 1 - logits.squeeze(-1).sigmoid()

        bs = AA.shape[0]
        col_inds = []
        row_inds = []
        for b in range(bs):
            A = AA[b]
            B = BB[b]
            B = B[B_mask[b] == 0]
            cls_s = cls_score[b]
            mse_cost = mse_distance(A, B)
            cls_cost = cls_s.unsqueeze(-1)
            cost_matrix = mse_cost + cls_cost
            cost_matrix = cost_matrix.cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            row_inds.append(row_ind)
            col_inds.append(col_ind)

        return row_inds, col_inds


    def forward_gen(self, memories, src_key_padding_mask):
        pred, cls_logit = self.imputator(memories, src_key_padding_mask, self.query_embed.weight)

        return pred, cls_logit

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(0.0)

    def _current_progress(self) -> float:
        trainer = getattr(self, "_trainer", None)
        if trainer is None or getattr(trainer, "max_epochs", None) in {None, 0}:
            return 0.0
        max_epochs = max(float(trainer.max_epochs), 1.0)
        return min(max(float(getattr(self, "current_epoch", 0)) / max_epochs, 0.0), 1.0)

    @staticmethod
    def _augment_complement_spectrum(spectra: torch.Tensor, precursors: torch.Tensor) -> torch.Tensor:
        augmented = torch.cat([spectra, spectra], dim=1)
        half = augmented.shape[1] // 2
        augmented[:, half:, 0] = augmented[:, half:, 0] - precursors[:, [0]]
        return augmented

    def _encode_imputation_branch(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(spectra, precursors=precursors, complement_augmented=True)
        peak_memory = encoded[0][:, 1:]
        peak_mask = encoded[1][:, 1:]
        generated_embeddings, cls_logits = self.forward_gen(peak_memory, peak_mask)
        return {
            "encoded": encoded,
            "generated_embeddings": generated_embeddings,
            "cls_logits": cls_logits,
        }

    def _compose_decoder_memory(
        self,
        encoded: tuple[torch.Tensor, torch.Tensor],
        generated_embeddings: torch.Tensor,
        cls_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gen_mask = cls_logits.squeeze(-1).sigmoid() < self.gen_threshold
        memories = torch.cat(
            [
                encoded[0][:, :1, :],
                generated_embeddings,
                encoded[0][:, 1:, :],
            ],
            dim=1,
        )
        masks = torch.cat(
            [
                encoded[1][:, :1],
                gen_mask,
                encoded[1][:, 1:],
            ],
            dim=1,
        )
        return memories, masks

    def _compute_decoder_ce_for_branch(
        self,
        sequences: Sequence[str],
        precursors: torch.Tensor,
        branch_outputs: dict[str, Any],
        *,
        loss_fn: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        complement_encoded = self._compose_decoder_memory(
            branch_outputs["encoded"],
            branch_outputs["generated_embeddings"],
            branch_outputs["cls_logits"],
        )
        pred, truth = self.decoder(sequences, precursors, *complement_encoded)
        decoder_logits = pred[:, :-1, :]
        flat_pred = decoder_logits.reshape(-1, self.decoder.vocab_size + 1)
        loss = (loss_fn or self.celoss)(flat_pred, truth.flatten())
        return loss, decoder_logits, truth

    def _build_theoretical_batch(
        self,
        spectra: torch.Tensor,
        sequences: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        max_intensity = torch.max(spectra[:, :, 1], dim=-1).values.detach().cpu().tolist()
        target_batch = self.theoretical_builder.build_batch(
            sequences,
            intensity_mode="flat",
            intensity_scales=max_intensity,
            device=spectra.device,
            dtype=spectra.dtype,
        )
        return target_batch

    def _truncate_target_set(
        self,
        target_embeddings: torch.Tensor,
        target_mask: torch.Tensor,
        target_annotations: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if target_embeddings.shape[1] <= self.num_queries:
            return target_embeddings, target_mask, target_annotations
        target_embeddings = target_embeddings[:, : self.num_queries, :]
        target_mask = target_mask[:, : self.num_queries]
        truncated = {
            key: value[:, : self.num_queries]
            if value.dim() > 1 and value.shape[1] >= self.num_queries
            else value
            for key, value in target_annotations.items()
        }
        return target_embeddings, target_mask, truncated

    def _compute_imputation_losses(
        self,
        generated_embeddings: torch.Tensor,
        cls_logits: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_mask: torch.Tensor,
        target_weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        row_inds, col_inds = self.hungarian_matching(
            generated_embeddings,
            cls_logits,
            target_embeddings,
            target_mask,
        )
        loss_mse = self._zero_loss(generated_embeddings)
        batch_size = max(target_embeddings.shape[0], 1)
        if target_weights is None:
            target_weights = torch.ones(
                target_mask.shape,
                device=target_embeddings.device,
                dtype=target_embeddings.dtype,
            )
        for batch_idx in range(len(row_inds)):
            if len(row_inds[batch_idx]) == 0:
                continue
            matched_generated = generated_embeddings[batch_idx][row_inds[batch_idx]]
            current_targets = target_embeddings[batch_idx][target_mask[batch_idx] == 0]
            matched_targets = current_targets[col_inds[batch_idx]]
            current_weights = target_weights[batch_idx][target_mask[batch_idx] == 0]
            matched_weights = current_weights[col_inds[batch_idx]].to(matched_generated.dtype)
            per_match_mse = torch.mean((matched_generated - matched_targets) ** 2, dim=-1)
            loss_mse = loss_mse + (
                per_match_mse * matched_weights
            ).sum() / matched_weights.sum().clamp_min(1e-6)
        loss_mse = loss_mse / batch_size

        cls_label = torch.zeros_like(cls_logits.squeeze(-1), device=cls_logits.device)
        cls_weight = torch.ones_like(cls_label, device=cls_logits.device)
        for batch_idx, row_ind in enumerate(row_inds):
            if len(row_ind) > 0:
                cls_label[batch_idx][row_ind] = 1
                current_weights = target_weights[batch_idx][target_mask[batch_idx] == 0]
                cls_weight[batch_idx][row_ind] = current_weights[col_inds[batch_idx]].to(cls_weight.dtype)
        loss_bce = F.binary_cross_entropy_with_logits(
            cls_logits.squeeze(-1),
            cls_label,
            weight=cls_weight,
        )
        return {
            "loss_mse": loss_mse,
            "loss_bce": loss_bce,
            "loss_total": loss_mse + loss_bce,
            "row_inds": row_inds,
            "col_inds": col_inds,
            "generated_embeddings": generated_embeddings,
            "cls_logits": cls_logits,
        }

    def _compute_decoder_conditioning(
        self,
        decoder_logits: torch.Tensor,
        truth: torch.Tensor,
        target_annotations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        peak_mask = target_annotations["peak_mask"]
        target_weights = torch.ones(
            peak_mask.shape,
            device=decoder_logits.device,
            dtype=decoder_logits.dtype,
        )
        batch_size = decoder_logits.shape[0]
        sample_scales = torch.ones(batch_size, device=decoder_logits.device, dtype=decoder_logits.dtype)
        zero = decoder_logits.new_tensor(0.0)
        if (
            not self.decoder_conditioning_enabled
            or decoder_logits.numel() == 0
        ):
            return {
                "target_weights": target_weights,
                "sample_scales": sample_scales,
                "sample_ce_mean": zero,
                "sample_entropy_mean": zero,
                "sample_scale_mean": sample_scales.mean(),
                "target_weight_mean": target_weights.mean(),
                "hard_fraction": zero,
            }

        with torch.no_grad():
            detached_logits = torch.nan_to_num(
                decoder_logits.detach(),
                nan=0.0,
                posinf=30.0,
                neginf=-30.0,
            )
            token_loss = F.cross_entropy(
                detached_logits.reshape(-1, detached_logits.shape[-1]),
                truth.reshape(-1),
                reduction="none",
                ignore_index=0,
            ).reshape_as(truth)
            valid_tokens = truth != 0
            valid_counts = valid_tokens.sum(dim=1).clamp_min(1)
            last_valid_indices = (valid_counts - 1).clamp_min(0)
            last_valid_tokens = truth.gather(1, last_valid_indices.unsqueeze(1)).squeeze(1)
            has_stop_token = last_valid_tokens == self.stop_token
            peptide_lengths = (valid_counts - has_stop_token.to(valid_counts.dtype)).clamp_min(0)

            log_probs = F.log_softmax(detached_logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1) / max(
                float(np.log(detached_logits.shape[-1])),
                1e-6,
            )

            return self._compute_vanilla_aa_ce_conditioning(
                token_loss=token_loss,
                entropy=entropy,
                valid_tokens=valid_tokens,
                truth=truth,
                peak_mask=peak_mask,
                peptide_lengths=peptide_lengths,
                target_annotations=target_annotations,
                decoder_logits=decoder_logits,
            )

    def _compute_vanilla_aa_ce_conditioning(
        self,
        *,
        token_loss: torch.Tensor,
        entropy: torch.Tensor,
        valid_tokens: torch.Tensor,
        truth: torch.Tensor,
        peak_mask: torch.Tensor,
        peptide_lengths: torch.Tensor,
        target_annotations: dict[str, torch.Tensor],
        decoder_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Use amino-acid CE directly for imputation weights and synthetic difficulty."""
        aa_tokens = valid_tokens & (truth != self.stop_token)
        aa_counts = aa_tokens.sum(dim=1).clamp_min(1)
        sample_ce = (token_loss * aa_tokens).sum(dim=1) / aa_counts
        sample_ce = torch.nan_to_num(sample_ce, nan=0.0, posinf=0.0, neginf=0.0)

        sample_entropy = (entropy * aa_tokens).sum(dim=1) / aa_counts
        sample_entropy = torch.nan_to_num(sample_entropy, nan=0.0, posinf=0.0, neginf=0.0)

        temperature = (
            self.decoder_conditioning_loss_temperature
            if math.isfinite(self.decoder_conditioning_loss_temperature)
            else 1.0
        )
        temperature = max(temperature, 1e-6)
        finite_token_loss = torch.nan_to_num(
            token_loss,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        token_error = 1.0 - torch.exp(-finite_token_loss / temperature)
        token_error = torch.where(aa_tokens, token_error, torch.zeros_like(token_error))
        token_error = torch.nan_to_num(token_error, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0,
            1.0,
        )

        sample_error = 1.0 - torch.exp(-sample_ce / temperature)
        sample_error = torch.nan_to_num(sample_error, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0,
            1.0,
        )

        strength = (
            self.decoder_conditioning_strength
            if math.isfinite(self.decoder_conditioning_strength)
            else 0.0
        )
        strength = max(strength, 0.0)
        sample_scale_min = (
            self.decoder_sample_scale_min if math.isfinite(self.decoder_sample_scale_min) else 1.0
        )
        sample_scale_max = (
            self.decoder_sample_scale_max if math.isfinite(self.decoder_sample_scale_max) else 1.0
        )
        lower_scale = min(sample_scale_min, sample_scale_max)
        upper_scale = max(sample_scale_min, sample_scale_max)
        sample_scales = 1.0 + strength * sample_error
        sample_scales = sample_scales.clamp(min=lower_scale, max=upper_scale)
        sample_scales = torch.nan_to_num(
            sample_scales,
            nan=1.0,
            posinf=upper_scale,
            neginf=lower_scale,
        )

        target_weights = torch.ones(
            peak_mask.shape,
            device=decoder_logits.device,
            dtype=decoder_logits.dtype,
        )
        focus_strength = (
            self.decoder_conditioning_peak_focus_strength
            if math.isfinite(self.decoder_conditioning_peak_focus_strength)
            else 0.0
        )
        focus_strength = max(focus_strength, 0.0)
        target_weight_min = (
            self.decoder_conditioning_target_weight_min
            if math.isfinite(self.decoder_conditioning_target_weight_min)
            else 1.0
        )
        target_weight_max = (
            self.decoder_conditioning_target_weight_max
            if math.isfinite(self.decoder_conditioning_target_weight_max)
            else 1.0 + max(focus_strength, 1.0)
        )
        target_weight_min, target_weight_max = sorted((target_weight_min, target_weight_max))
        radius = max(self.decoder_conditioning_peak_focus_radius, 0)
        cleavage_index = target_annotations["cleavage_index"].to(decoder_logits.device)
        ion_family = target_annotations.get("ion_family")
        ion_family = ion_family.to(decoder_logits.device) if ion_family is not None else None
        for batch_idx in range(decoder_logits.shape[0]):
            active_peaks = ~peak_mask[batch_idx].to(decoder_logits.device)
            peak_indices = torch.where(active_peaks)[0]
            peptide_length = int(peptide_lengths[batch_idx].item())
            for target_idx in peak_indices.tolist():
                cleavage = int(cleavage_index[batch_idx, target_idx].item())
                family = (
                    int(ion_family[batch_idx, target_idx].item())
                    if ion_family is not None
                    else -1
                )
                focus_positions = self._fragment_decoder_focus_positions(
                    cleavage_index=cleavage,
                    ion_family=family,
                    peptide_length=peptide_length,
                )
                if not focus_positions:
                    continue
                local_terms = []
                for focus_position in focus_positions:
                    start = max(focus_position - radius, 0)
                    end = min(focus_position + radius + 1, peptide_length, token_error.shape[1])
                    if start < end:
                        local_terms.append(token_error[batch_idx, start:end].max())
                if not local_terms:
                    continue
                local_aa_ce_error = torch.stack(local_terms).max()
                target_weights[batch_idx, target_idx] = (
                    1.0 + focus_strength * local_aa_ce_error
                ).clamp(min=target_weight_min, max=target_weight_max)
        target_weights = torch.nan_to_num(
            target_weights,
            nan=1.0,
            posinf=target_weight_max,
            neginf=target_weight_min,
        )

        hard_fraction = (sample_error >= self.decoder_conditioning_hard_threshold).to(
            decoder_logits.dtype
        ).mean()
        active_targets = ~peak_mask.to(decoder_logits.device)
        return {
            "target_weights": target_weights.detach(),
            "sample_scales": sample_scales.detach(),
            "sample_ce_mean": sample_ce.mean().detach(),
            "sample_entropy_mean": sample_entropy.mean().detach(),
            "sample_scale_mean": sample_scales.mean().detach(),
            "target_weight_mean": target_weights[active_targets].mean().detach()
            if bool(active_targets.any().detach().cpu().item())
            else target_weights.mean().detach(),
            "hard_fraction": hard_fraction.detach(),
        }

    def _fragment_decoder_focus_positions(
        self,
        *,
        cleavage_index: int,
        ion_family: int,
        peptide_length: int,
    ) -> tuple[int, ...]:
        """Map a theoretical b/y fragment cleavage to decoder residue positions."""
        if peptide_length <= 0 or cleavage_index < 0 or cleavage_index >= peptide_length - 1:
            return ()

        if ion_family == 0:
            residue_positions = (cleavage_index, cleavage_index + 1)
        elif ion_family == 1:
            residue_positions = (
                peptide_length - cleavage_index - 2,
                peptide_length - cleavage_index - 1,
            )
        else:
            residue_positions = (cleavage_index, cleavage_index + 1)

        decoder_reversed = bool(getattr(self.decoder, "reverse", False))
        focus_positions = []
        for residue_position in residue_positions:
            if residue_position < 0 or residue_position >= peptide_length:
                continue
            if decoder_reversed:
                residue_position = peptide_length - 1 - residue_position
            focus_positions.append(int(residue_position))
        return tuple(sorted(set(focus_positions)))

    def _condition_difficulties(
        self,
        difficulties: Sequence[DifficultyVector],
        sample_scales: torch.Tensor,
    ) -> list[DifficultyVector]:
        if (
            not self.decoder_conditioning_enabled
            or len(difficulties) == 0
        ):
            return list(difficulties)
        sample_scale_min = (
            self.decoder_sample_scale_min if math.isfinite(self.decoder_sample_scale_min) else 1.0
        )
        sample_scale_max = (
            self.decoder_sample_scale_max if math.isfinite(self.decoder_sample_scale_max) else 1.0
        )
        scales = torch.nan_to_num(
            sample_scales.detach(),
            nan=1.0,
            posinf=max(sample_scale_min, sample_scale_max),
            neginf=min(sample_scale_min, sample_scale_max),
        ).cpu().tolist()
        conditioned = []
        for difficulty, scale in zip(difficulties, scales):
            scale = float(scale) if np.isfinite(float(scale)) else 1.0
            conditioned.append(
                DifficultyVector(
                    peak_missingness=difficulty.peak_missingness * scale,
                    b_ion_dropout=difficulty.b_ion_dropout * scale,
                    y_ion_dropout=difficulty.y_ion_dropout * scale,
                    fragmentation_ladder_gap=difficulty.fragmentation_ladder_gap * scale,
                    intensity_jitter=difficulty.intensity_jitter * scale,
                    mz_jitter=difficulty.mz_jitter * scale,
                    structured_noise_count=difficulty.structured_noise_count * scale,
                    ptm_visibility_reduction=difficulty.ptm_visibility_reduction * scale,
                ).clamp(self.curriculum_scheduler.maximum)
            )
        return conditioned

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        return self.beam_search_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )

    def beam_search_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        spectra = self._augment_complement_spectrum(spectra, precursors)
        branch_outputs = self._encode_imputation_branch(spectra, precursors)
        memories, mem_masks = self._compose_decoder_memory(
            branch_outputs["encoded"],
            branch_outputs["generated_embeddings"],
            branch_outputs["cls_logits"],
        )

        # Sizes.
        batch = spectra.shape[0]  # B
        length = self.max_length + 1  # L
        vocab = self.decoder.vocab_size + 1  # V
        beam = self.n_beams  # S
        # Initialize scores and tokens.
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        )
        scores = scores.type_as(spectra)
        tokens = torch.zeros(batch, length, beam, dtype=torch.int64)
        tokens = tokens.to(self.encoder.device)

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        pred, _ = self.decoder(None, precursors, memories, mem_masks)
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        # The main decoding loop.
        for step in range(0, self.max_length):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
            ) = self._finish_beams(tokens, precursors, step)
            # Cache peptide predictions from the finished beams (but not the
            # discarded beams).
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )

            # Stop decoding when all current beams have been finished.
            # Continue with beams that have not been finished and not discarded.
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            # Update the scores.
            scores[~finished_beams, : step + 2, :], _ = self.decoder(
                tokens[~finished_beams, : step + 1],
                precursors[~finished_beams, :],
                memories[~finished_beams, :, :],
                mem_masks[~finished_beams, :],
            )
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return list(self._get_top_peptide(pred_cache))

    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """
        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass = [None]
        for aa, mass in self.peptide_mass_calculator.masses.items():
            if mass < 0:
                aa_neg_mass.append(aa)
        # Find N-terminal residues.
        n_term = torch.Tensor(
            [
                self.decoder._aa2idx[aa]
                for aa in self.peptide_mass_calculator.masses
                if aa.startswith(("+", "-"))
            ]
        ).to(self.decoder.device)

        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.encoder.device)
        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        ends_stop_token = tokens[:, step] == self.stop_token
        finished_beams[ends_stop_token] = True
        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        discarded_beams[tokens[:, step] == 0] = True
        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            # Multiple N-terminal modifications.
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)
            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])
            internal_mods = torch.isin(
                torch.where(mask.to(self.encoder.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)
            peptide = self.decoder.detokenize(pred_tokens)
            if len(peptide) == 0:
                discarded_beams[i] = True
                continue
            # Omit stop token.
            if self.decoder.reverse and peptide[0] == "$":
                peptide = peptide[1:]
                peptide_len -= 1
            elif not self.decoder.reverse and peptide[-1] == "$":
                peptide = peptide[:-1]
                peptide_len -= 1
            if len(peptide) == 0:
                if finished_beams[i] or peptide_len < self.min_peptide_len:
                    discarded_beams[i] = True
                    continue
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 2]
            matches_precursor_mz = exceeds_precursor_mz = False
            for aa in [None] if finished_beams[i] else aa_neg_mass:
                if aa is None:
                    calc_peptide = peptide
                else:
                    calc_peptide = peptide.copy()
                    calc_peptide.append(aa)
                try:
                    calc_mz = self.peptide_mass_calculator.mass(
                        seq=calc_peptide, charge=precursor_charge
                    )
                    delta_mass_ppm = [
                        _calc_mass_error(
                            calc_mz,
                            precursor_mz,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            self.isotope_error_range[0],
                            self.isotope_error_range[1] + 1,
                        )
                    ]
                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.
            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, amino acid-level scores, and the predicted tokens is
            stored.
        """
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            pred_tokens = tokens[i][: step + 1]
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            # Don't cache this peptide if it was already predicted previously.
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            smx = self.softmax(scores[i : i + 1, : step + 1, :])
            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].tolist()
            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            if not has_stop_token:
                aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)
            # Calculate the updated amino acid-level and the peptide scores.
            aa_scores, peptide_score = _aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )
            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    np.random.random_sample(),
                    aa_scores,
                    torch.clone(pred_peptide),
                ),
            )

    def _get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.decoder.vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores[:, step, :, :], "B V S -> B (V S)"
        )

        # Find all still active beams by masking out terminated beams.
        active_mask = (
            ~finished_beams.reshape(batch, beam).repeat(1, vocab)
        ).float()
        # Mask out the index '0', i.e. padding token, by default.
        # FIXME: Set this to a very small, yet non-zero value, to only
        # get padding after stop token.
        active_mask[:, :beam] = 1e-8

        # Figure out the top K decodings.
        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)
        v_idx_np, s_idx_np = np.unravel_index(top_idx.detach().cpu().numpy(), (vocab, beam))
        v_idx = torch.as_tensor(v_idx_np, device=tokens.device, dtype=tokens.dtype)
        s_idx = torch.as_tensor(s_idx_np, device=tokens.device, dtype=torch.long)
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch, device=tokens.device), "B -> (B S)", S=beam)

        # Record the top K decodings.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, step, :] = v_idx
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return tokens, scores

    def _get_top_peptide(
        self,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each spectrum.

        Parameters
        ----------
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the peptide
            score, amino acid-level scores, and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores,
                        "".join(self.decoder.detokenize(pred_tokens)),
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

    def _decode_top_predictions(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
    ) -> tuple[list[str], list[float]]:
        predictions = self.forward(spectra, precursors)
        peptides_pred, peptides_score = [], []
        for spectrum_preds in predictions:
            if spectrum_preds == []:
                peptides_pred.append("")
                peptides_score.append(float("-inf"))
                continue
            top_prediction = spectrum_preds[0]
            peptides_pred.append(top_prediction[2])
            peptides_score.append(float(top_prediction[0]))
        return peptides_pred, peptides_score

    def _gather_validation_records(self) -> list[dict[str, float | str]]:
        records = list(self._validation_records)
        try:
            trainer = getattr(self, "trainer", None)
        except RuntimeError:
            trainer = None
        trainer_world_size = int(getattr(trainer, "world_size", 1) or 1)
        if (
            trainer_world_size > 1
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            gathered: list[list[dict[str, float | str]] | None] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, records)
            records = [record for shard in gathered if shard is not None for record in shard]
        return records

    def _split_peptide(self, peptide: str) -> list[str]:
        text = str(peptide).strip()
        if not text or text.lower() in {"nan", "none"}:
            return []
        try:
            return self.theoretical_builder.tokenizer.split(text)
        except Exception:
            if not isinstance(self.residues, dict):
                return list(text)
            tokens = sorted(self.residues.keys(), key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in tokens)
            parts = re.findall(pattern, text)
            if "".join(parts) != text:
                return []
            return parts

    def _novor_match(self, target: Sequence[str], prediction: Sequence[str]) -> int:
        n_matches = 0
        if len(target) == 0 or len(prediction) == 0:
            return n_matches

        target_masses = np.array([self.residues[token] for token in target], dtype=np.float64)
        prediction_masses = np.array([self.residues[token] for token in prediction], dtype=np.float64)
        target_prefix = np.cumsum(target_masses)
        prediction_prefix = np.cumsum(prediction_masses)

        i, j = 0, 0
        while i < len(target) and j < len(prediction):
            if abs(target_prefix[i] - prediction_prefix[j]) < 0.5:
                n_matches += int(abs(target_masses[i] - prediction_masses[j]) < 0.1)
                i += 1
                j += 1
            elif target_prefix[i] > prediction_prefix[j]:
                j += 1
            else:
                i += 1
        return n_matches

    @staticmethod
    def _levenshtein_distance(target: Sequence[str], prediction: Sequence[str]) -> int:
        if len(target) == 0:
            return len(prediction)
        if len(prediction) == 0:
            return len(target)

        previous = list(range(len(prediction) + 1))
        for i, target_token in enumerate(target, start=1):
            current = [i]
            for j, prediction_token in enumerate(prediction, start=1):
                substitution = previous[j - 1] + int(target_token != prediction_token)
                insertion = current[j - 1] + 1
                deletion = previous[j] + 1
                current.append(min(substitution, insertion, deletion))
            previous = current
        return previous[-1]

    def _compute_prediction_metrics(self, records: Sequence[dict[str, float | str]]) -> dict[str, float]:
        if len(records) == 0:
            return {}
        n_target_aa, n_predicted_aa, n_matched_aa = 0, 0, 0
        n_peptide_matches, total_edit_distance = 0, 0
        for record in records:
            target = self._split_peptide(str(record["peptides_true"]))
            prediction = self._split_peptide(str(record["peptides_pred"]))

            n_target_aa += len(target)
            n_predicted_aa += len(prediction)
            n_matches = self._novor_match(target, prediction)
            n_matched_aa += n_matches
            n_peptide_matches += int(len(target) == len(prediction) and len(target) == n_matches)
            total_edit_distance += self._levenshtein_distance(target, prediction)

        aa_precision = n_matched_aa / max(n_predicted_aa, 1)
        aa_recall = n_matched_aa / max(n_target_aa, 1)
        pep_recall = n_peptide_matches / max(len(records), 1)
        aa_er = total_edit_distance / max(n_target_aa, 1)
        return {
            "aa_precision": aa_precision,
            "aa_recall": aa_recall,
            "pep_recall": pep_recall,
            "pep_precision": pep_recall,
            "aa_error_rate": aa_er,
        }

    def _is_peptide_match(self, target_sequence: str, prediction_sequence: str) -> bool:
        target = self._split_peptide(str(target_sequence))
        prediction = self._split_peptide(str(prediction_sequence))
        return bool(
            len(target) == len(prediction)
            and len(target) == self._novor_match(target, prediction)
        )

    @staticmethod
    def _batch_has_sequence_labels(sequences: Sequence[str]) -> bool:
        labeled = []
        for sequence in sequences:
            if sequence is None:
                labeled.append(False)
                continue
            if isinstance(sequence, float) and np.isnan(sequence):
                labeled.append(False)
                continue
            text = str(sequence).strip()
            labeled.append(bool(text) and text.lower() not in {"nan", "none"})
        if any(labeled) and not all(labeled):
            raise ValueError("LipNovo validation batches cannot mix labeled and unlabeled spectra")
        return all(labeled)

    def _validation_decode_indices(
        self,
        batch_size: int,
        batch_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.validation_decode_fraction >= 1.0:
            return torch.arange(batch_size, device=device)
        if self.validation_decode_fraction <= 0.0 or batch_size <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        stride = max(int(round(1.0 / self.validation_decode_fraction)), 1)
        # Keep the decoded validation subset fixed across epochs/runs so
        # checkpoint metrics compare the same spectra throughout training.
        offset = int(batch_idx) % stride
        indices = torch.arange(batch_size, device=device)
        selected = indices[(indices % stride) == offset]
        if selected.numel() == 0:
            selected = indices[:1]
        return selected

    @staticmethod
    def _normalize_original_lipnovo_sequence(sequence: str) -> str:
        return (
            str(normalize_lipnovo_sequence(str(sequence)))
            .replace("(+57.02)", "")
            .replace("M(+15.99)", "1")
            .replace("N(+.98)", "2")
            .replace("Q(+.98)", "3")
            .replace("M(ox)", "4")
        )

    @staticmethod
    def _calculate_all_ion_mz_original(sequence: str) -> torch.Tensor:
        aa_mass = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C": 160.030649,
            "L": 113.084064,
            "I": 113.084064,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
            "1": 147.035400,
            "2": 115.026943,
            "3": 129.042594,
            "4": 147.035400,
        }
        water_mass = 18.01056
        oh_mass = 17.0027
        h_mass = 1.0078

        b_ions = []
        y_ions = []
        for j in range(len(sequence) - 1):
            b_mass = 0.0
            for i in range(j + 1):
                b_mass += aa_mass[sequence[i]]
                b_mass += water_mass
            b_mass -= water_mass * j
            b_mass -= oh_mass
            b_ions.append(b_mass)

        rev_sequence = sequence[::-1]
        for j in range(len(sequence) - 1):
            y_mass = 0.0
            for i in range(j + 1):
                y_mass += aa_mass[rev_sequence[i]]
                y_mass += water_mass
            y_mass -= water_mass * j
            y_mass -= h_mass
            y_mass += h_mass * 2
            y_ions.append(y_mass)

        ions = b_ions + y_ions
        ions.sort()
        return torch.tensor(ions, dtype=torch.float32)

    def _original_lipnovo_training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """Original LIPNovo training objective, kept for the base baseline."""
        raw_spectra, precursors, sequences = batch
        sequences = list(sequences)
        spectra = torch.cat([raw_spectra, raw_spectra], dim=1)
        half = spectra.shape[1] // 2
        spectra[:, half:, 0] = spectra[:, half:, 0] - precursors[:, [0]]

        theoretical_mz = []
        max_len = 0
        for sequence in sequences:
            ions = self._calculate_all_ion_mz_original(
                self._normalize_original_lipnovo_sequence(str(sequence))
            ).to(device=spectra.device, dtype=spectra.dtype)
            max_len = max(max_len, ions.shape[0])
            theoretical_mz.append(ions)

        max_len = max(max_len, 1)
        padded_mz = []
        for ions in theoretical_mz:
            padded = torch.zeros(max_len, device=spectra.device, dtype=spectra.dtype)
            padded[: ions.shape[0]] = ions
            padded_mz.append(padded.unsqueeze(0))
        the_mz = torch.cat(padded_mz, dim=0)

        exp_intensity = spectra[:, :, 1]
        the_intensity = torch.max(exp_intensity, dim=-1).values.unsqueeze(-1)
        the_intensity = the_intensity.repeat(1, the_mz.shape[-1])
        the_mask = the_mz > 0
        the_intensity = the_intensity * the_mask
        the_ms = torch.cat([the_mz.unsqueeze(-1), the_intensity.unsqueeze(-1)], dim=-1)

        exp_encoded = self.encoder(spectra)
        exp_memories = exp_encoded[0][:, 1:]
        exp_src_key_padding_mask = exp_encoded[1][:, 1:]

        generated_embeddings, cls_logits = self.forward_gen(exp_memories, exp_src_key_padding_mask)
        gen_mask = cls_logits.squeeze(-1).sigmoid() < self.gen_threshold

        memories = torch.cat(
            [
                exp_encoded[0][:, :1, :],
                generated_embeddings,
                exp_encoded[0][:, 1:, :],
            ],
            dim=1,
        )
        src_key_padding_mask = torch.cat(
            [
                exp_encoded[1][:, :1],
                gen_mask,
                exp_encoded[1][:, 1:],
            ],
            dim=1,
        )
        pred, truth = self.decoder(sequences, precursors, memories, src_key_padding_mask)
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)

        if mode == "train":
            loss_ce = self.celoss(pred, truth.flatten())

            the_encoded = self.encoder(the_ms)
            the_pred, the_truth = self.decoder(sequences, precursors, *the_encoded)
            the_pred = the_pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            loss_the = self.celoss(the_pred, the_truth.flatten())

            tgt_embeddings = the_encoded[0][:, 1:].detach()
            tgt_src_mask = the_encoded[1][:, 1:].detach()
            if tgt_embeddings.shape[1] > self.num_queries:
                tgt_embeddings = tgt_embeddings[:, : self.num_queries, :]
                tgt_src_mask = tgt_src_mask[:, : self.num_queries]

            row_inds, col_inds = self.hungarian_matching(
                generated_embeddings,
                cls_logits,
                tgt_embeddings,
                tgt_src_mask,
            )

            loss_mse = pred.new_tensor(0.0)
            batch_size = max(tgt_embeddings.shape[0], 1)
            for batch_idx in range(len(row_inds)):
                row_ind = row_inds[batch_idx]
                col_ind = col_inds[batch_idx]
                matched_gen = generated_embeddings[batch_idx][row_ind]
                cur_tgt = tgt_embeddings[batch_idx][tgt_src_mask[batch_idx] == 0]
                matched_tgt = cur_tgt[col_ind]
                loss_mse = loss_mse + self.gen_mseloss(matched_gen, matched_tgt)
            loss_mse = loss_mse / batch_size

            cls_label = torch.zeros_like(cls_logits.squeeze(-1), device=cls_logits.device)
            for batch_idx in range(len(row_inds)):
                cls_label[batch_idx][row_inds[batch_idx]] = 1
            loss_bce = self.gen_bceloss(cls_logits.squeeze(-1), cls_label)
            loss = loss_ce + loss_the + loss_bce + loss_mse
        else:
            loss = self.val_celoss(pred, truth.flatten())

        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int = 0,
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        if self.use_original_lipnovo_training:
            return self._original_lipnovo_training_step(batch, *args, mode=mode)

        raw_spectra, precursors, sequences = batch
        sequences = list(sequences)
        spectra = self._augment_complement_spectrum(raw_spectra, precursors)

        real_branch = self._encode_imputation_branch(spectra, precursors)
        complement_encoded = self._compose_decoder_memory(
            real_branch["encoded"],
            real_branch["generated_embeddings"],
            real_branch["cls_logits"],
        )
        pred, truth = self.decoder(sequences, precursors, *complement_encoded)
        decoder_logits = pred[:, :-1, :]
        pred = decoder_logits.reshape(-1, self.decoder.vocab_size + 1)

        if mode != "train":
            loss = self.val_celoss(pred, truth.flatten())
            self.log(
                f"{mode}_CELoss",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return loss

        loss_ce = self.celoss(pred, truth.flatten())

        target_batch = self._build_theoretical_batch(raw_spectra, sequences)
        the_encoded = self.encoder(target_batch["spectra"], precursors=precursors)
        target_encoded = the_encoded
        target_embeddings = target_encoded[0][:, 1:].detach()
        target_mask = target_encoded[1][:, 1:].detach()
        target_annotations = {
            "cleavage_index": target_batch["cleavage_index"],
            "ion_family": target_batch["ion_family"],
            "is_ptm_related": target_batch["is_ptm_related"],
            "peak_mask": target_batch["peak_mask"],
        }
        target_embeddings, target_mask, target_annotations = self._truncate_target_set(
            target_embeddings,
            target_mask,
            target_annotations,
        )
        decoder_conditioning = self._compute_decoder_conditioning(
            decoder_logits,
            truth,
            target_annotations,
        )

        loss_the = self._zero_loss(pred)
        if self.lambda_theory > 0.0:
            the_pred, the_truth = self.decoder(sequences, precursors, *the_encoded)
            the_pred = the_pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            loss_the = self.celoss(the_pred, the_truth.flatten())

        real_imp = self._compute_imputation_losses(
            real_branch["generated_embeddings"],
            real_branch["cls_logits"],
            target_embeddings,
            target_mask,
            target_weights=decoder_conditioning["target_weights"],
        )

        loss_imp_aug = self._zero_loss(pred)
        loss_decode_aug = self._zero_loss(pred)
        aug_decode_losses: dict[str, torch.Tensor] = {
            "easy": self._zero_loss(pred),
            "hard": self._zero_loss(pred),
        }
        augmentation_activation_sums: dict[str, float] = collections.defaultdict(float)
        augmentation_activation_count = 0
        if self.augmentation_enabled and self.augmentor is not None and self.num_aug_views > 0:
            progress = self._current_progress()
            weighted_losses = []
            weighted_decode_losses = []
            total_view_weight = 0.0
            view_parameters = {
                "easy": (self.easy_view_scale, self.easy_view_weight),
                "hard": (self.hard_view_scale, self.hard_view_weight),
            }
            view_specs = [
                (view_name, *view_parameters[view_name])
                for view_name in self.augmentation_view_types
            ]
            for view_name, view_scale, view_weight in view_specs:
                difficulties = self.curriculum_scheduler.batch_difficulties(
                    sequences,
                    progress=progress,
                    view_scale=view_scale,
                )
                difficulties = self._condition_difficulties(
                    difficulties,
                    decoder_conditioning["sample_scales"],
                )
                seed = (
                    self.random_seed
                    + int(getattr(self, "current_epoch", 0)) * 100_000
                    + int(batch_idx) * 1_000
                    + int(getattr(self, "global_rank", 0)) * 10
                    + int(view_name == "hard")
                )
                augmented_base = self.augmentor.render_batch(
                    sequences=sequences,
                    precursors=precursors,
                    reference_spectra=raw_spectra,
                    difficulties=difficulties,
                    seed=seed,
                    device=raw_spectra.device,
                    dtype=raw_spectra.dtype,
                    target_focus_weights=decoder_conditioning["target_weights"],
                    view_policy=f"{view_name}_decoder_focused",
                )
                for key, value in self.augmentor.last_activation_stats.items():
                    augmentation_activation_sums[key] += float(value)
                augmentation_activation_count += 1
                augmented_spectra = self._augment_complement_spectrum(augmented_base, precursors)
                aug_branch = self._encode_imputation_branch(augmented_spectra, precursors)
                aug_losses = self._compute_imputation_losses(
                    aug_branch["generated_embeddings"],
                    aug_branch["cls_logits"],
                    target_embeddings,
                    target_mask,
                    target_weights=decoder_conditioning["target_weights"],
                )
                weighted_losses.append(float(view_weight) * aug_losses["loss_total"])
                if self.decode_augmented_views:
                    aug_loss_decode, _, _ = self._compute_decoder_ce_for_branch(
                        sequences,
                        precursors,
                        aug_branch,
                    )
                    aug_decode_losses[view_name] = aug_loss_decode
                    weighted_decode_losses.append(float(view_weight) * aug_loss_decode)
                total_view_weight += float(view_weight)
            if total_view_weight > 0.0:
                loss_imp_aug = sum(weighted_losses) / total_view_weight
                if weighted_decode_losses:
                    loss_decode_aug = sum(weighted_decode_losses) / total_view_weight

        loss = (
            loss_ce
            + self.lambda_theory * loss_the
            + self.lambda_imp_real * real_imp["loss_total"]
            + self.lambda_imp_aug * loss_imp_aug
            + self.lambda_decode_aug * loss_decode_aug
        )

        logged_losses = {
            "loss_total": loss.detach(),
            "loss_real_decode": loss_ce.detach(),
            "loss_theory_decode": loss_the.detach(),
            "loss_imp_real": real_imp["loss_total"].detach(),
            "loss_imp_real_mse": real_imp["loss_mse"].detach(),
            "loss_imp_real_bce": real_imp["loss_bce"].detach(),
            "loss_imp_aug": loss_imp_aug.detach(),
            "loss_decode_aug": loss_decode_aug.detach(),
            "loss_decode_aug_easy": aug_decode_losses["easy"].detach(),
            "loss_decode_aug_hard": aug_decode_losses["hard"].detach(),
        }
        if self.decoder_conditioning_enabled:
            logged_losses.update(
                {
                    "decoder_conditioning_ce": decoder_conditioning["sample_ce_mean"],
                    "decoder_conditioning_entropy": decoder_conditioning["sample_entropy_mean"],
                    "decoder_conditioning_sample_scale": decoder_conditioning["sample_scale_mean"],
                    "decoder_conditioning_target_weight": decoder_conditioning["target_weight_mean"],
                    "decoder_conditioning_hard_fraction": decoder_conditioning["hard_fraction"],
                    "mean_sample_scale": decoder_conditioning["sample_scale_mean"],
                    "mean_target_weight": decoder_conditioning["target_weight_mean"],
                    "high_error_fraction": decoder_conditioning["hard_fraction"],
                }
            )
        if augmentation_activation_count > 0:
            for key, value in augmentation_activation_sums.items():
                logged_losses[f"aug_activation_{key}"] = pred.new_tensor(
                    value / float(augmentation_activation_count)
                )
        for key, value in logged_losses.items():
            self.log(
                f"{mode}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if self.augmentation_enabled:
            progress = self._current_progress()
            curriculum_values = self.curriculum_scheduler.current_logging_difficulty(progress).to_dict()
            for key, value in curriculum_values.items():
                self.log(
                    f"{mode}_curriculum_{key}",
                    torch.tensor(value, device=self.device, dtype=torch.float32),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_records = []

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int = 0,
        *args,
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of  (i)     MS/MS spectra,[batch_size, peak_num, 2] 
                        (ii)    precursor information, [batch_size, 3],mass,charge,m/z
                        (iii)   peptide sequences.[batch_size],peptides

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # import pdb;pdb.set_trace()
        
        # Record the loss.
        loss = self.training_step(batch, batch_idx=batch_idx, mode="valid")

        sequence_labels = list(batch[2])
        has_sequence_labels = self._batch_has_sequence_labels(sequence_labels)
        decode_indices = self._validation_decode_indices(
            batch_size=batch[0].shape[0],
            batch_idx=batch_idx,
            device=batch[0].device,
        )
        if decode_indices.numel() > 0:
            selected_indices = decode_indices.to(batch[0].device)
            peptides_pred, peptides_score = self._decode_top_predictions(
                batch[0][selected_indices],
                batch[1][selected_indices],
            )
            selected_cpu = decode_indices.detach().cpu().tolist()
            peptides_true = [sequence_labels[idx] for idx in selected_cpu]

            if has_sequence_labels and len(peptides_pred) == len(peptides_true):
                batch_records = [
                    {
                        "peptides_true": truth,
                        "peptides_pred": pred,
                        "peptides_score": score,
                        "pep_matches": self._is_peptide_match(truth, pred),
                    }
                    for truth, pred, score in zip(peptides_true, peptides_pred, peptides_score)
                ]
                self._validation_records.extend(batch_records)

            if self.saved_path and len(peptides_pred) == len(peptides_true):
                batch_df = pd.DataFrame({
                    "peptides_true": peptides_true,
                    "peptides_pred": peptides_pred,
                    "peptides_score": peptides_score,
                    "pep_matches": [
                        self._is_peptide_match(truth, pred)
                        for truth, pred in zip(peptides_true, peptides_pred)
                    ],
                })
                output_path = Path(self.saved_path)
                batch_df.to_csv(self.saved_path, mode="a", header=not output_path.exists(), index=False)

        return loss


    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["train_CELoss"].detach()
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss.item(),
        }
        self._history.append(metrics)
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        if self._validation_records:
            records = self._gather_validation_records()
            prediction_metrics = self._compute_prediction_metrics(records)
            for key, value in prediction_metrics.items():
                self.log(
                    f"valid_{key}",
                    torch.tensor(value, device=self.device, dtype=torch.float32),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
            failure_profile = build_failure_profile(
                records,
                high_conf_threshold=self.analysis_high_confidence_threshold,
            )
            for key, value in flatten_failure_profile_for_logging(failure_profile).items():
                self.log(
                    f"valid_{key}",
                    torch.tensor(value, device=self.device, dtype=torch.float32),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "step": self.trainer.global_step,
            "valid": callback_metrics["valid_CELoss"].detach().item(),
        }
        self._history.append(metrics)
        self._log_history()


    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain loss\tValid loss\t"
            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f\t%.6f"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
                metrics.get("valid", np.nan),
            ]
            logger.info(msg % tuple(vals))

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup_iters, max_iters=self.max_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


def _aa_pep_score(
    aa_scores: np.ndarray, fits_precursor_mz: bool
) -> Tuple[np.ndarray, float]:
    """
    Calculate amino acid and peptide-level confidence score from the raw amino
    acid scores.

    The peptide score is the mean of the raw amino acid scores. The amino acid
    scores are the mean of the raw amino acid scores and the peptide score.

    Parameters
    ----------
    aa_scores : np.ndarray
        Amino acid level confidence scores.
    fits_precursor_mz : bool
        Flag indicating whether the prediction fits the precursor m/z filter.

    Returns
    -------
    aa_scores : np.ndarray
        The amino acid scores.
    peptide_score : float
        The peptide score.
    """
    peptide_score = np.mean(aa_scores)
    aa_scores = (aa_scores + peptide_score) / 2
    if not fits_precursor_mz:
        peptide_score -= 1
    return aa_scores, peptide_score
