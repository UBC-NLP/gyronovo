"""
Training and testing functionality for the de novo peptide sequencing model.
"""

import copy
import logging
import sys
import warnings
from pathlib import Path
import time
from typing import Optional, Union

import lightning.pytorch as pln
import polars as pl
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from novobench.utils.wandb import build_wandb_logger

from .lipnovo_dataloader import LipNovoDataModule
from .lipnovo_modeling import Spec2Pep
from .preprocessing import LipNovoPreprocessingPipeline

logger = logging.getLogger("lipnovo")


def init_logger(config):
    """Configure logging once per output file."""
    output = Path(config.logger_save_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    has_console = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(log_formatter)
        root.addHandler(console_handler)

    output_str = str(output)
    has_file = any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == output_str
        for handler in root.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(output_str)
        file_handler.setFormatter(log_formatter)
        root.addHandler(file_handler)

    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    

class LipNovoRunner:
    """A class to run LipNovo models.
    Parameters
    ----------
    config : Config object
        The lipnovo configuration.
    model_filename : str, optional
        The model filename is required for eval and de novo modes,
        but not for training a model from scratch.
    """

    def __init__(
        self,
        config,
        model_filename: Optional[str] = None,
        saved_path: Optional[str] = "",
    ) -> None:
        seed_everything(seed=config.random_seed, workers=True)
        init_logger(config)
        self.config = config
        self.model_filename = model_filename
        self.saved_path = saved_path

        Path(self.config.model_save_folder_path).mkdir(parents=True, exist_ok=True)

        # Initialized later:
        self.trainer = None
        self.model = None

        # Configure checkpoints.
        monitor_metric = getattr(config, "monitor_metric", None) or "valid_CELoss"
        monitor_mode = getattr(config, "monitor_mode", None) or self._infer_checkpoint_mode(monitor_metric)
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")
        callbacks = []
        if config.save_top_k is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=config.model_save_folder_path,
                    monitor=monitor_metric,
                    mode=monitor_mode,
                    save_top_k=config.save_top_k,
                )
            )

        early_stopping_enabled = bool(getattr(config, "early_stopping_enabled", False))
        if early_stopping_enabled:
            early_stopping_monitor = (
                getattr(config, "early_stopping_monitor", None) or monitor_metric
            )
            early_stopping_mode = (
                getattr(config, "early_stopping_mode", None)
                or (
                    monitor_mode
                    if early_stopping_monitor == monitor_metric
                    else self._infer_checkpoint_mode(early_stopping_monitor)
                )
            )
            if early_stopping_mode not in {"min", "max"}:
                raise ValueError("early_stopping_mode must be 'min' or 'max'")
            callbacks.append(
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    mode=early_stopping_mode,
                    patience=int(getattr(config, "early_stopping_patience", 5)),
                    min_delta=float(getattr(config, "early_stopping_min_delta", 0.0)),
                    check_on_train_epoch_end=False,
                )
            )
        self.callbacks = callbacks or None


    @staticmethod
    def _infer_checkpoint_mode(monitor_metric: str) -> str:
        lower_metric = monitor_metric.lower()
        if any(token in lower_metric for token in ("loss", "error", "mae")):
            return "min"
        return "max"


    @staticmethod
    def preprocessing_pipeline(config):
        return LipNovoPreprocessingPipeline(
            min_mz=config.min_mz,
            max_mz=config.max_mz,
            remove_precursor_tol=config.remove_precursor_tol,
            min_intensity=config.min_intensity,
            n_peaks=config.n_peaks,
        )
    

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
    ) -> None:
        """Train the LipNovo model.

        Parameters
        ----------
        train_peak_path : iterable of str
            The path to the MS data files for training.
        valid_peak_path : iterable of str
            The path to the MS data files for validation.

        Returns
        -------
        self
        """
        self.initialize_trainer(train=True)
        self.initialize_model(train=True)

        train_loader = LipNovoDataModule(
            df = train_df,
            n_workers=self.config.n_workers,
            batch_size=self.config.train_batch_size // self.trainer.num_devices,
        ).get_dataloader(shuffle=True)
        
        val_loader = LipNovoDataModule(
            df = val_df,
            n_workers=self.config.n_workers,
            batch_size=self.config.train_batch_size // self.trainer.num_devices
        ).get_dataloader()

        start_time = time.time()
        self.trainer.fit(
            self.model,
            train_loader,
            val_loader,
        )
        training_time = time.time() - start_time
        logger.info(f"Training took {training_time:.2f} seconds")
        best_model_path = self._write_best_checkpoint_path()
        self._save_best_validation_predictions(val_df, best_model_path)

    def _write_best_checkpoint_path(self) -> str | None:
        if not self.callbacks:
            return None
        for callback in self.callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            best_model_path = callback.best_model_path
            if not best_model_path:
                continue
            trainer = getattr(self, "trainer", None)
            if trainer is None or trainer.is_global_zero:
                output_path = Path(self.config.model_save_folder_path) / "best_model_path.txt"
                output_path.write_text(f"{best_model_path}\n", encoding="utf-8")
                logger.info(
                    "Best checkpoint by %s (%s): %s",
                    callback.monitor,
                    callback.mode,
                    best_model_path,
                )
            return best_model_path
        return None

    def _best_validation_predictions_output_path(self) -> Path:
        analysis_config = getattr(self.config, "analysis", {}) or {}
        configured_path = analysis_config.get("best_validation_predictions_output")
        if configured_path:
            return Path(configured_path).expanduser()
        return Path(self.config.model_save_folder_path).expanduser().parent / "best_validation_predictions.csv"

    def _save_best_validation_predictions(
        self,
        val_df: pl.DataFrame,
        best_model_path: str | None,
    ) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        if not best_model_path:
            logger.warning(
                "Skipping best-checkpoint validation prediction export because no best checkpoint is available."
            )
            return
        if not Path(best_model_path).exists():
            logger.warning(
                "Skipping best-checkpoint validation prediction export because checkpoint does not exist: %s",
                best_model_path,
            )
            return

        output_path = self._best_validation_predictions_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        logger.info(
            "Saving validation predictions from best checkpoint to %s",
            output_path,
        )

        prediction_config = copy.copy(self.config)
        prediction_config.devices = 1
        prediction_runner = LipNovoRunner(
            prediction_config,
            model_filename=best_model_path,
            saved_path=str(output_path),
        )
        prediction_runner.denovo(val_df)

    def denovo(self, test_df: pl.DataFrame,) -> None:
        """Evaluate peptide sequence preditions from a trained LipNovo model.

        Parameters
        ----------
        peak_path : iterable of str
            The path with MS data files for predicting peptide sequences.

        Returns
        -------
        self
        """
        self.initialize_trainer(train=False)
        self.initialize_model(train=False)
        test_loader = LipNovoDataModule(
            df = test_df,
            n_workers=self.config.n_workers,
            batch_size=self.config.predict_batch_size // self.trainer.num_devices 
        ).get_dataloader()
        
        start_time = time.time()
        self.trainer.validate(self.model, test_loader)
        training_time = time.time() - start_time
        logger.info(f"denovo took {training_time:.2f} seconds")



    def initialize_trainer(self, train: bool) -> None:
        """Initialize the lightning Trainer.

        Parameters
        ----------
        train : bool
            Determines whether to set the trainer up for model training
            or evaluation / inference.
        """
        trainer_logger = build_wandb_logger(
            self.config,
            model_name="lipnovo",
            job_type="train" if train else "denovo",
        )
        configured_devices = getattr(self.config, "devices", None)
        devices = "auto" if configured_devices is None else configured_devices
        if not train and self.saved_path:
            devices = 1

        trainer_cfg = dict(
            accelerator=self.config.accelerator,
            devices=devices,
            enable_checkpointing=False,
            logger=trainer_logger if trainer_logger is not None else True,
        )

        if train:
            additional_cfg = dict(
                callbacks=self.callbacks,
                enable_checkpointing=self.config.save_top_k is not None,
                max_epochs=self.config.max_epochs,
                num_sanity_val_steps=self.config.num_sanity_val_steps,
                strategy=self._get_strategy(),
                val_check_interval=self.config.val_check_interval,
                check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            )
            gradient_clip_val = getattr(self.config, "gradient_clip_val", None)
            if gradient_clip_val is not None:
                additional_cfg["gradient_clip_val"] = float(gradient_clip_val)
            trainer_cfg.update(additional_cfg)

        self.trainer = pln.Trainer(**trainer_cfg)

    def initialize_model(self, train: bool) -> None:
        """Initialize the LipNovo model.

        Parameters
        ----------
        train : bool
            Determines whether to set the model up for model training
            or evaluation / inference.
        """
        validation_decode_fraction = getattr(self.config, "validation_decode_fraction", 0.1)
        if not train:
            validation_decode_fraction = 1.0

        model_cls = Spec2Pep
        gen_num = self.config.gen_num
        gen_enc_layers = getattr(self.config, "gen_enc_layers", 3)
        gen_dec_layers = getattr(self.config, "gen_dec_layers", 3)
        gen_threshold = getattr(self.config, "gen_threshold", 0.8)

        model_params = dict(
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            dim_intensity=self.config.dim_intensity,
            max_length=self.config.max_length,
            residues=self.config.residues,
            max_charge=self.config.max_charge,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            min_peptide_len=self.config.min_peptide_len,
            n_beams=self.config.n_beams,
            top_match=self.config.top_match,
            train_label_smoothing=self.config.train_label_smoothing,
            validation_decode_fraction=validation_decode_fraction,
            warmup_iters=self.config.warmup_iters,
            max_iters=self.config.max_iters,
            saved_path=self.saved_path,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gen_num=gen_num,
            gen_enc_layers=gen_enc_layers,
            gen_dec_layers=gen_dec_layers,
            gen_threshold=gen_threshold,
            n_peaks=self.config.n_peaks,
            random_seed=self.config.random_seed,
            augmentation=getattr(self.config, "augmentation", {}),
            curriculum=getattr(self.config, "curriculum", {}),
            auxiliary_losses=getattr(self.config, "auxiliary_losses", {}),
            analysis=getattr(self.config, "analysis", {}),
            mz_rope=getattr(self.config, "mz_rope", {}),
            training_code=getattr(self.config, "training_code", "extended"),
        )

        # Reconfigurable non-architecture related parameters for a loaded model
        loaded_model_params = dict(
            max_length=self.config.max_length,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            n_beams=self.config.n_beams,
            min_peptide_len=self.config.min_peptide_len,
            top_match=self.config.top_match,
            train_label_smoothing=self.config.train_label_smoothing,
            validation_decode_fraction=validation_decode_fraction,
            warmup_iters=self.config.warmup_iters,
            max_iters=self.config.max_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gen_num=gen_num,
            gen_enc_layers=gen_enc_layers,
            gen_dec_layers=gen_dec_layers,
            gen_threshold=gen_threshold,
            n_peaks=self.config.n_peaks,
            random_seed=self.config.random_seed,
            augmentation=getattr(self.config, "augmentation", {}),
            curriculum=getattr(self.config, "curriculum", {}),
            auxiliary_losses=getattr(self.config, "auxiliary_losses", {}),
            analysis=getattr(self.config, "analysis", {}),
            mz_rope=getattr(self.config, "mz_rope", {}),
            training_code=getattr(self.config, "training_code", "extended"),
        )

        if self.model_filename is None:
            # Train a model from scratch if no model file is provided.
            if train:
                self.model = model_cls(**model_params)
                return
            # Else we're not training, so a model file must be provided.
            else:
                logger.error("A model file must be provided")
                raise ValueError("A model file must be provided")
        # Else a model file is provided (to continue training or for inference).

        if not Path(self.model_filename).exists():
            logger.error(
                "Could not find the model weights at file %s",
                self.model_filename,
            )
            raise FileNotFoundError("Could not find the model weights file")

        # First try loading model details from the weights file, otherwise use
        # the provided configuration.
        device = torch.empty(1).device  # Use the default device.
        try:
            self.model = model_cls.load_from_checkpoint(
                self.model_filename,
                map_location=device,
                saved_path=self.saved_path,
                **loaded_model_params,
            )

            architecture_params = set(model_params.keys()) - set(
                loaded_model_params.keys()
            )
            for param in architecture_params:
                checkpoint_value = self.model.hparams.get(param, None)
                if model_params[param] != checkpoint_value:
                    warnings.warn(
                        f"Mismatching {param} parameter in "
                        f"model checkpoint ({checkpoint_value}) "
                        f"vs config file ({model_params[param]}); "
                        "using the checkpoint."
                    )
        except RuntimeError:
            # This only doesn't work if the weights are from an older version
            try:
                self.model = model_cls.load_from_checkpoint(
                    self.model_filename,
                    map_location=device,
                    **model_params,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Weights file incompatible with the current version of "
                    "LipNovo. "
                ) from exc

    def _get_strategy(self) -> Union[str, DDPStrategy]:
        """Get the strategy for the Trainer.

        The DDP strategy works best when multiple GPUs are used. It can work
        for CPU-only, but definitely fails using MPS (the Apple Silicon chip)
        due to Gloo.

        Returns
        -------
        Union[str, DDPStrategy]
            The strategy parameter for the Trainer.

        """
        if self.config.accelerator in ("cpu", "mps"):
            return "auto"
        elif self.config.devices == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"
