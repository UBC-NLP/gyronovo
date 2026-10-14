"""Helpers for optional W&B logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def wandb_is_configured(mode: str | None = None) -> bool:
    """Check whether W&B should be considered available for auto-logging."""
    if _truthy(os.getenv("WANDB_DISABLED")):
        return False

    effective_mode = (mode or os.getenv("WANDB_MODE") or "").strip().lower()
    if effective_mode == "disabled":
        return False
    if effective_mode in {"offline", "dryrun"}:
        return True
    if os.getenv("WANDB_API_KEY"):
        return True

    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            netrc_text = netrc_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return "api.wandb.ai" in netrc_text or "wandb.ai" in netrc_text
    return False


def build_wandb_logger(config, model_name: str, job_type: str):
    """Create a WandbLogger only when the package and credentials are present."""
    if not getattr(config, "use_wandb", True):
        return None

    try:
        from lightning.pytorch.loggers import WandbLogger
    except Exception:
        logger.info("W&B logging requested, but lightning WandbLogger is unavailable; continuing without W&B.")
        return None

    wandb_mode = getattr(config, "wandb_mode", None)
    if not wandb_is_configured(wandb_mode):
        logger.info("W&B is not configured; continuing without W&B logging.")
        return None

    save_dir = Path(getattr(config, "wandb_save_dir", "./wandb")).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = getattr(config, "dataset_name", None)
    run_name = getattr(config, "wandb_name", None)
    if run_name is None:
        run_suffix = Path(getattr(config, "model_save_folder_path", model_name)).expanduser().resolve().parent.name
        run_name = f"{model_name}-{run_suffix}-{job_type}"
    if dataset_name and str(dataset_name) not in str(run_name):
        run_name = f"{run_name}-{dataset_name}"

    offline = (wandb_mode or os.getenv("WANDB_MODE") or "").strip().lower() in {"offline", "dryrun"}
    logger_kwargs = dict(
        project=getattr(config, "wandb_project", "novobench"),
        entity=getattr(config, "wandb_entity", None),
        name=run_name,
        save_dir=str(save_dir),
        log_model=getattr(config, "wandb_log_model", False),
        group=f"{model_name}-{dataset_name}" if dataset_name else model_name,
        job_type=job_type,
        tags=[f"dataset:{dataset_name}"] if dataset_name else None,
        offline=offline,
        anonymous="never",
    )
    logger_kwargs = {key: value for key, value in logger_kwargs.items() if value is not None}
    return WandbLogger(**logger_kwargs)
