"""Configuration loading for the retained GyroNovo and LIPNovo workflows."""

import yaml


class Config:
    def __init__(self, config_file, model_name):
        self.model_name = model_name
        with open(config_file) as f_in:
            self.config = yaml.safe_load(f_in)
        self.check_config_type()
        print(self.config)

    def check_config_type(self):
        config_types = dict(
            random_seed=int,
            n_peaks=int,
            min_mz=float,
            max_mz=float,
            min_intensity=float,
            remove_precursor_tol=float,
            max_charge=int,
            precursor_mass_tol=float,
            isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
            warmup_iters=int,
            max_iters=int,
            gradient_clip_val=float,
            num_sanity_val_steps=int,
            learning_rate=float,
            weight_decay=float,
            max_epochs=int,
            train_batch_size=int,
            save_weights_only=bool,
            model_save_folder_path=str,
            logger_save_path=str,
            predict_batch_size=int,
            val_check_interval=int,
            check_val_every_n_epoch=int,
            n_workers=int,
            save_top_k=int,
            devices=int,
            monitor_metric=str,
            monitor_mode=str,
            early_stopping_enabled=bool,
            early_stopping_patience=int,
            early_stopping_min_delta=float,
            early_stopping_monitor=lambda value: None if value is None else str(value),
            early_stopping_mode=lambda value: None if value is None else str(value),
            use_wandb=bool,
            wandb_project=str,
            wandb_entity=lambda value: None if value is None else str(value),
            wandb_name=lambda value: None if value is None else str(value),
            wandb_save_dir=str,
            wandb_mode=lambda value: None if value is None else str(value),
            wandb_log_model=bool,
        )
        for key, value_type in config_types.items():
            try:
                if key in self.config and self.config[key] is not None:
                    self.config[key] = value_type(self.config[key])
            except (TypeError, ValueError) as error:
                print("Incorrect type for configuration value %s: %s", key, error)
                raise TypeError(
                    f"Incorrect type for configuration value {key}: {error}"
                ) from error

        for key, value in self.config.items():
            setattr(self, key, value)
        self.check_model_config_type(self.model_name)

    def check_model_config_type(self, model_name):
        if model_name != "lipnovo":
            raise ValueError(f"Unsupported model: {model_name}")
        self.check_lipnovo_config_type()

    def check_lipnovo_config_type(self):
        section = "lipnovo"
        config_types = dict(
            dim_model=int,
            n_head=int,
            dim_feedforward=int,
            n_layers=int,
            dropout=float,
            dim_intensity=lambda value: None if value is None else int(value),
            max_length=int,
            min_peptide_len=int,
            train_label_smoothing=float,
            validation_decode_fraction=float,
            residues=dict,
            n_beams=int,
            top_match=int,
            training_code=str,
            accelerator=str,
            gen_num=int,
            gen_enc_layers=int,
            gen_dec_layers=int,
            gen_threshold=float,
        )
        for key, value_type in config_types.items():
            try:
                if key in self.config[section] and self.config[section][key] is not None:
                    self.config[section][key] = value_type(self.config[section][key])
            except (TypeError, ValueError) as error:
                print("Incorrect type for configuration value %s: %s", key, error)
                raise TypeError(
                    f"Incorrect type for configuration value {key}: {error}"
                ) from error

        nested_sections = {
            "augmentation": {
                "enabled": bool,
                "num_views": int,
                "view_types": list,
                "easy_view_scale": float,
                "hard_view_scale": float,
                "easy_view_weight": float,
                "hard_view_weight": float,
                "no_decoder_for_augmented_views": bool,
                "views": {
                    "types": list,
                    "decode": bool,
                    "easy": {
                        "difficulty_scale": float,
                        "loss_weight": float,
                    },
                    "hard": {
                        "difficulty_scale": float,
                        "loss_weight": float,
                    },
                },
                "max_rendered_peaks": int,
                "min_mz": float,
                "observed_anchor_enabled": bool,
                "observed_anchor_mz_tolerance": float,
                "observed_anchor_keep_prob": float,
                "observed_anchor_noise_keep_prob": float,
                "observed_anchor_theory_add_prob": float,
                "observed_anchor_focus_drop_scale": float,
                "observed_anchor_focus_keep_scale": float,
                "observed_anchor_focus_add_scale": float,
                "theoretical_spectrum": {
                    "position_baseline": float,
                    "position_center_boost": float,
                    "b_ion_intensity_scale": float,
                    "y_ion_intensity_scale": float,
                    "ptm_intensity_boost": float,
                },
                "fallback_theoretical": {
                    "ptm_dropout_scale": float,
                    "maximum_drop_probability": float,
                },
                "observed_anchor": {
                    "enabled": bool,
                    "mz_tolerance": float,
                    "fragment_keep_probability": float,
                    "unannotated_keep_probability": float,
                    "missing_fragment_add_probability": float,
                    "easy_focus": {
                        "keep_probability_boost": float,
                        "add_probability_boost": float,
                        "intensity_boost": float,
                    },
                    "hard_focus": {
                        "keep_probability_penalty": float,
                        "add_probability_penalty": float,
                        "extra_drop_probability": float,
                        "intensity_penalty": float,
                    },
                    "keep_penalties": {
                        "peak_missingness": float,
                        "ion_dropout": float,
                        "ptm_visibility": float,
                    },
                    "add_penalties": {
                        "peak_missingness": float,
                        "ptm_visibility": float,
                    },
                    "probability_bounds": {
                        "keep_min": float,
                        "keep_max": float,
                        "add_min": float,
                        "hard_add_min": float,
                        "add_max": float,
                    },
                    "easy_ladder_focus_threshold": float,
                },
                "ladder_gap": {
                    "max_run_scale": float,
                },
                "structured_noise": {
                    "neutral_loss_masses": lambda values: [float(value) for value in values],
                    "mz_jitter_std": float,
                    "intensity_fraction_range": lambda values: [float(value) for value in values],
                    "reference_intensity_cap": float,
                    "observed_anchor_count_scale": float,
                },
            },
            "curriculum": {
                "mode": str,
                "ptm_visibility_multiplier": float,
                "decoder_guidance": {
                    "enabled": bool,
                    "mode": str,
                    "strength": float,
                    "sample_scale_min": float,
                    "sample_scale_max": float,
                    "loss_temperature": float,
                    "peak_focus_strength": float,
                    "peak_focus_radius": int,
                    "target_weight_min": float,
                    "target_weight_max": float,
                    "hard_threshold": float,
                },
                "initial_difficulty": {
                    "peak_missingness": float,
                    "b_ion_dropout": float,
                    "y_ion_dropout": float,
                    "fragmentation_ladder_gap": float,
                    "intensity_jitter": float,
                    "mz_jitter": float,
                    "structured_noise_count": float,
                    "ptm_visibility_reduction": float,
                },
                "max_difficulty": {
                    "peak_missingness": float,
                    "b_ion_dropout": float,
                    "y_ion_dropout": float,
                    "fragmentation_ladder_gap": float,
                    "intensity_jitter": float,
                    "mz_jitter": float,
                    "structured_noise_count": float,
                    "ptm_visibility_reduction": float,
                },
            },
            "auxiliary_losses": {
                "lambda_theory": float,
                "lambda_imp_real": float,
                "lambda_imp_aug": float,
                "lambda_decode_aug": float,
            },
            "analysis": {
                "best_validation_predictions_output": lambda value: None
                if value is None
                else str(value),
                "high_confidence_threshold": float,
            },
            "mz_rope": {
                "enabled": bool,
                "min_wavelength": float,
                "max_wavelength": float,
            },
        }
        def coerce_nested_values(values, value_types, path):
            for field, value_type in value_types.items():
                if field not in values or values[field] is None:
                    continue
                field_path = f"{path}.{field}"
                if isinstance(value_type, dict):
                    if not isinstance(values[field], dict):
                        raise TypeError(
                            f"Incorrect type for configuration value {field_path}: expected dict"
                        )
                    coerce_nested_values(values[field], value_type, field_path)
                    continue
                try:
                    values[field] = value_type(values[field])
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"Incorrect type for configuration value {field_path}: {error}"
                    ) from error

        for nested_key, nested_types in nested_sections.items():
            nested_values = self.config[section].get(nested_key)
            if nested_values is None:
                self.config[section][nested_key] = {}
                continue
            if not isinstance(nested_values, dict):
                raise TypeError(
                    f"Incorrect type for configuration value {nested_key}: expected dict"
                )
            coerce_nested_values(nested_values, nested_types, nested_key)

        self.config[section]["residues"] = {
            str(amino_acid): float(mass)
            for amino_acid, mass in self.config[section]["residues"].items()
        }
        for key, value in self.config[section].items():
            setattr(self, key, value)
