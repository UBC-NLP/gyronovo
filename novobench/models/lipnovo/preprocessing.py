"""Preprocessing helpers for LipNovo spectra."""

import os
from itertools import repeat
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".numba-cache"))

import numpy as np
import polars as pl
import spectrum_utils.spectrum as sus

from novobench.data.base import SpectrumData
from novobench.transforms.base import BaseTransform
from novobench.transforms.filter import _map_transform


def _dummy_lipnovo_spectrum():
    """Return the row-preserving fallback used for invalid/empty spectra."""
    return [0.0], [1.0]


def _process_lipnovo_spectrum(args):
    (
        precursor_mz,
        precursor_charge,
        mz_array,
        int_array,
        min_mz,
        max_mz,
        remove_precursor_tol,
        min_intensity,
        n_peaks,
    ) = args

    mz_values = mz_array.to_numpy().astype(np.float32)
    intensity_values = int_array.to_numpy().astype(np.float32)
    if mz_values.size == 0 or intensity_values.size == 0:
        return _dummy_lipnovo_spectrum()

    spectrum = sus.MsmsSpectrum(
        "",
        precursor_mz,
        precursor_charge,
        mz_values,
        intensity_values,
    )
    spectrum.set_mz_range(min_mz, max_mz)
    if len(spectrum.mz) == 0:
        return _dummy_lipnovo_spectrum()

    spectrum.remove_precursor_peak(remove_precursor_tol, "Da")
    if len(spectrum.mz) == 0:
        return _dummy_lipnovo_spectrum()

    spectrum.filter_intensity(min_intensity, n_peaks)
    if len(spectrum.mz) == 0:
        return _dummy_lipnovo_spectrum()

    spectrum.scale_intensity("root", 1)
    intensity_norm = np.linalg.norm(spectrum.intensity)
    if not np.isfinite(intensity_norm) or intensity_norm <= 0.0:
        return _dummy_lipnovo_spectrum()

    intensities = spectrum.intensity / intensity_norm
    if not np.all(np.isfinite(intensities)):
        return _dummy_lipnovo_spectrum()
    return spectrum.mz.tolist(), intensities.tolist()


class LipNovoPreprocessingPipeline(BaseTransform):
    """Apply LipNovo spectrum preprocessing in one pass over a split."""

    def __init__(
        self,
        min_mz: float,
        max_mz: float,
        remove_precursor_tol: float,
        min_intensity: float,
        n_peaks: int,
    ) -> None:
        super().__init__()
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity
        self.n_peaks = n_peaks

    def __call__(self, data: SpectrumData) -> SpectrumData:
        results = _map_transform(
            _process_lipnovo_spectrum,
            zip(
                data.precursor_mz,
                data.precursor_charge,
                data.mz_array,
                data.intensity_array,
                repeat(self.min_mz),
                repeat(self.max_mz),
                repeat(self.remove_precursor_tol),
                repeat(self.min_intensity),
                repeat(self.n_peaks),
            ),
        )

        updated_mz_arrays = []
        updated_intensity_arrays = []
        invalid_spectra = 0
        for mz_lists, intensity_lists in results:
            updated_mz_arrays.append(mz_lists)
            updated_intensity_arrays.append(intensity_lists)
            if mz_lists == [0.0] and intensity_lists == [1.0]:
                invalid_spectra += 1

        data.set_df(
            data.df.with_columns(
                [
                    pl.Series("mz_array", updated_mz_arrays),
                    pl.Series("intensity_array", updated_intensity_arrays),
                ]
            )
        )

        print(
            "LIPNOVO PREPROCESSING DONE "
            f"(replaced {invalid_spectra} empty/invalid spectra with a dummy peak)"
        )
        return data
