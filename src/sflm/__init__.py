"""SFLM: shared functional latent map (slow stage, latent.py) and the SFLM-EB closed-form fast stage.

Default fast stage: fast_cont.py (A_cont, continuous Gaussian RFs of the population voxels as candidates); fast.py keeps the
mask-dictionary variant (eb_mackay) as a reference.
"""

from src.sflm.fast_cont import EBcont, build_cache_cont, eb_fit_mackay_cont, population_cont
from src.sflm.fast import EB, TRAIN, build_cache, eb_fit_mackay, fit_alpha, kernel_average, kernel_sums, population, ridge_fixed, support_offset
from src.sflm.latent import (DATA, DEV, STREAMS, GaussianReadout, GridStore, LatentMap, SourceReadout, gaussian_dictionary, mask_dictionary,
                             split_8400_600, train_latent)

__all__ = ["EBcont", "build_cache_cont", "eb_fit_mackay_cont", "population_cont", "EB", "TRAIN", "build_cache", "eb_fit_mackay", "fit_alpha", "kernel_average", "kernel_sums", "population", "ridge_fixed",
           "support_offset", "DATA", "DEV", "STREAMS", "GridStore", "LatentMap", "SourceReadout", "GaussianReadout", "gaussian_dictionary", "mask_dictionary", "split_8400_600",
           "train_latent"]
