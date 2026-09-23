"""SFLM: shared functional latent map (slow stage, latent.py) and the SFLM-EB closed-form fast stage (fast.py)."""

from src.sflm.fast import EB, TRAIN, build_cache, eb_fit_mackay, fit_alpha, kernel_average, kernel_sums, population, ridge_fixed, support_offset
from src.sflm.latent import (DATA, DEV, STREAMS, GaussianReadout, GridStore, LatentMap, SourceReadout, gaussian_dictionary, mask_dictionary,
                             split_8400_600, train_latent)

__all__ = ["EB", "TRAIN", "build_cache", "eb_fit_mackay", "fit_alpha", "kernel_average", "kernel_sums", "population", "ridge_fixed",
           "support_offset", "DATA", "DEV", "STREAMS", "GridStore", "LatentMap", "SourceReadout", "GaussianReadout", "gaussian_dictionary", "mask_dictionary", "split_8400_600",
           "train_latent"]
