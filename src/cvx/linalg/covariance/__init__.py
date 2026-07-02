"""Covariance utilities: conversion, random generation, EWM, and PCA.

``ewm_covariance`` lives in the :mod:`cvx.linalg.covariance.ewm_cov` submodule
and is intentionally *not* re-exported here, because it requires the optional
``polars`` dependency (the ``ewm`` extra). Import it explicitly with
``from cvx.linalg.covariance.ewm_cov import ewm_covariance``.
"""

from .cov_to_corr import cov_to_corr as cov_to_corr
from .pca import pca as pca
from .rand_cov import rand_cov as rand_cov

__all__ = [
    "cov_to_corr",
    "pca",
    "rand_cov",
]
