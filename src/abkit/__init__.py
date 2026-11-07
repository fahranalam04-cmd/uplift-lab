"""abkit: a small, tested toolkit for analysing randomized experiments.

Modules
-------
srm                 sample-ratio-mismatch check
balance             covariate balance (SMD, treatment-predictability AUC)
inference           Welch / bootstrap difference in means, absolute and relative lift
power               power, MDE and sample size
variance_reduction  CUPED, Lin regression adjustment, cross-fitted CUPAC
sequential          mSPRT always-valid p-values and confidence sequences
uplift              meta-learners and uplift metrics
"""

from .balance import smd_from_moments, standardized_mean_differences, treatment_predictability_auc
from .inference import (
    ArmSummary,
    EffectEstimate,
    bootstrap_diff_in_means,
    bootstrap_from_binary_counts,
    diff_in_means,
    welch_from_summaries,
)
from .power import minimum_detectable_effect, power_at, required_sample_size
from .sequential import always_valid_ci, always_valid_p_values, msprt_test
from .srm import srm_test
from .variance_reduction import (
    MomentAccumulator,
    crossfit_predictions,
    cupac,
    cuped,
    equivalent_sample_size_saving,
    lin_from_moments,
    regression_adjusted,
)

__version__ = "0.1.0"

__all__ = [
    "ArmSummary",
    "EffectEstimate",
    "MomentAccumulator",
    "always_valid_ci",
    "always_valid_p_values",
    "bootstrap_diff_in_means",
    "bootstrap_from_binary_counts",
    "crossfit_predictions",
    "cupac",
    "cuped",
    "diff_in_means",
    "equivalent_sample_size_saving",
    "lin_from_moments",
    "minimum_detectable_effect",
    "msprt_test",
    "power_at",
    "regression_adjusted",
    "required_sample_size",
    "smd_from_moments",
    "srm_test",
    "standardized_mean_differences",
    "treatment_predictability_auc",
    "welch_from_summaries",
]
