"""
Covariate encoding specification for the ICPSR 22140 dataset.

The 72-dimensional one-hot covariate vector is composed of 17 categorical
fields. Each field occupies a contiguous slice of the vector, with exactly
one entry set to 1 within each group.

This module is the single source of truth for the covariate schema,
replacing hardcoded index ranges scattered across the codebase.
"""

import numpy as np

# (field_name, start_index, end_index) — end is exclusive
# Group sizes derived from the reference diffusion_model.py one_hot() function,
# which reflects the unique values per column in the ICPSR dataset.
COVARIATE_GROUPS: list[tuple[str, int, int]] = [
    ("LOCAL",    0,  4),
    ("RACE",     4,  11),
    ("ETHN",     11, 15),
    ("SEX",      15, 18),
    ("ORIENT",   18, 24),
    ("BEHAV",    24, 27),
    ("PRO",      27, 31),
    ("PIMP",     31, 35),
    ("JOHN",     35, 39),
    ("DEALER",   39, 43),
    ("DRUGMAN",  43, 47),
    ("THIEF",    47, 51),
    ("RETIRED",  51, 55),
    ("HWIFE",    55, 59),
    ("DISABLE",  59, 64),
    ("UNEMP",    64, 68),
    ("STREETS",  68, 72),
]

COVARIATE_DIM = 72


def continuous_to_one_hot(x: np.ndarray) -> np.ndarray:
    """Convert continuous diffusion output to valid one-hot covariates.

    For each covariate group, sets the argmax entry to 1 and all others to 0.

    Args:
        x: Array of shape (n, 72) with continuous values.

    Returns:
        Array of shape (n, 72) with valid one-hot encoding (int).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    assert x.shape[1] == COVARIATE_DIM

    result = np.zeros_like(x, dtype=int)
    for _, start, end in COVARIATE_GROUPS:
        group = x[:, start:end]
        winners = np.argmax(group, axis=1)
        result[np.arange(x.shape[0]), start + winners] = 1
    return result


def validate_one_hot(x: np.ndarray) -> bool:
    """Check that a covariate array has valid one-hot encoding.

    Args:
        x: Array of shape (n, 72) or (72,).

    Returns:
        True if every row has exactly one 1 per covariate group and 0 elsewhere.
    """
    x = np.asarray(x, dtype=int)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.shape[1] != COVARIATE_DIM:
        return False
    for _, start, end in COVARIATE_GROUPS:
        group_sums = x[:, start:end].sum(axis=1)
        if not np.all(group_sums == 1):
            return False
    if not np.all((x == 0) | (x == 1)):
        return False
    return True
