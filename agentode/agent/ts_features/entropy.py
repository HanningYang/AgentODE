"""Entropy and complexity statistics for time series."""

from collections import Counter

import numpy as np


def permutation_entropy_m3(x: np.ndarray, tau: int = 1) -> float:
    """Compute permutation entropy with embedding dimension m=3."""
    return _permutation_entropy(x, m=3, tau=tau)


def permutation_entropy_m5(x: np.ndarray, tau: int = 1) -> float:
    """Compute permutation entropy with embedding dimension m=5."""
    return _permutation_entropy(x, m=5, tau=tau)


def _permutation_entropy(x: np.ndarray, m: int, tau: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < m * tau:
        return np.nan
    patterns = [tuple(np.argsort(x[i : i + m * tau : tau])) for i in range(n - (m - 1) * tau)]
    counts = Counter(patterns)
    total = len(patterns)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return float(-np.sum(probs * np.log(probs)))


def lempel_ziv_complexity(x: np.ndarray) -> float:
    """Compute normalized Lempel-Ziv complexity of a median-thresholded binary series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    binary = (x >= np.median(x)).astype(int)
    s = "".join(map(str, binary))
    n = len(s)
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return float(c / (n / np.log2(n + 1e-10)))
