"""Interval estimates for hand-labelled validation samples."""
import math
from statistics import NormalDist


def wilson_ci(k, n, conf=0.95):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
