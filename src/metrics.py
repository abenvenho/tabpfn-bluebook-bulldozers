"""RMSLE, paired bootstrap and interval coverage — as pre-registered."""
import numpy as np


def rmsle(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def bootstrap_rmsle_diff(y_true, pred_a, pred_b, n_boot=10_000, seed=42) -> dict:
    """RMSLE(A) − RMSLE(B) with a paired percentile bootstrap over the evaluation sales.

    Positive diff => A is worse. Returns the point difference and the 95% CI.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=float)
    a = np.clip(np.asarray(pred_a, dtype=float), 0, None)
    b = np.clip(np.asarray(pred_b, dtype=float), 0, None)
    la, lb = (np.log1p(a) - np.log1p(y)) ** 2, (np.log1p(b) - np.log1p(y)) ** 2
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.sqrt(la[idx].mean(axis=1)) - np.sqrt(lb[idx].mean(axis=1))
    return {
        "diff": float(np.sqrt(la.mean()) - np.sqrt(lb.mean())),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def interval_coverage(y_true, lo, hi) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lo, dtype=float)) &
                         (y <= np.asarray(hi, dtype=float))))
