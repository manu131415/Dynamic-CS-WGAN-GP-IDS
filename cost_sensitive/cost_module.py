"""
Common cost module - reused by: Static-cost IDS, Dynamic-cost IDS,
and later WGAN-GP / Stackelberg stages.

Every cost object exposes:
    fit(df)              -> learn any statistics from TRAINING raw data
    costs(df)            -> (c_fn, c_fp) arrays, one value per sample
    sample_weights(df)   -> per-sample training weights (mean-normalised)
    total_cost(df, y_pred) -> evaluation cost of a set of predictions

`df` is the RAW dataframe (before preprocessing) containing at least:
label, attack_cat, sbytes, dbytes.
"""

import numpy as np


# Assumed severity of MISSING each attack type (justify in the paper;
# adjust to your threat model).
ATTACK_SEVERITY = {
    "Generic": 2.0,
    "Fuzzers": 2.0,
    "Reconnaissance": 3.0,
    "Analysis": 3.0,
    "DoS": 4.0,
    "Exploits": 5.0,
    "Backdoor": 6.0,
    "Shellcode": 7.0,
    "Worms": 8.0,
}
DEFAULT_SEVERITY = 3.0


class StaticCost:
    """Fixed cost for every FN and every FP."""

    name = "static"

    def __init__(self, c_fn=5.0, c_fp=1.0):
        self.c_fn = c_fn
        self.c_fp = c_fp

    def fit(self, df):
        return self

    def costs(self, df):
        n = len(df)
        return np.full(n, self.c_fn), np.full(n, self.c_fp)

    def sample_weights(self, df):
        c_fn, c_fp = self.costs(df)
        y = df["label"].values
        w = np.where(y == 1, c_fn, c_fp)
        return w / w.mean()

    def total_cost(self, df, y_pred):
        return _total_cost(self, df, y_pred)


class DynamicCost:
    """
    Per-sample cost.
      FN cost = attack severity x traffic-volume factor
      FP cost = base x traffic-volume factor (blocking heavy benign traffic hurts more)
    Costs are rescaled on the training set so that the MEAN FN cost = target_mean_fn
    and the MEAN FP cost = c_fp_base. This keeps the average cost magnitude equal to
    the static scheme, so any difference in results comes from the cost being
    dynamic, not from larger numbers.
    """

    name = "dynamic"

    def __init__(self, target_mean_fn=5.0, c_fp_base=1.0):
        self.target_mean_fn = target_mean_fn
        self.c_fp_base = c_fp_base
        self.vmax_ = None
        self.fn_scale_ = None
        self.fp_scale_ = None

    # ---- internals ----
    def _volume(self, df):
        return np.log1p(df["sbytes"].values + df["dbytes"].values)

    def _vol_factor(self, df):
        return 1.0 + np.clip(self._volume(df) / self.vmax_, 0.0, 1.0)  # range 1..2

    def _raw_fn(self, df):
        sev = df["attack_cat"].map(ATTACK_SEVERITY).fillna(DEFAULT_SEVERITY).values
        return sev * self._vol_factor(df)

    # ---- API ----
    def fit(self, df):
        self.vmax_ = self._volume(df).max()
        attack = df["label"].values == 1
        self.fn_scale_ = self.target_mean_fn / self._raw_fn(df)[attack].mean()
        self.fp_scale_ = self.c_fp_base / self._vol_factor(df)[~attack].mean()
        return self

    def costs(self, df):
        c_fn = self.fn_scale_ * self._raw_fn(df)
        c_fp = self.fp_scale_ * self._vol_factor(df)
        return c_fn, c_fp

    def sample_weights(self, df):
        c_fn, c_fp = self.costs(df)
        y = df["label"].values
        w = np.where(y == 1, c_fn, c_fp)
        return w / w.mean()

    def total_cost(self, df, y_pred):
        return _total_cost(self, df, y_pred)


def _total_cost(cost_obj, df, y_pred):
    """Sum of costs of all errors (FN uses c_fn, FP uses c_fp). TP/TN cost 0."""
    y_true = df["label"].values
    y_pred = np.asarray(y_pred)
    c_fn, c_fp = cost_obj.costs(df)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    total = c_fn[fn].sum() + c_fp[fp].sum()
    return float(total), float(total / len(df))