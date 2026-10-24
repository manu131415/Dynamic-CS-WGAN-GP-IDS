"""
Feedback-based dynamic cost (Experiment 3).
Place at: <project>/feedback_cost/feedback_cost_module.py

Same API as StaticCost / DynamicCost (fit, costs, sample_weights, total_cost), so it
plugs directly into later WGAN-GP and Stackelberg stages.

Cost = DynamicCost (attack severity x traffic volume)  *  group multiplier
  FN groups : attack_cat (9 attack types)
  FP groups : service of NORMAL flows (top-k services + "other")
Multipliers are updated from validation error rates every round:

  FN group k:  FNR_k above target  -> raise m_fn[k]
               FNR_k clearly below target AND global FPR excessive -> lower m_fn[k]
  FP group g:  FPR_g above target  -> raise m_fp[g]
               FPR_g clearly below target AND global FNR excessive -> lower m_fp[g]
  otherwise (inside the dead band)  -> unchanged
"""
import numpy as np
from cost_module import DynamicCost


class FeedbackCost:
    name = "feedback"

    def __init__(self, target_mean_fn=5.0, c_fp_base=1.0, n_services=8, m_min=0.25, m_max=8.0):
        self.target_mean_fn, self.c_fp_base = target_mean_fn, c_fp_base
        self.n_services, self.m_min, self.m_max = n_services, m_min, m_max

    # ---- fit ----
    def fit(self, df):
        self.base_ = DynamicCost(self.target_mean_fn, self.c_fp_base).fit(df)
        self.fn_groups_ = sorted(df.loc[df["label"] == 1, "attack_cat"].unique())
        top = df.loc[df["label"] == 0, "service"].value_counts().head(self.n_services).index
        self.top_services_ = set(top)
        self.fp_groups_ = sorted(self.top_services_) + ["other"]
        self.m_fn_ = {g: 1.0 for g in self.fn_groups_}
        self.m_fp_ = {g: 1.0 for g in self.fp_groups_}
        return self

    def _fp_group(self, df):
        s = df["service"]
        return s.where(s.isin(self.top_services_), "other")

    # ---- cost API ----
    def costs(self, df):
        c_fn, c_fp = self.base_.costs(df)
        m_fn = df["attack_cat"].map(self.m_fn_).fillna(1.0).values
        m_fp = self._fp_group(df).map(self.m_fp_).fillna(1.0).values
        return c_fn * m_fn, c_fp * m_fp

    def sample_weights(self, df):
        c_fn, c_fp = self.costs(df)
        w = np.where(df["label"].values == 1, c_fn, c_fp)
        return w / w.mean()

    def total_cost(self, df, y_pred):
        y, p = df["label"].values, np.asarray(y_pred)
        c_fn, c_fp = self.costs(df)
        t = c_fn[(y == 1) & (p == 0)].sum() + c_fp[(y == 0) & (p == 1)].sum()
        return float(t), float(t / len(df))

    # ---- feedback ----
    def _set(self, d, g, factor):
        new = float(np.clip(d[g] * factor, self.m_min, self.m_max))
        changed = abs(new - d[g]) > 1e-9
        d[g] = new
        return int(changed)

    @staticmethod
    def _tgt(target, g, floor=0.005):
        """target: float (same everywhere) or dict {group: value, "__global__": value}."""
        if isinstance(target, dict):
            t = target.get(g, target.get("__global__", floor))
        else:
            t = target
        return max(float(t), floor)

    def update(self, raw, pred, fn_target=0.02, fpr_target=0.05,
               eta=0.35, band=0.2, min_group=30):
        """raw: tuning dataframe, pred: hard predictions of the current model.
        Targets may be per-group dicts (e.g. the baseline's own per-group error rates)."""
        y = raw["label"].values
        pred = np.asarray(pred).astype(int)
        atk, nor = y == 1, y == 0
        g_fnr, g_fpr = (pred[atk] == 0).mean(), (pred[nor] == 1).mean()
        fpr_pressure = g_fpr / self._tgt(fpr_target, "__global__") - 1.0
        fnr_pressure = g_fnr / self._tgt(fn_target, "__global__") - 1.0
        changed, fnr_by, fpr_by = 0, {}, {}

        def factor(ratio, other_pressure):
            if ratio > 1 + band:
                return 1 + eta * min(ratio - 1, 1.0)               # raise
            if ratio < 1 - band and other_pressure > band:
                return 1 / (1 + eta * min(other_pressure, 1.0))    # lower
            return 1.0                                             # stable

        cat = raw["attack_cat"].values
        for g in self.fn_groups_:
            m = atk & (cat == g)
            if m.sum() < min_group:
                continue
            fnr_by[g] = (pred[m] == 0).mean()
            changed += self._set(self.m_fn_, g, factor(fnr_by[g] / self._tgt(fn_target, g), fpr_pressure))

        grp = self._fp_group(raw).values
        for g in self.fp_groups_:
            m = nor & (grp == g)
            if m.sum() < min_group:
                continue
            fpr_by[g] = (pred[m] == 1).mean()
            changed += self._set(self.m_fp_, g, factor(fpr_by[g] / self._tgt(fpr_target, g), fnr_pressure))

        return {"global_FNR": g_fnr, "global_FPR": g_fpr, "changed": changed,
                "fnr_by_cat": fnr_by, "fpr_by_service": fpr_by}