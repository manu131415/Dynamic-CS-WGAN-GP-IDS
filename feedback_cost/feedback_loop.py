"""
Reusable pieces for Experiment 3 (used by train_feedback.py AND multi_seed_comparison.py).
Place at: <project>/feedback_cost/feedback_loop.py

Two design fixes vs. the first version
  FIX 1  targets are ACHIEVABLE: by default each group's target error rate is the retrained
         unweighted baseline's own rate on the tuning set ("baseline-parity" targets):
             FN target of attack type k = baseline FNR_k      (floor 0.01)
             FP target of service g     = baseline FPR_g      (floor 0.01)
             global targets             = baseline global FNR / FPR
         The loop therefore stops pushing once the model is no worse than the baseline.
  FIX 2  the best round is chosen by a THRESHOLD-FREE ranking score on the tuning set
         (mean recall at FPR 5/10/20%), not by cost on a flat cost curve.
"""
import copy
import warnings
from types import SimpleNamespace
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from exp_common import *
from feedback_cost_module import FeedbackCost

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # MLP max_iter=30 is intentional


# ---------------- data context ----------------
def make_context(tuning_source):
    S = load_splits(tuning_source)
    pre = joblib.load(BASE_DIR / "preprocessor.pkl")
    sc, dc = load_fixed_costs()
    raw_tr, raw_t, raw_f = S["raw_train"], S["raw_tune"], S["raw_final"]
    cfn_t, cfp_t = dc.costs(raw_t)                       # FIXED evaluation costs
    return SimpleNamespace(
        tag=tuning_source, raw_tr=raw_tr, raw_t=raw_t, raw_f=raw_f,
        X_tr=featurize(pre, raw_tr), X_t=featurize(pre, raw_t), X_f=featurize(pre, raw_f),
        y_tr=raw_tr["label"].values, y_t=raw_t["label"].values, y_f=raw_f["label"].values,
        static_cost=sc, dynamic_cost=dc, cfn_t=cfn_t, cfp_t=cfp_t)


# ---------------- models ----------------
def build_model(name, seed, rf_trees=100):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=rf_trees, random_state=seed, n_jobs=-1)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                         batch_size=256, learning_rate_init=0.001, max_iter=30,
                         early_stopping=True, validation_fraction=0.1, random_state=seed)


def fit_model(ctx, name, seed, w=None, rng=None, rf_trees=100):
    """w=None -> unweighted baseline. MLP has no reliable sample_weight -> weighted resampling."""
    m = build_model(name, seed, rf_trees)
    if w is None:
        m.fit(ctx.X_tr, ctx.y_tr)
    elif name == "mlp":
        idx = rng.choice(len(w), size=len(w), replace=True, p=w / w.sum())
        m.fit(ctx.X_tr[idx], ctx.y_tr[idx])
    else:
        m.fit(ctx.X_tr, ctx.y_tr, sample_weight=w)
    return m


# ---------------- targets (FIX 1) ----------------
def baseline_targets(ctx, cost, p_base, floor=0.01, slack=1.0, min_group=30):
    """Per-group error rates of the unweighted baseline on the tuning set (x slack)."""
    pred = (p_base >= 0.5).astype(int)
    y, raw = ctx.y_t, ctx.raw_t
    atk, nor = y == 1, y == 0
    fn = {"__global__": max(floor, (pred[atk] == 0).mean() * slack)}
    cat = raw["attack_cat"].values
    for k in cost.fn_groups_:
        m = atk & (cat == k)
        if m.sum() >= min_group:
            fn[k] = max(floor, (pred[m] == 0).mean() * slack)
    fp = {"__global__": max(floor, (pred[nor] == 1).mean() * slack)}
    grp = cost._fp_group(raw).values
    for g in cost.fp_groups_:
        m = nor & (grp == g)
        if m.sum() >= min_group:
            fp[g] = max(floor, (pred[m] == 1).mean() * slack)
    return fn, fp


# ---------------- operating points / metrics ----------------
def operating_points(ctx, p_t, fpr_cap=0.10):
    """All thresholds are chosen on the TUNING set."""
    return {"default": 0.5,
            "cost": best_threshold(ctx.y_t, p_t, ctx.cfn_t, ctx.cfp_t),
            "cap": fpr_cap_threshold(ctx.y_t, p_t, fpr_cap)}


def eval_row(ctx, p_f, thr):
    m = evaluate(ctx.y_f, p_f, thr, ctx.raw_f, ctx.static_cost, ctx.dynamic_cost)
    m["ROC-AUC"] = float(roc_auc_score(ctx.y_f, p_f))
    rc = recall_at_fpr(ctx.y_f, p_f)
    m.update({f"Recall@FPR{c}": v for c, v in rc.items()})
    m["MeanRecall@FPR"] = float(np.mean(list(rc.values())))
    fz = (ctx.raw_f["attack_cat"].values == "Fuzzers") & (ctx.y_f == 1)
    m[f"Fuzzers_Recall@FPR{FUZZ_CAP}"] = group_recall_at_fpr(ctx.y_f, p_f, fz)
    return m


# ---------------- the feedback loop ----------------
def run_feedback_loop(ctx, name, seed, rounds=8, eta=0.35, target_mode="baseline",
                      fn_target=0.02, fpr_target=0.05, p_base=None, slack=1.0,
                      patience=3, rf_trees=100, verbose=True):
    """Round 0 = plain dynamic-cost model (all multipliers 1). Returns dict with
    history rows, round0 record, best record (chosen by tuning-set ranking score)."""
    cost = FeedbackCost().fit(ctx.raw_tr)
    rng = np.random.default_rng(seed)
    if target_mode == "baseline":
        assert p_base is not None, "baseline targets need the baseline's tuning-set probabilities"
        fn_t, fp_t = baseline_targets(ctx, cost, p_base, slack=slack)
    else:
        fn_t, fp_t = fn_target, fpr_target
    if verbose:
        if target_mode == "baseline":
            print(f"  targets (baseline parity): FNR_global={fn_t['__global__']:.3f}  "
                  f"FPR_global={fp_t['__global__']:.3f}")
        else:
            print(f"  targets (fixed): FNR={fn_t}  FPR={fp_t}")

    history, best, round0, stale, stop = [], None, None, 0, "max rounds"
    for r in range(rounds + 1):
        used = copy.deepcopy(cost)                               # multipliers that produced this model
        model = fit_model(ctx, name, seed, used.sample_weights(ctx.raw_tr), rng, rf_trees)
        p_t = model.predict_proba(ctx.X_t)[:, 1]
        score = mean_recall_at_fpr(ctx.y_t, p_t)                 # FIX 2
        m05 = evaluate(ctx.y_t, p_t, 0.5, ctx.raw_t, ctx.static_cost, ctx.dynamic_cost)

        info = {"changed": 0, "fnr_by_cat": {}, "fpr_by_service": {}}
        last = r == rounds
        if not last:
            info = cost.update(ctx.raw_t, (p_t >= 0.5).astype(int), fn_t, fp_t, eta)

        row = {"Classifier": name, "seed": seed, "round": r, "score_MeanRecall@FPR": score,
               "FNR@0.5": m05["FNR"], "FPR@0.5": m05["FPR"],
               "DynCost@0.5": m05["Dynamic_Avg_Cost"], "changed": info["changed"]}
        row.update({f"FNR_{k}": v for k, v in info["fnr_by_cat"].items()})
        row.update({f"FPR_{k}": v for k, v in info["fpr_by_service"].items()})
        row.update({f"mfn_{k}": v for k, v in used.m_fn_.items()})
        row.update({f"mfp_{k}": v for k, v in used.m_fp_.items()})
        history.append(row)
        if verbose:
            print(f"  round {r}: score={score:.4f}  FNR@0.5={m05['FNR']:.4f}  "
                  f"FPR@0.5={m05['FPR']:.4f}  updated groups={'-' if last else info['changed']}")

        rec = {"round": r, "model": model, "cost": used, "p_t": p_t, "score": score}
        if r == 0:
            round0 = rec
        if best is None or score > best["score"] + 1e-4:         # ties keep the EARLIER round
            best, stale = rec, 0
        else:
            stale += 1
        if not last and info["changed"] == 0:
            stop = "converged (no multiplier changed)"
            break
        if stale >= patience:
            stop = f"no improvement for {patience} rounds"
            break
    if verbose:
        print(f"  stopped: {stop}; best round = {best['round']} (score {best['score']:.4f})")
    return {"history": history, "round0": round0, "best": best, "stop": stop}