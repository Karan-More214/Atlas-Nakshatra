"""
STEP 4 - TRAIN
Compares simple baselines with logistic regression and LightGBM, using a chronological
split so the test set contains only models newer than anything seen in training.

    train  -> fit models
    valid  -> choose hyper-parameters, calibrate probabilities, pick a decision threshold
    test   -> report final, untouched metrics once

Outputs:
    models/atlas_nakshatra.joblib        calibrated model + metadata (used by predict.py / app)
    reports/metrics.json                 final test metrics
    reports/model_comparison.csv         every model on valid and test
    reports/figures/pr_curve.png, calibration.png

Usage:
    python src/train.py
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from features import CATEGORICAL, FEATURES, NUMERIC, TEXT

MODEL_PATH = C.MODELS_DIR / "atlas_nakshatra.joblib"


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=False), CATEGORICAL),
        ("num", Pipeline(num_steps), NUMERIC),
        ("txt", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=C.TEXT_MAX_FEATURES, ngram_range=(1, 2),
                                      min_df=5, sublinear_tf=True)),
            ("svd", TruncatedSVD(C.TEXT_SVD_COMPONENTS, random_state=C.RANDOM_STATE)),
        ]), TEXT),
    ], verbose_feature_names_out=True)


def lgbm(**params) -> Pipeline:
    base = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=C.RANDOM_STATE, verbose=-1)
    base.update(params)
    return Pipeline([("prep", make_preprocessor(scale=False)), ("clf", LGBMClassifier(**base))])


def checklist_score(X: pd.DataFrame) -> np.ndarray:
    """Baseline a human might use: count of documentation 'good practices'."""
    cols = ["sec_usage", "sec_training_data", "sec_evaluation", "sec_limitations",
            "sec_citation", "has_model_index", "has_safetensors"]
    return X[cols].sum(axis=1).values - 2 * X["card_is_template"].values - 2 * X["name_is_throwaway"].values


def metrics(y, p, threshold=None):
    order = np.argsort(-p)
    top = order[: max(1, int(0.1 * len(p)))]
    out = {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)) if p.min() >= 0 and p.max() <= 1 else float("nan"),
        "precision_top10pct": y[top].mean(),
        "lift_top10pct": y[top].mean() / y.mean(),
        "base_rate": y.mean(),
    }
    if threshold is not None:
        pred = (p >= threshold).astype(int)
        out.update(threshold=threshold, precision=precision_score(y, pred, zero_division=0),
                   recall=recall_score(y, pred), f1=f1_score(y, pred))
    return {k: float(v) for k, v in out.items()}


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])])


def save_figures(y_test, curves, calibrated_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in curves.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, p):.2f})")
    ax.axhline(y_test.mean(), ls="--", c="grey", lw=1, label=f"base rate ({y_test.mean():.0%})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall on the test set (newest models)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "pr_curve.png", dpi=150); plt.close(fig)

    frac, mean_p = calibration_curve(y_test, calibrated_p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.plot(mean_p, frac, marker="o")
    ax.set(xlabel="Predicted probability", ylabel="Observed adoption rate", title="Calibration (test)")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "calibration.png", dpi=150); plt.close(fig)


def main():
    data = pd.read_parquet(C.PROCESSED_DIR / "dataset.parquet")
    parts = {s: data[data["split"] == s] for s in ("train", "valid", "test")}
    X = {s: d[FEATURES] for s, d in parts.items()}
    y = {s: d["is_adopted"].values for s, d in parts.items()}
    print(" | ".join(f"{s}: {len(d):,} models, {d['is_adopted'].mean():.1%} adopted" for s, d in parts.items()))

    results, test_preds = [], {}

    def record(name, p_valid, p_test):
        results.append({"model": name, "split": "valid", **metrics(y["valid"], p_valid)})
        results.append({"model": name, "split": "test", **metrics(y["test"], p_test)})
        test_preds[name] = p_test

    # 1) baselines
    dummy = DummyClassifier(strategy="prior").fit(X["train"], y["train"])
    record("Baseline: base rate", dummy.predict_proba(X["valid"])[:, 1], dummy.predict_proba(X["test"])[:, 1])
    record("Baseline: docs checklist", checklist_score(X["valid"]), checklist_score(X["test"]))

    # 2) logistic regression
    # regularisation strength chosen on validation; stronger C keeps coefficients stable
    # when documentation features overlap (e.g. evaluation section vs. metrics listed)
    logit, logit_ap, logit_c = None, -1, None
    for c in (0.03, 0.1, 0.3, 1.0):
        m = Pipeline([("prep", make_preprocessor(scale=True)),
                      ("clf", LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))])
        m.fit(X["train"], y["train"])
        ap = average_precision_score(y["valid"], m.predict_proba(X["valid"])[:, 1])
        print(f"  Logistic C={c} -> valid PR-AUC {ap:.3f}")
        if ap > logit_ap:
            logit, logit_ap, logit_c = m, ap, c
    record("Logistic regression", logit.predict_proba(X["valid"])[:, 1], logit.predict_proba(X["test"])[:, 1])

    # 3) LightGBM: small grid, chosen on validation PR-AUC only
    grid = [dict(num_leaves=15, min_child_samples=50), dict(num_leaves=31, min_child_samples=30),
            dict(num_leaves=63, min_child_samples=20, n_estimators=600)]
    best, best_ap, best_params = None, -1, None
    for params in grid:
        m = lgbm(**params).fit(X["train"], y["train"])
        ap = average_precision_score(y["valid"], m.predict_proba(X["valid"])[:, 1])
        print(f"  LightGBM {params} -> valid PR-AUC {ap:.3f}")
        if ap > best_ap:
            best, best_ap, best_params = m, ap, params
    record("LightGBM", best.predict_proba(X["valid"])[:, 1], best.predict_proba(X["test"])[:, 1])

    # pick the final model family on VALIDATION PR-AUC (never on test)
    if logit_ap > best_ap:
        best, best_params, chosen = logit, {"C": logit_c, "class_weight": "balanced"}, "Logistic regression"
    else:
        chosen = "LightGBM"
    print(f"  Chosen on validation: {chosen}")

    # 4) calibrate on validation (model frozen), choose threshold on validation, test once
    # Platt scaling: more stable than isotonic on a small validation set
    calibrated = CalibratedClassifierCV(FrozenEstimator(best), method="sigmoid").fit(X["valid"], y["valid"])
    p_valid = calibrated.predict_proba(X["valid"])[:, 1]
    p_test = calibrated.predict_proba(X["test"])[:, 1]
    threshold = best_f1_threshold(y["valid"], p_valid)
    final = metrics(y["test"], p_test, threshold)
    results.append({"model": f"{chosen} (calibrated) - FINAL", "split": "test", **final})

    table = pd.DataFrame(results)
    table.to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n" + table[["model", "split", "pr_auc", "roc_auc", "brier", "precision_top10pct", "lift_top10pct"]]
          .to_string(index=False, float_format="{:.3f}".format))
    print(f"\nFINAL (test, newest {len(y['test']):,} models): PR-AUC {final['pr_auc']:.3f} "
          f"vs base rate {final['base_rate']:.3f} | ROC-AUC {final['roc_auc']:.3f} | "
          f"at threshold {threshold:.2f}: precision {final['precision']:.2f}, recall {final['recall']:.2f}")

    save_figures(y["test"], {"LightGBM": test_preds["LightGBM"],
                             "Logistic regression": test_preds["Logistic regression"],
                             "Docs checklist": test_preds["Baseline: docs checklist"]}, p_test)

    bundle = {
        "model": calibrated,            # use for probabilities
        "base_pipeline": best,          # uncalibrated model, used for SHAP
        "model_family": chosen,
        "background": (best.named_steps["prep"].transform(X["train"].sample(min(300, len(X["train"])),
                       random_state=C.RANDOM_STATE)) if chosen != "LightGBM" else None),
        "features": FEATURES, "threshold": threshold, "lgbm_params": best_params,
        "target": f"downloads_30d >= {C.ADOPTION_MIN_DOWNLOADS_30D}",
        "test_metrics": final, "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_period": [str(parts["train"]["created_at"].min().date()), str(parts["train"]["created_at"].max().date())],
        "demo": C.DEMO,
    }
    joblib.dump(bundle, MODEL_PATH)
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {k: v for k, v in bundle.items() if k not in ("model", "base_pipeline", "background")}, indent=2))
    print(f"Saved model -> {MODEL_PATH}")

    try:  # optional experiment tracking
        import mlflow
        with mlflow.start_run(run_name="atlas-nakshatra"):
            mlflow.log_params({**best_params, "target": bundle["target"], "demo": C.DEMO})
            mlflow.log_metrics({f"test_{k}": v for k, v in final.items()})
            mlflow.log_artifacts(str(C.FIGURES_DIR))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
