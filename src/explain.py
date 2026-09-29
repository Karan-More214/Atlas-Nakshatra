"""
STEP 5 - EXPLAIN
Global explanations of the trained model on the test set (the newest models):

    reports/shap_importance.csv          mean |SHAP| per original feature + direction
    reports/figures/shap_importance.png  top features
    reports/adoption_by_group.csv        observed adoption rate by task, language, builder, license
    reports/findings_draft.md            auto-generated bullet points to start your write-up

SHAP shows association inside the model, not cause and effect. Use it to form hypotheses
("well-documented models are adopted more"), not to claim that editing a card causes downloads.

Usage:
    python src/explain.py
"""
import numpy as np
import pandas as pd

import config as C
from features import CATEGORICAL, FEATURES
from predict import LABELS, load_bundle, shap_by_feature


def direction(values: pd.Series, shap_col: pd.Series) -> str:
    """Describe how a feature pushes predictions, for numeric/binary features."""
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() < 10 or v.nunique() < 2:
        return ""
    corr = np.corrcoef(v.fillna(v.median()), shap_col)[0, 1]
    if np.isnan(corr) or abs(corr) < 0.2:
        return "mixed"
    return "higher -> more adoption" if corr > 0 else "higher -> less adoption"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = load_bundle(str(C.MODELS_DIR / "atlas_nakshatra.joblib"))
    data = pd.read_parquet(C.PROCESSED_DIR / "dataset.parquet")
    test = data[data["split"] == "test"].reset_index(drop=True)

    sv = shap_by_feature(bundle, test[FEATURES])
    imp = pd.DataFrame({
        "feature": sv.columns,
        "label": [LABELS.get(f, f) for f in sv.columns],
        "mean_abs_shap": sv.abs().mean().values,
        "direction": [("see categories" if f in CATEGORICAL else
                       "text topics" if f == "card_text" else direction(test[f], sv[f])) for f in sv.columns],
    }).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(C.REPORTS_DIR / "shap_importance.csv", index=False)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["label"], top["mean_abs_shap"], color="#3b6fb6")
    ax.set(xlabel="Mean |SHAP| (log-odds)", title=f"What drives adoption? ({bundle['model_family']}, test set)")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "shap_importance.png", dpi=150); plt.close(fig)

    # descriptive context for the write-up (all eligible models)
    groups = []
    for col in ["task_group", "primary_lang", "org_type", "license_group", "base_author"]:
        g = (data.groupby(col)["is_adopted"].agg(models="count", adoption_rate="mean")
             .query("models >= 30").reset_index().rename(columns={col: "value"}))
        g.insert(0, "group", col)
        groups.append(g.sort_values("adoption_rate", ascending=False))
    by_group = pd.concat(groups)
    by_group.to_csv(C.REPORTS_DIR / "adoption_by_group.csv", index=False)

    n_sections = data[["sec_usage", "sec_training_data", "sec_evaluation",
                       "sec_limitations", "sec_citation"]].sum(axis=1)
    docs = data.groupby(n_sections)["is_adopted"].agg(models="count", adoption_rate="mean")

    m = bundle["test_metrics"]
    lines = [
        "# Findings draft (auto-generated - edit before publishing)",
        "",
        "> Trained on SYNTHETIC demo data: these numbers are placeholders, not findings." if bundle["demo"] else "",
        f"- Target: {bundle['target']}. Base rate on the test set: {m['base_rate']:.1%}.",
        f"- Final model ({bundle['model_family']}, calibrated) on the newest {len(test):,} models: "
        f"PR-AUC {m['pr_auc']:.2f} (vs {m['base_rate']:.2f} for guessing), ROC-AUC {m['roc_auc']:.2f}.",
        f"- The top 10% of models by predicted probability are adopted {m['lift_top10pct']:.1f}x more "
        f"often than average ({m['precision_top10pct']:.0%} vs {m['base_rate']:.0%}).",
        "- Strongest drivers: " + ", ".join(imp.head(5)["label"]) + ".",
        "- Adoption rate by number of documentation sections (0-5): " +
        ", ".join(f"{int(k)}: {v:.0%}" for k, v in docs["adoption_rate"].items()) + ".",
    ]
    (C.REPORTS_DIR / "findings_draft.md").write_text("\n".join(l for l in lines if l is not None) + "\n")

    print(imp.head(12)[["label", "mean_abs_shap", "direction"]].to_string(index=False, float_format="{:.3f}".format))
    print("\n" + "\n".join(lines[3:]))
    print(f"\nSaved reports -> {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
