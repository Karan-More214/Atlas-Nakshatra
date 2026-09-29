"""
STEP 6 - PREDICT ("Launch Checker" engine)
Scores one model: adoption probability, the SHAP drivers behind it, and which launch
changes would raise it the most. Used by the CLI below and by app/streamlit_app.py.

Usage:
    python src/predict.py --model-id ai4bharat/indic-bert      # live, from Hugging Face
    python src/predict.py --from-test 5                         # 5 examples from the test set
"""
import argparse
import json
import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import config as C
from features import CATEGORICAL, FEATURES, NUMERIC, build_features

MODEL_PATH = C.MODELS_DIR / "atlas_nakshatra.joblib"

LABELS = {
    "task_group": "Task", "library": "Library", "license_group": "License type",
    "base_author": "Base model provider", "base_relation": "How it builds on a base model",
    "org_type": "Builder type", "primary_lang": "Language", "is_derivative": "Built on another model",
    "is_gated": "Gated access", "is_indian_builder": "Indian builder",
    "n_indic_languages": "No. of Indian languages", "is_india_dedicated": "Dedicated to one Indian language",
    "n_language_tags": "No. of language tags", "log_params": "Model size (parameters)",
    "log_storage_mb": "Repo size", "n_files": "No. of files", "has_safetensors": "Safetensors weights",
    "has_gguf": "GGUF (local-run) weights", "has_onnx": "ONNX weights", "has_tokenizer": "Tokenizer files",
    "has_model_index": "Structured eval results (model-index)", "card_n_datasets": "Datasets listed in card",
    "card_has_metrics": "Metrics listed in card", "card_has_base_model": "Base model declared in card",
    "has_card": "Has a model card", "card_log_words": "Model card length", "card_n_headings": "Card headings",
    "card_n_code_blocks": "Code examples in card", "card_n_links": "Links in card",
    "card_is_template": "Unedited template text in card", "sec_usage": "Usage section",
    "sec_training_data": "Training-data section", "sec_evaluation": "Evaluation section",
    "sec_limitations": "Limitations/bias section", "sec_citation": "Citation section",
    "name_is_throwaway": "Test/demo-style name", "name_len": "Name length",
    "log_author_prior_models": "Builder's earlier models", "card_text": "Model card wording (topics)",
}

# Changes a builder can make before/after launch: (label, features to set, text to add to the card)
ACTIONS = [
    ("Add a 'How to use' section with a code example", {"sec_usage": 1, "card_n_code_blocks": "+1"},
     "\n## How to use\n```python\n```\n"),
    ("Report evaluation results (and a model-index block)", {"sec_evaluation": 1, "has_model_index": 1,
                                                              "card_has_metrics": 1}, "\n## Evaluation results\n"),
    ("Describe the training data and list datasets", {"sec_training_data": 1, "card_n_datasets": "max1"},
     "\n## Training data\n"),
    ("Add a limitations & bias section", {"sec_limitations": 1}, "\n## Limitations and bias\n"),
    ("Add a citation", {"sec_citation": 1}, "\n## Citation\n"),
    ("Replace the '[More Information Needed]' template text", {"card_is_template": 0}, ""),
    ("Upload weights as safetensors", {"has_safetensors": 1}, ""),
    ("Use a permissive license (e.g. Apache-2.0/MIT)", {"license_group": "Permissive"}, ""),
    ("Give it a descriptive name (not test/demo)", {"name_is_throwaway": 0}, ""),
]


# ------------------------------------------------------------------ model + SHAP
@lru_cache(maxsize=1)
def load_bundle(path=str(MODEL_PATH)):
    return joblib.load(path)


def _parent(name: str) -> str:
    kind, _, rest = name.partition("__")
    if kind == "txt":
        return "card_text"
    if kind == "num":
        return rest
    return max((c for c in CATEGORICAL if rest.startswith(c + "_")), key=len, default=rest)


def shap_by_feature(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (log-odds) summed back to the original features: rows = models."""
    import shap

    pipe = bundle["base_pipeline"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xt = prep.transform(X[FEATURES])
    names = prep.get_feature_names_out()
    if hasattr(clf, "booster_"):
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
    else:
        bg = bundle.get("background")
        sv = shap.LinearExplainer(clf, bg if bg is not None else Xt).shap_values(Xt)
    frame = pd.DataFrame(np.asarray(sv), columns=names)
    return frame.T.groupby(_parent).sum().T  # one column per original feature


def predict_proba(bundle, X: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(X[FEATURES])[:, 1]


def _apply(row: pd.DataFrame, changes: dict, add_text: str) -> pd.DataFrame:
    new = row.copy()
    for col, val in changes.items():
        if val == "+1":
            new[col] = new[col] + 1
        elif val == "max1":
            new[col] = np.maximum(new[col], 1)
        else:
            new[col] = val
    if add_text:
        new["card_text"] = new["card_text"] + add_text
    return new


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (float, np.floating)) else v


def explain_one(bundle, row: pd.DataFrame, top_n=8):
    """row: one-row frame with FEATURES. Returns probability, drivers and suggestions."""
    p = float(predict_proba(bundle, row)[0])
    sv = shap_by_feature(bundle, row).iloc[0]
    order = sv.abs().sort_values(ascending=False).index[:top_n]
    drivers = pd.DataFrame({
        "feature": order,
        "label": [LABELS.get(f, f) for f in order],
        "value": [_fmt(row.iloc[0][f]) if f != "card_text" else "..." for f in order],
        "impact": sv[order].values,
    })

    suggestions = []
    for label, changes, text in ACTIONS:
        fixed = {c: v for c, v in changes.items() if v not in ("+1", "max1")}
        if fixed and all(str(row.iloc[0][c]) == str(v) for c, v in fixed.items()):
            continue  # already done
        new_p = float(predict_proba(bundle, _apply(row, changes, text))[0])
        if new_p - p > 0.005:
            suggestions.append({"action": label, "new_probability": new_p, "change": new_p - p})
    sugg = pd.DataFrame(suggestions, columns=["action", "new_probability", "change"])
    return p, drivers, sugg.sort_values("change", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ inputs
@lru_cache(maxsize=1)
def _language_codes():
    langs = pd.read_csv(C.CONFIG_DIR / "languages.csv")
    return {code: row.lang_code for row in langs.itertuples() for code in str(row.hf_codes).split("|")}


@lru_cache(maxsize=1)
def _author_map():
    return pd.read_csv(C.CONFIG_DIR / "author_mapping.csv")


@lru_cache(maxsize=1)
def _author_model_counts() -> pd.DataFrame:
    """(author, created_at) for every model in the training snapshot, one row per model.
    Prefers the full raw snapshot (data/raw) when present; falls back to the small lookup
    checked into config/ so this works without the (git-ignored, multi-hundred-MB) raw data,
    e.g. on Streamlit Cloud."""
    files = sorted(C.RAW_DIR.glob("models_raw_*.parquet"))
    if files:
        raw = pd.read_parquet(files[-1], columns=["model_id", "author", "created_at"]).drop_duplicates("model_id")
        raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True)
        return raw[["author", "created_at"]]
    path = C.CONFIG_DIR / "author_model_counts.csv"
    if not path.exists():
        return pd.DataFrame(columns=["author", "created_at"])
    counts = pd.read_csv(path)
    counts["created_at"] = pd.to_datetime(counts["created_at"], utc=True)
    return counts


def _author_prior_models(author, created_at):
    df = _author_model_counts()
    if df.empty:
        return 0
    return int(((df["author"] == author) & (df["created_at"] < created_at)).sum())


def features_from_raw(raw_row: dict) -> pd.DataFrame:
    """raw_row: the fields collect.py stores for one model, plus card_text."""
    tags = [t for t in (raw_row.get("tags") or "").split("|") if t]
    codes = _language_codes()
    langs = sorted({codes[t] for t in tags if t in codes})
    raw_row = {**raw_row, "n_indic_languages": max(len(langs), 1),
               "primary_lang": langs[0] if len(langs) == 1 else "multi"}
    return build_features(pd.DataFrame([raw_row]), _author_map())


def fetch_live(model_id: str) -> pd.DataFrame:
    """Download one model's metadata + card from Hugging Face and build its features."""
    import requests
    from huggingface_hub import HfApi
    from collect import EXPAND, model_row

    m = HfApi(token=C.HF_TOKEN).model_info(model_id, expand=EXPAND)
    row = model_row(m, None, None)
    headers = {"Authorization": f"Bearer {C.HF_TOKEN}"} if C.HF_TOKEN else {}
    r = requests.get(f"https://huggingface.co/{model_id}/raw/main/README.md", headers=headers, timeout=20)
    row["card_text"] = r.text if r.status_code == 200 else ""
    created = pd.Timestamp(row["created_at"]) if row["created_at"] else pd.Timestamp.now(tz="UTC")
    row["author_prior_models"] = _author_prior_models(row["author"], created)
    return features_from_raw(row)


def features_from_draft(d: dict) -> pd.DataFrame:
    """Build features for a model that is not published yet (the app's planning form)."""
    tags = [d["lang"]] + (["en"] if d.get("english") else [])
    if d.get("license"):
        tags.append(f"license:{d['license']}")
    if d.get("base_model"):
        tags.append(f"base_model:finetune:{d['base_model']}")
    sections = {"usage": "## How to use\n```python\n```\n", "training_data": "## Training data\n",
                "evaluation": "## Evaluation results\n", "limitations": "## Limitations and bias\n",
                "citation": "## Citation\n"}
    card = f"# {d['name']}\n{d.get('description', '')}\n" + "".join(
        text for key, text in sections.items() if key in d.get("sections", []))
    files = ["README.md", "config.json", "model.safetensors" if d.get("safetensors") else "pytorch_model.bin"]
    if d.get("tokenizer", True):
        files.append("tokenizer.json")
    raw = {
        "model_id": f"{d['author']}/{d['name']}", "author": d["author"], "pipeline_tag": d["task"],
        "library_name": d.get("library", "transformers"), "gated": "False", "tags": "|".join(tags),
        "card_data": json.dumps({"datasets": ["x"] if "training_data" in d.get("sections", []) else [],
                                 "metrics": ["m"] if "evaluation" in d.get("sections", []) else [],
                                 "base_model": d.get("base_model")}),
        "has_model_index": int("evaluation" in d.get("sections", [])),
        "n_params": d.get("n_params"), "used_storage": None, "siblings": "|".join(files),
        "card_text": card, "author_prior_models": d.get("author_prior_models", 0),
    }
    return features_from_raw(raw)


# ------------------------------------------------------------------ CLI
def print_report(model_id, p, drivers, sugg, bundle):
    verdict = "likely to be adopted" if p >= bundle["threshold"] else "at risk of going unused"
    print(f"\n{model_id}\n  Adoption probability: {p:.0%}  ({verdict})")
    print("  Top drivers (+ raises, - lowers):")
    for r in drivers.itertuples():
        print(f"    {'+' if r.impact > 0 else '-'} {r.label:<42} {str(r.value)[:24]:<24} {r.impact:+.2f}")
    if len(sugg):
        print("  What would help most:")
        for r in sugg.head(4).itertuples():
            print(f"    -> {r.action:<55} {r.new_probability:.0%} ({r.change:+.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id")
    parser.add_argument("--from-test", type=int, default=0)
    args = parser.parse_args()
    bundle = load_bundle()
    if bundle.get("demo"):
        print("NOTE: this model was trained on SYNTHETIC demo data - scores are not real findings.")

    if args.model_id:
        row = fetch_live(args.model_id)
        print_report(args.model_id, *explain_one(bundle, row), bundle)
    if args.from_test:
        path = C.PROCESSED_DIR / "dataset.parquet"
        if not path.exists():
            raise SystemExit(f"--from-test needs {path}, which isn't built here. "
                             "Run src/build_dataset.py first, or use --model-id instead.")
        data = pd.read_parquet(path)
        sample = data[data["split"] == "test"].sample(args.from_test, random_state=1)
        for _, r in sample.iterrows():
            p, drv, sug = explain_one(bundle, r.to_frame().T.infer_objects())
            print_report(f"{r['model_id']}  (actual: {'adopted' if r['is_adopted'] else 'not adopted'})",
                         p, drv, sug, bundle)


if __name__ == "__main__":
    main()
