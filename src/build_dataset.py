"""
STEP 3 - BUILD THE MODELLING DATASET
Raw snapshot + model cards -> one row per model with features, target and split.

    target  is_adopted = downloads in last 30 days >= ADOPTION_MIN_DOWNLOADS_30D
    rows    models with a real created date, aged MIN_AGE_DAYS..MAX_AGE_DAYS at the snapshot
    split   chronological by created date (train = oldest, test = newest)

Output: data/processed/dataset.parquet

Usage:
    python src/build_dataset.py
    python src/build_dataset.py --file data/raw/models_raw_2026-09-27.parquet
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from features import build_features, FEATURES

PLACEHOLDER_DATE = date(2022, 3, 2)  # Hugging Face's createdAt for all pre-March-2022 models


def latest_raw_file() -> Path:
    files = sorted(C.RAW_DIR.glob("models_raw_*.parquet"))
    if not files:
        raise SystemExit("No raw snapshot found. Run: python src/collect.py  (or --demo)")
    return files[-1]


def load_cards() -> pd.DataFrame:
    path = C.CARDS_DIR / "cards.jsonl"
    if not path.exists():
        print("  ! no model cards found - card features will be empty (run src/fetch_cards.py)")
        return pd.DataFrame(columns=["model_id", "card_text"])
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    cards = pd.DataFrame(rows)[["model_id", "card_text"]]
    return cards.drop_duplicates("model_id", keep="last")


def prepare_models(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per model, with language info and author history attached."""
    raw = raw.dropna(subset=["model_id"]).copy()
    raw["author"] = raw["author"].fillna(raw["model_id"].str.split("/").str[0]).str.strip()
    for col in ("downloads_30d", "downloads_all_time", "likes"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype("int64")
    raw["created_at"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)

    langs = raw[["model_id", "query_lang"]].drop_duplicates()
    per_model = langs.groupby("model_id")["query_lang"].agg(["nunique", "first"])
    per_model.columns = ["n_indic_languages", "only_lang"]
    per_model["primary_lang"] = np.where(per_model["n_indic_languages"] == 1,
                                         per_model["only_lang"], "multi")

    models = raw.drop_duplicates("model_id").merge(
        per_model[["n_indic_languages", "primary_lang"]], left_on="model_id", right_index=True)

    models["is_placeholder_date"] = (models["created_at"].dt.date == PLACEHOLDER_DATE).astype(int)
    # author experience: how many models the author had published BEFORE this one
    models = models.sort_values(["created_at", "model_id"])
    models["author_prior_models"] = models.groupby("author").cumcount()
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None)
    args = parser.parse_args()

    raw_path = args.file or latest_raw_file()
    raw = pd.read_parquet(raw_path)
    snapshot = pd.Timestamp(raw["snapshot_date"].iloc[0], tz="UTC")
    print(f"Loaded {len(raw):,} rows from {raw_path.name} (snapshot {snapshot.date()})")

    models = prepare_models(raw)
    models = models.merge(load_cards(), on="model_id", how="left")
    models["card_text"] = models["card_text"].fillna("")

    models["age_days"] = (snapshot - models["created_at"]).dt.days
    models["is_adopted"] = (models["downloads_30d"] >= C.ADOPTION_MIN_DOWNLOADS_30D).astype(int)
    eligible = ((models["is_placeholder_date"] == 0)
                & models["age_days"].between(C.MIN_AGE_DAYS, C.MAX_AGE_DAYS))
    models = models[eligible].copy()

    author_map = pd.read_csv(C.CONFIG_DIR / "author_mapping.csv")
    feats = build_features(models, author_map)

    meta_cols = ["model_id", "author", "created_at", "age_days", "downloads_30d",
                 "downloads_all_time", "likes", "is_adopted"]
    data = models[meta_cols].merge(feats, on="model_id")

    # chronological split: no model in validation/test was created before a training model
    data = data.sort_values("created_at").reset_index(drop=True)
    n = len(data)
    cut1, cut2 = int(n * C.TRAIN_FRAC), int(n * (C.TRAIN_FRAC + C.VALID_FRAC))
    data["split"] = np.select([data.index < cut1, data.index < cut2], ["train", "valid"], "test")

    assert data["model_id"].is_unique
    assert not (set(FEATURES) & C.LEAKY_COLUMNS), "a leaky column is being used as a feature"

    out = C.PROCESSED_DIR / "dataset.parquet"
    data.to_parquet(out, index=False)

    print(f"\nEligible models: {n:,} (aged {C.MIN_AGE_DAYS}-{C.MAX_AGE_DAYS} days, real created date)")
    print(f"Target: >= {C.ADOPTION_MIN_DOWNLOADS_30D} downloads in last 30 days\n")
    summary = data.groupby("split").agg(
        models=("model_id", "count"),
        adoption_rate=("is_adopted", "mean"),
        created_from=("created_at", "min"),
        created_to=("created_at", "max"))
    summary["created_from"] = summary["created_from"].dt.date
    summary["created_to"] = summary["created_to"].dt.date
    print(summary.loc[["train", "valid", "test"]].to_string(float_format="{:.1%}".format))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
