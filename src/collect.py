"""
STEP 1 - COLLECT
Pulls metadata for every Hugging Face model tagged with one of India's 22 official
languages. Compared with Bharat AI Atlas, it also collects what a builder controls at
launch: card metadata, evaluation results, file list, parameter count and repo size.

Output: data/raw/models_raw_<date>.parquet (one row per model x matched language code)

Usage:
    python src/collect.py                 # all 22 languages
    python src/collect.py --langs mr hi   # only some languages
    python src/collect.py --limit 50      # quick test: 50 models per language code
"""
import argparse
import json
import sys
import time
from datetime import date

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import CONFIG_DIR, HF_TOKEN, RAW_DIR

EXPAND = [
    "author", "createdAt", "lastModified", "downloads", "downloadsAllTime", "likes",
    "pipeline_tag", "library_name", "tags", "gated", "cardData", "safetensors",
    "siblings", "model-index", "baseModels",
]


def _json(value):
    """Store nested API objects as JSON text so they fit in a flat table."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return json.dumps(str(value))


def model_row(m, lang_code, hf_code):
    params = None
    if getattr(m, "safetensors", None) is not None:
        params = getattr(m.safetensors, "total", None)
    siblings = [s.rfilename for s in (getattr(m, "siblings", None) or [])]
    return {
        "model_id": m.id,
        "author": m.author or m.id.split("/")[0],
        "query_lang": lang_code,
        "hf_code": hf_code,
        "pipeline_tag": m.pipeline_tag,
        "library_name": m.library_name,
        "downloads_30d": m.downloads,
        "downloads_all_time": getattr(m, "downloads_all_time", None),
        "likes": m.likes,
        "gated": str(getattr(m, "gated", None)),  # False | "auto" | "manual"
        "created_at": m.created_at,
        "last_modified": getattr(m, "last_modified", None),
        "tags": "|".join(m.tags or []),
        "card_data": _json(getattr(m, "card_data", None)),
        "has_model_index": int(bool(getattr(m, "model_index", None))),
        "n_params": params,
        "used_storage": getattr(m, "used_storage", None),
        "siblings": "|".join(siblings),
    }


def fetch_language(api, lang_code, hf_code, limit=None, retries=3):
    # A single model with malformed card metadata can make the Hub API raise mid-listing
    # (e.g. an invalid model-index breaks huggingface_hub's own parsing). Retrying from
    # scratch hits the same record again, so keep whatever was fetched before the failure
    # instead of throwing an entire language's models away.
    best = []
    for attempt in range(1, retries + 1):
        rows = []
        try:
            for m in api.list_models(filter=hf_code, expand=EXPAND, limit=limit):
                rows.append(model_row(m, lang_code, hf_code))
            return rows
        except Exception as err:  # network hiccup / rate limit / bad record from the Hub
            wait = 15 * attempt
            print(f"   ! {hf_code}: {err.__class__.__name__} after {len(rows)} models - retrying in {wait}s")
            best = max(best, rows, key=len)
            time.sleep(wait)
    print(f"   x {hf_code}: failed after {retries} attempts, kept {len(best)} models fetched before the failure")
    return best


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", help="language codes to collect (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="max models per language code")
    args = parser.parse_args()

    langs = pd.read_csv(CONFIG_DIR / "languages.csv")
    if args.langs:
        langs = langs[langs["lang_code"].isin(args.langs)]

    api = HfApi(token=HF_TOKEN)
    rows = []
    for _, lang in langs.iterrows():
        for hf_code in str(lang["hf_codes"]).split("|"):
            got = fetch_language(api, lang["lang_code"], hf_code, args.limit)
            print(f"{lang['lang_name']:<10} [{hf_code:<4}] {len(got):>6} models")
            rows.extend(got)
            time.sleep(0.5)  # be polite to the API

    df = pd.DataFrame(rows)
    df["snapshot_date"] = date.today().isoformat()
    for col in ("created_at", "last_modified"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    out = RAW_DIR / f"models_raw_{date.today().isoformat()}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df):,} rows ({df['model_id'].nunique():,} unique models) -> {out}")


if __name__ == "__main__":
    main()
