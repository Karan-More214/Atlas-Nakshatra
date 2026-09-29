"""
STEP 2 - FETCH MODEL CARDS
Downloads each model's README.md (its "model card") for the NLP features.
Cards are cached in data/raw/cards/cards.jsonl, so the script can be stopped and resumed.

Usage:
    python src/fetch_cards.py                 # all models in the newest raw snapshot
    python src/fetch_cards.py --max 500       # quick test
    python src/fetch_cards.py --workers 8
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from config import CARDS_DIR, HF_TOKEN
from build_dataset import latest_raw_file

CARDS_FILE = CARDS_DIR / "cards.jsonl"
MAX_CHARS = 20_000  # long cards are truncated; the start carries most of the signal


def already_fetched():
    if not CARDS_FILE.exists():
        return set()
    with CARDS_FILE.open(encoding="utf-8") as f:
        return {json.loads(line)["model_id"] for line in f if line.strip()}


def fetch_card(session, model_id):
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        r = session.get(url, timeout=20)
        text = r.text[:MAX_CHARS] if r.status_code == 200 else ""
        return {"model_id": model_id, "status": r.status_code, "card_text": text}
    except requests.RequestException as err:
        return {"model_id": model_id, "status": -1, "card_text": "", "error": str(err)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    raw = pd.read_parquet(latest_raw_file(), columns=["model_id", "siblings"])
    raw = raw.drop_duplicates("model_id")
    # only models that actually have a README.md
    has_readme = raw["siblings"].fillna("").str.split("|").apply(lambda s: "README.md" in s)
    todo = [m for m in raw.loc[has_readme, "model_id"] if m not in already_fetched()]
    if args.max:
        todo = todo[: args.max]
    print(f"{len(todo):,} cards to fetch")

    session = requests.Session()
    if HF_TOKEN:
        session.headers["Authorization"] = f"Bearer {HF_TOKEN}"
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(args.workers) as pool, CARDS_FILE.open("a", encoding="utf-8") as out:
        for fut in as_completed(pool.submit(fetch_card, session, m) for m in todo):
            with lock:
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                done += 1
                if done % 500 == 0:
                    out.flush()
                    print(f"  {done:,}/{len(todo):,}")
    print(f"Done. Cards cached in {CARDS_FILE}")


if __name__ == "__main__":
    main()
