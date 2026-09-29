"""
SYNTHETIC DEMO DATA - for testing the pipeline offline only.

Writes a fake raw snapshot + model cards with the same columns as collect.py and
fetch_cards.py, into data/demo/. The relationships are invented, so NOTHING learned
from this data is a finding about Hugging Face. Use it to check the code runs; use
real data (collect.py) for results.

Usage:
    ATLAS_DEMO=1 python src/make_demo_data.py --n 6000
"""
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

import config as C

if not C.DEMO:
    raise SystemExit("Refusing to write demo data outside demo mode. Set ATLAS_DEMO=1.")

TASKS = {"automatic-speech-recognition": ("transformers", 0.30), "text-generation": ("transformers", 0.18),
         "translation": ("transformers", 0.10), "text-classification": ("transformers", 0.10),
         "text-to-speech": ("transformers", 0.06), "fill-mask": ("transformers", 0.06),
         "sentence-similarity": ("sentence-transformers", 0.06), "token-classification": ("transformers", 0.05),
         None: (None, 0.09)}
LICENSES = {"apache-2.0": 0.40, "mit": 0.20, "cc-by-nc-4.0": 0.08, "llama3.1": 0.07, "gemma": 0.03,
            "cc-by-sa-4.0": 0.02, None: 0.20}
BASES = {"openai/whisper-small": 0.18, "meta-llama/Llama-3.2-1B": 0.10, "google/gemma-2-2b": 0.05,
         "FacebookAI/xlm-roberta-base": 0.10, "facebook/mbart-large-50": 0.05,
         "ai4bharat/indic-bert": 0.04, "Qwen/Qwen2.5-1.5B": 0.05, None: 0.43}
SECTIONS = {
    "usage": "## How to use\n```python\nfrom transformers import pipeline\np = pipeline('task', model='MODEL')\n```\n",
    "data": "## Training data\nFine-tuned on a public corpus of LANG text and speech.\n",
    "eval": "## Evaluation results\nWER on the test split: 18.2. BLEU: 24.1. Accuracy benchmark reported.\n",
    "limits": "## Limitations and bias\nThe model may reflect biases in its training data.\n",
    "cite": "## Citation\n```bibtex\n@misc{model}\n```\n",
}
TEMPLATE = "## Model description\n[More Information Needed]\n## Training procedure\n[More Information Needed]\n"


def pick(rng, dist, size):
    keys = list(dist)
    idx = rng.choice(len(keys), size=size, p=np.array(list(dist.values())) / sum(dist.values()))
    return [keys[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.n

    langs = pd.read_csv(C.CONFIG_DIR / "languages.csv")
    amap = pd.read_csv(C.CONFIG_DIR / "author_mapping.csv")
    lang_w = np.sqrt(langs["speakers_millions"].clip(lower=0.5).to_numpy(dtype=float, copy=True))
    lang_w /= lang_w.sum()

    # authors: a few known orgs + a long tail of individuals, some very prolific
    tail = [f"user{i:04d}" for i in range(1800)]
    tail_w = rng.pareto(1.3, len(tail)) + 1
    known_w = np.full(len(amap), 3.0)
    all_authors = list(amap["author"]) + tail
    aw = np.concatenate([known_w, tail_w]); aw /= aw.sum()
    authors = rng.choice(all_authors, size=n, p=aw)
    known = dict(zip(amap["author"], amap["org_type"]))

    # created dates: 2022-04 .. 2026-09, growing volume
    days = (date(2026, 9, 20) - date(2022, 4, 1)).days
    created = pd.Timestamp("2022-04-01", tz="UTC") + pd.to_timedelta(
        (rng.beta(2.2, 1.0, n) * days).astype(int), unit="D")
    placeholder = rng.random(n) < 0.06
    created = created.where(~placeholder, pd.Timestamp("2022-03-02", tz="UTC"))

    tasks, licenses, bases = pick(rng, {k: v[1] for k, v in TASKS.items()}, n), pick(rng, LICENSES, n), pick(rng, BASES, n)
    quality = rng.normal(0, 1, n)  # hidden "care" of the builder
    raw_rows, cards = [], []
    for i in range(n):
        org = known.get(authors[i], "Individual")
        q = quality[i] + (0.8 if org in ("Global Big Tech", "Indian Research & Academia") else 0)
        throwaway = rng.random() < 0.12 - 0.05 * (q > 0)
        name = ("test-" if throwaway else "") + f"{tasks[i] or 'model'}-{i}".replace("automatic-", "")
        multi = rng.random() < 0.25
        mlangs = list(rng.choice(langs["lang_code"], size=rng.integers(2, 8), replace=False, p=lang_w)) if multi \
            else [rng.choice(langs["lang_code"], p=lang_w)]
        tags = [*mlangs, *(["en"] if rng.random() < 0.4 else [])]
        if licenses[i]:
            tags.append(f"license:{licenses[i]}")
        if bases[i]:
            tags.append(f"base_model:{rng.choice(['finetune', 'finetune', 'adapter', 'quantized'])}:{bases[i]}")
        if tasks[i]:
            tags.append(tasks[i])

        # model card: better builders write more complete cards
        n_sec = int(np.clip(rng.poisson(1.2 + 1.3 * max(q, -0.8)), 0, 5))
        secs = list(rng.choice(list(SECTIONS), size=n_sec, replace=False))
        template = n_sec <= 1 and rng.random() < 0.6
        card = f"---\nlanguage: {mlangs}\n---\n# {name}\nA model for {', '.join(mlangs)}.\n"
        card += TEMPLATE if template else ""
        card += "".join(SECTIONS[s] for s in secs)
        has_readme = rng.random() > 0.05
        files = ["config.json", "README.md" if has_readme else ".gitattributes",
                 "model.safetensors" if rng.random() < 0.55 + 0.1 * q else "pytorch_model.bin"]
        if rng.random() < 0.6:
            files += ["tokenizer.json", "tokenizer_config.json"]
        if rng.random() < 0.06:
            files.append("model-q4.gguf")

        # hidden adoption process (invented for testing)
        logit = (-1.3 + 0.55 * q + 0.35 * ("eval" in secs) + 0.3 * ("usage" in secs)
                 - 0.9 * template - 1.1 * throwaway + 0.5 * (licenses[i] in ("apache-2.0", "mit"))
                 - 0.4 * (licenses[i] is None) + 0.6 * (bases[i] is not None and "whisper" in bases[i])
                 + 0.8 * (org in ("Global Big Tech", "Indian Research & Academia", "Indian Startup & Company"))
                 + 0.3 * multi + rng.normal(0, 0.9))
        lam = np.exp(logit * 1.6 + 1.2)
        d30 = int(rng.poisson(lam) if rng.random() < 1 / (1 + np.exp(-logit)) + 0.1 else rng.poisson(0.5))
        age = max((pd.Timestamp("2026-09-27", tz="UTC") - created[i]).days, 1)
        d_all = int(d30 * age / 30 * rng.uniform(0.5, 2.5)) + rng.integers(0, 30)
        mid = f"{authors[i]}/{name}"
        cd = {"language": mlangs, "license": licenses[i],
              "datasets": ["mozilla-foundation/common_voice_17_0"] if "data" in secs else [],
              "metrics": ["wer"] if "eval" in secs else [], "base_model": bases[i]}
        for lang in mlangs:
            raw_rows.append({
                "model_id": mid, "author": authors[i], "query_lang": lang, "hf_code": lang,
                "pipeline_tag": tasks[i], "library_name": TASKS[tasks[i]][0],
                "downloads_30d": d30, "downloads_all_time": d_all,
                "likes": int(rng.poisson(max(d30, 0) ** 0.3)), "gated": "manual" if rng.random() < 0.03 else "False",
                "created_at": created[i], "last_modified": created[i],
                "tags": "|".join(tags), "card_data": json.dumps(cd),
                "has_model_index": int("eval" in secs and rng.random() < 0.7),
                "n_params": float(rng.choice([2.4e8, 1.1e8, 7.6e8, 1.2e9, 2.6e9])) if rng.random() < 0.7 else None,
                "used_storage": float(rng.lognormal(20, 1.5)), "siblings": "|".join(files),
            })
        if has_readme:
            cards.append({"model_id": mid, "status": 200, "card_text": card})

    df = pd.DataFrame(raw_rows)
    df["snapshot_date"] = "2026-09-27"
    df.to_parquet(C.RAW_DIR / "models_raw_2026-09-27.parquet", index=False)
    with (C.CARDS_DIR / "cards.jsonl").open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")
    print(f"DEMO: wrote {df['model_id'].nunique():,} synthetic models -> {C.RAW_DIR}")


if __name__ == "__main__":
    main()
