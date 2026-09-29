"""
Feature engineering, shared by training (build_dataset.py) and live scoring (predict.py).

Rule: every feature must be something the builder controls or knows AT LAUNCH.
Downloads, likes, trending score, Spaces and child models are outcomes, never inputs
(see config.LEAKY_COLUMNS and tests/test_features.py).
"""
import json
import math
import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ groupings
# (same rules as Bharat AI Atlas src/clean.py, so both projects agree)
TASK_GROUPS = {
    "translation": "Translation",
    "text2text-generation": "Text-to-Text",
    "summarization": "Text-to-Text",
    "automatic-speech-recognition": "Speech Recognition",
    "text-to-speech": "Text-to-Speech",
    "text-to-audio": "Text-to-Speech",
    "audio-classification": "Audio Classification",
    "text-generation": "Text Generation (LLMs)",
    "fill-mask": "Language Understanding (Fill-Mask)",
    "text-classification": "Classification & NER",
    "token-classification": "Classification & NER",
    "zero-shot-classification": "Classification & NER",
    "question-answering": "Question Answering",
    "sentence-similarity": "Embeddings & Search",
    "feature-extraction": "Embeddings & Search",
    "image-to-text": "Vision & Multimodal",
    "image-text-to-text": "Vision & Multimodal",
    "visual-question-answering": "Vision & Multimodal",
    "text-to-image": "Vision & Multimodal",
    "image-classification": "Vision & Multimodal",
}
LICENSE_GROUPS = {
    "Permissive": ["apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0",
                   "cc-by-3.0", "cc-by-2.0", "cc0-1.0", "unlicense", "afl-3.0", "isc",
                   "artistic-2.0", "ecl-2.0", "odc-by", "pddl"],
    "Copyleft / Share-alike": ["gpl", "gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl", "lgpl-2.1",
                               "lgpl-3.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "mpl-2.0", "odbl"],
    "Non-commercial": ["cc-by-nc-2.0", "cc-by-nc-3.0", "cc-by-nc-4.0", "cc-by-nc-sa-2.0",
                       "cc-by-nc-sa-3.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-3.0", "cc-by-nc-nd-4.0",
                       "cc-by-nd-4.0"],
}
BASE_RELATIONS = {"finetune", "adapter", "quantized", "merge"}
NOT_LANGUAGE_TAGS = {"mlx", "tts", "jax", "asr", "nlp", "llm", "ner", "iso", "art", "moe", "sft",
                     "rag", "pii", "trl", "dpo", "gpt", "vit", "ocr", "ctc", "awq", "tf", "kto",
                     "qat", "sql", "api", "cpu", "gpu"}
INDIAN_LANGUAGE_TAGS = {
    "hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "or", "ory", "ml", "pa", "as", "mai", "sat",
    "ks", "ne", "npi", "sd", "doi", "kok", "gom", "mni", "brx", "sa",
    "hin", "ben", "mar", "tel", "tam", "guj", "urd", "kan", "ori", "mal", "pan", "asm", "nep",
    "san", "snd", "kas",
    "bho", "awa", "mag", "hne", "tcy", "lus", "kha", "gbm", "anp", "raj", "bgc", "sck", "gon",
    "kru", "unr", "hoc", "bpy",
}

# Model-card sections that good documentation usually has
CARD_SECTIONS = {
    "sec_usage": r"(how to use|usage|quick ?start|inference|getting started|example)",
    "sec_training_data": r"(training data|dataset|data used|corpus)",
    "sec_evaluation": r"(evaluation|results|benchmark|metrics|performance|wer|bleu|accuracy)",
    "sec_limitations": r"(limitation|bias|risks|intended use|out-of-scope|caveat)",
    "sec_citation": r"(citation|bibtex|cite)",
}
TEMPLATE_PHRASE = "[more information needed]"  # left behind in unedited auto-generated cards
THROWAWAY_NAME = re.compile(r"(?:test|demo|dummy|tmp|temp|trial|sample|debug|practice|my[-_]?model)", re.I)

CATEGORICAL = ["task_group", "library", "license_group", "base_author", "base_relation",
               "org_type", "primary_lang"]
NUMERIC = [
    "is_derivative", "is_gated", "is_indian_builder", "n_indic_languages", "is_india_dedicated",
    "n_language_tags", "log_params", "log_storage_mb", "n_files", "has_safetensors", "has_gguf",
    "has_onnx", "has_tokenizer", "has_model_index", "card_n_datasets", "card_has_metrics",
    "card_has_base_model", "has_card", "card_log_words", "card_n_headings", "card_n_code_blocks",
    "card_n_links", "card_is_template", *CARD_SECTIONS.keys(), "name_is_throwaway",
    "name_len", "log_author_prior_models",
]
TEXT = "card_text"
FEATURES = CATEGORICAL + NUMERIC + [TEXT]


# ------------------------------------------------------------------ helpers
def indian_tag_share(tags):
    langs = {t for t in tags if re.fullmatch(r"[a-z]{2,3}", t) and t not in NOT_LANGUAGE_TAGS}
    langs -= {"en", "eng"}
    return len(langs & INDIAN_LANGUAGE_TAGS) / len(langs) if langs else 0.0


def n_language_tags(tags):
    return len({t for t in tags if re.fullmatch(r"[a-z]{2,3}", t) and t not in NOT_LANGUAGE_TAGS})


def license_of(tags):
    for t in tags:
        if t.startswith("license:"):
            return t.split(":", 1)[1].lower()
    return None


def license_group(lic):
    if not isinstance(lic, str) or not lic:
        return "Not specified"
    for group, names in LICENSE_GROUPS.items():
        if lic in names:
            return group
    if any(k in lic for k in ("llama", "gemma", "openrail", "bigscience", "bigcode")):
        return "Model-specific (Llama/Gemma/RAIL)"
    return "Other"


def base_model_of(tags):
    """'base_model:finetune:google/gemma-2b' -> ('google/gemma-2b', 'finetune')."""
    base = None
    for t in tags:
        if not t.startswith("base_model:"):
            continue
        parts = t.split(":")
        if len(parts) == 3 and parts[1] in BASE_RELATIONS:
            return parts[2], parts[1]
        if len(parts) == 2 and base is None:
            base = parts[1]
    return base, ("finetune" if base else None)


def strip_front_matter(text):
    """Remove the YAML metadata block at the top of a README."""
    if not isinstance(text, str):
        return ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:]
    return text


def card_features(text):
    body = strip_front_matter(text)
    low = body.lower()
    headings = re.findall(r"^#{1,6}\s+(.+)$", body, flags=re.M)
    heading_text = " ".join(headings).lower()
    words = len(re.findall(r"\w+", body))
    feats = {
        "has_card": int(words > 0),
        "card_log_words": math.log1p(words),
        "card_n_headings": len(headings),
        "card_n_code_blocks": body.count("```") // 2,
        "card_n_links": len(re.findall(r"https?://", body)),
        "card_is_template": int(TEMPLATE_PHRASE in low),
    }
    for name, pattern in CARD_SECTIONS.items():
        # a section counts if it is a heading, or clearly discussed in the body
        feats[name] = int(bool(re.search(pattern, heading_text)) or len(re.findall(pattern, low)) >= 2)
    return feats, body


def card_data_features(card_json):
    try:
        cd = json.loads(card_json) if isinstance(card_json, str) and card_json else {}
    except json.JSONDecodeError:
        cd = {}
    if not isinstance(cd, dict):
        cd = {}
    datasets = cd.get("datasets") or []
    if isinstance(datasets, str):
        datasets = [datasets]
    return {
        "card_n_datasets": len(datasets),
        "card_has_metrics": int(bool(cd.get("metrics"))),
        "card_has_base_model": int(bool(cd.get("base_model"))),
    }


def file_features(siblings):
    files = [f for f in (siblings or "").split("|") if f]
    low = [f.lower() for f in files]
    return {
        "n_files": len(files),
        "has_safetensors": int(any(f.endswith(".safetensors") for f in low)),
        "has_gguf": int(any(f.endswith(".gguf") for f in low)),
        "has_onnx": int(any(f.endswith(".onnx") for f in low)),
        "has_tokenizer": int(any("tokenizer" in f for f in low)),
    }


def _safe_log(x, scale=1.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return math.log1p(x / scale) if x == x and x >= 0 else np.nan


# ------------------------------------------------------------------ main entry
def build_features(models: pd.DataFrame, author_map: pd.DataFrame) -> pd.DataFrame:
    """
    models: one row per model with raw columns (tags, siblings, card_data, card_text,
            n_params, used_storage, has_model_index, pipeline_tag, library_name, gated,
            author, model_id, n_indic_languages, primary_lang, author_prior_models).
    Returns a frame with exactly FEATURES columns (plus model_id).
    """
    df = models.copy()
    tags = df["tags"].fillna("").str.split("|")

    out = pd.DataFrame({"model_id": df["model_id"].values})
    out["task_group"] = df["pipeline_tag"].map(TASK_GROUPS).fillna(
        df["pipeline_tag"].apply(lambda p: "Unspecified" if pd.isna(p) else "Other")).values
    out["library"] = df["library_name"].fillna("unspecified").values
    out["license_group"] = tags.apply(license_of).apply(license_group).values
    base = tags.apply(base_model_of)
    out["base_author"] = base.str[0].str.split("/").str[0].fillna("none (original)").values
    out["base_relation"] = base.str[1].fillna("none").values
    out["is_derivative"] = base.str[0].notna().astype(int).values
    out["is_gated"] = df["gated"].astype(str).str.lower().isin(["auto", "manual", "true"]).astype(int).values

    amap = author_map.set_index("author")
    out["org_type"] = df["author"].map(amap["org_type"]).fillna("Unclassified").values
    out["is_indian_builder"] = df["author"].map(amap["is_indian"]).fillna(0).astype(int).values

    out["primary_lang"] = df["primary_lang"].fillna("multi").values
    out["n_indic_languages"] = df["n_indic_languages"].fillna(1).astype(int).values
    out["is_india_dedicated"] = ((out["n_indic_languages"] == 1)
                                 & (tags.apply(indian_tag_share).values >= 0.5)).astype(int)
    out["n_language_tags"] = tags.apply(n_language_tags).values

    out["log_params"] = df["n_params"].apply(_safe_log).values
    out["log_storage_mb"] = df["used_storage"].apply(lambda x: _safe_log(x, 1e6)).values
    ff = pd.DataFrame(list(df["siblings"].apply(file_features)))
    for c in ff.columns:
        out[c] = ff[c].values
    out["has_model_index"] = df["has_model_index"].fillna(0).astype(int).values

    cdf = pd.DataFrame(list(df["card_data"].apply(card_data_features)))
    for c in cdf.columns:
        out[c] = cdf[c].values

    parsed = df["card_text"].apply(card_features)
    cf = pd.DataFrame(list(parsed.str[0]))
    for c in cf.columns:
        out[c] = cf[c].values
    out[TEXT] = parsed.str[1].str.slice(0, 5000).values

    names = df["model_id"].str.split("/", n=1).str[-1]
    out["name_is_throwaway"] = names.str.contains(THROWAWAY_NAME).astype(int).values
    out["name_len"] = names.str.len().values
    out["log_author_prior_models"] = np.log1p(df["author_prior_models"].fillna(0)).values

    return out[["model_id"] + FEATURES]
