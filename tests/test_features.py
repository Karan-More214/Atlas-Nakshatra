"""Guards that keep the project honest: no leakage, time-ordered split, stable features."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
from features import FEATURES, build_features, card_features, base_model_of, license_group  # noqa: E402

AUTHOR_MAP = pd.DataFrame({"author": ["ai4bharat"], "org_name": ["AI4Bharat"],
                           "org_type": ["Indian Research & Academia"], "is_indian": [1]})


def raw_row(**overrides):
    row = {
        "model_id": "ai4bharat/test-model", "author": "ai4bharat", "pipeline_tag": "translation",
        "library_name": "transformers", "gated": "False",
        "tags": "mr|en|license:apache-2.0|base_model:finetune:facebook/mbart-large-50|translation",
        "card_data": json.dumps({"datasets": ["a", "b"], "metrics": ["bleu"]}),
        "has_model_index": 1, "n_params": 6.1e8, "used_storage": 2.4e9,
        "siblings": "README.md|config.json|model.safetensors|tokenizer.json",
        "card_text": "---\nlicense: mit\n---\n# Model\n## How to use\n```python\nx\n```\n## Evaluation\nBLEU 30",
        "n_indic_languages": 1, "primary_lang": "mr", "author_prior_models": 12,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_no_leaky_features():
    assert not set(FEATURES) & C.LEAKY_COLUMNS


def test_build_features_shape_and_values():
    f = build_features(raw_row(), AUTHOR_MAP).iloc[0]
    assert list(f.index) == ["model_id"] + FEATURES
    assert f["license_group"] == "Permissive"
    assert f["base_author"] == "facebook" and f["is_derivative"] == 1
    assert f["org_type"] == "Indian Research & Academia" and f["is_indian_builder"] == 1
    assert f["is_india_dedicated"] == 1
    assert f["has_safetensors"] == 1 and f["card_n_datasets"] == 2
    assert f["sec_usage"] == 1 and f["sec_evaluation"] == 1
    assert f["name_is_throwaway"] == 1  # "test-model"


def test_missing_values_do_not_crash():
    f = build_features(raw_row(tags="", card_data=None, siblings=None, card_text="", n_params=None,
                               used_storage=None, pipeline_tag=None), AUTHOR_MAP).iloc[0]
    assert f["has_card"] == 0 and f["license_group"] == "Not specified" and f["task_group"] == "Unspecified"


def test_front_matter_is_not_counted_as_card_body():
    feats, body = card_features("---\nlicense: mit\ndatasets: [x]\n---\n")
    assert feats["has_card"] == 0 and body.strip() == ""


def test_template_card_detected():
    feats, _ = card_features("# M\n## Model description\n[More Information Needed]\n")
    assert feats["card_is_template"] == 1


@pytest.mark.parametrize("tags,expected", [
    (["base_model:adapter:meta-llama/Llama-3.2-1B"], ("meta-llama/Llama-3.2-1B", "adapter")),
    (["base_model:google/gemma-2b"], ("google/gemma-2b", "finetune")),
    ([], (None, None)),
])
def test_base_model_parsing(tags, expected):
    assert base_model_of(tags) == expected


def test_license_groups():
    assert license_group("cc-by-nc-4.0") == "Non-commercial"
    assert license_group("llama3.1") == "Model-specific (Llama/Gemma/RAIL)"


def test_split_is_chronological():
    path = C.PROCESSED_DIR / "dataset.parquet"
    if not path.exists():
        pytest.skip("dataset not built yet")
    d = pd.read_parquet(path)
    assert d.loc[d.split == "train", "created_at"].max() <= d.loc[d.split == "valid", "created_at"].min()
    assert d.loc[d.split == "valid", "created_at"].max() <= d.loc[d.split == "test", "created_at"].min()
    assert d["age_days"].between(C.MIN_AGE_DAYS, C.MAX_AGE_DAYS).all()
