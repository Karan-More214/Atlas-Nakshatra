# ✨ Atlas-Nakshatra: Predicting Which Indian-Language AI Models Become Stars

> **Thousands of AI models are published for India's languages, but most go unused. Can we predict at launch which ones will be adopted, and tell builders what to change?**

*Nakshatra* (नक्षत्र) means "star". This is the data science sequel to my analytics project **[Bharat AI Atlas](https://github.com/Karan-More214/Bharat-AI-Atlas)**, which found that downloads of Indian-language AI are extremely concentrated: 41 models get 80% of all downloads and 11.8% are never downloaded. Bharat AI Atlas *described* that inequality. Atlas-Nakshatra *predicts and explains* it.

| | Bharat AI Atlas (analyst) | Atlas-Nakshatra (data scientist) |
|---|---|---|
| Question | What exists, who builds it, who uses it? | Which new models will be used, and why? |
| Methods | SQL, Power BI | scikit-learn, LightGBM, NLP on model cards, SHAP, calibration |
| Output | Dashboard | Trained model + Launch Checker app |

---

## 🔑 Results
Trained on 12,116 eligible Hugging Face models tagged with an Indian language (8,481 train / 1,817 validation / 1,818 test, split chronologically: train ends 2026-01-10, test starts 2026-04-11).

- **Final model:** LightGBM, calibrated · **PR-AUC 0.87** on the newest 1,818 models (vs. 0.62 for guessing the base rate, 0.66 for a documentation-checklist baseline) · **ROC-AUC 0.80**.
- The top 10% of models by predicted probability are adopted **1.6× more often** than average (98% vs. 62%).
- At the chosen threshold: precision 0.75, recall 0.86 (F1 0.80).
- **Strongest drivers** (by mean |SHAP|): GGUF (locally-runnable) weights, links in the model card, builder type, parameter count, number of files, and the card's text topics.

These are launch-time associations the model found useful for prediction, not proven causes — see [Doing it honestly](#-doing-it-honestly). Adding a link or a GGUF file does not, by itself, guarantee more downloads.

![What drives adoption](reports/figures/shap_importance.png)
![Precision-recall on the test set](reports/figures/pr_curve.png)

## ❓ Problem framing
- **Unit:** one Hugging Face model tagged with at least one of India's 22 official languages.
- **Target (`is_adopted`):** at least **10 downloads in the last 30 days** at the snapshot date, i.e. the model is still being used. The threshold is set in `src/config.py`.
- **Who is eligible:** models with a real created date that are **90 days to 3 years old**. Younger models haven't had time to be found. Models from before March 2022 carry Hugging Face's placeholder date and are excluded.
- **Features:** only what a builder controls or knows **at launch**:

| Group | Examples |
|---|---|
| What it is | task, library, language, number of languages, dedicated to one Indian language |
| Where it comes from | base model provider (Meta, OpenAI, Google…), fine-tune / adapter / quantized |
| Who built it | builder type, Indian builder, number of earlier models by the same author |
| Packaging | safetensors / GGUF / ONNX weights, tokenizer files, parameter count, repo size, gated access |
| Openness | license group (permissive, non-commercial, model-specific…) |
| Documentation | card length, headings, code examples, usage / training data / evaluation / limitations / citation sections, unedited template text, structured eval results, datasets and metrics listed |
| Card wording | TF-IDF of the model card compressed to 20 topics (SVD) |
| Name | test/demo-style name, name length |

## 🛡️ Doing it honestly
- **No leakage.** Downloads, likes, trending score, Spaces and child models are outcomes and never features. `tests/test_features.py` fails if one sneaks in.
- **Chronological split.** Train = oldest 70% of models, validation = next 15%, test = newest 15%. This mimics the real use: score models that did not exist when the model was trained.
- **Test set used once.** Hyper-parameters, the choice between logistic regression and LightGBM, probability calibration (Platt) and the decision threshold are all chosen on validation.
- **Baselines first.** Every model is compared with guessing the base rate and with a simple documentation-checklist score.
- **Right metric.** Adoption is rare, so the main metric is **PR-AUC**, plus ROC-AUC, Brier score and precision in the top 10%.
- **Association, not causation.** SHAP and the app's "what would help" suggestions show what the model associates with adoption. They are hypotheses, not proof that editing a card causes downloads.

## 🏗️ Pipeline
```
Hugging Face Hub API
   │  collect.py      metadata + card data + files + params + eval results   → data/raw/models_raw_<date>.parquet
   │  fetch_cards.py  README.md model cards (cached, resumable)              → data/raw/cards/cards.jsonl
   ▼
build_dataset.py      target, eligibility, features (features.py), time split → data/processed/dataset.parquet
   ▼
train.py              baselines → logistic regression → LightGBM → calibration → models/atlas_nakshatra.joblib
   ▼
explain.py            SHAP importance, adoption by group, findings draft     → reports/
   ▼
predict.py + app/     Launch Checker: probability, drivers, suggested fixes
```

## 🖥️ Launch Checker app
```bash
streamlit run app/streamlit_app.py
```
- **Check a published model:** paste a Hugging Face model ID and get its adoption probability, the top SHAP drivers, and the changes that would raise it most.
- **Plan a new model:** describe a model before publishing it (language, task, license, base model, card sections) and see its predicted adoption.

## ▶️ How to run
```bash
git clone https://github.com/Karan-More214/Atlas-Nakshatra.git
cd Atlas-Nakshatra
python -m venv venv && venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                                 # Mac/Linux: cp; add your HF token

python src/run_pipeline.py --limit 200                 # quick test on a small sample
python src/run_pipeline.py                             # full run (card download takes a while)
python src/predict.py --model-id ai4bharat/indic-bert  # score one model from the terminal
python -m pytest -q tests
```

**Offline demo.** To check the code without internet, use synthetic data. It is written to separate `demo/` folders and never mixes with real data. Its numbers mean nothing.
```bash
python src/run_pipeline.py --demo
set ATLAS_DEMO=1                                  # Mac/Linux: export ATLAS_DEMO=1
streamlit run app/streamlit_app.py
```

## 📁 Repository structure
```
├── config/        languages.csv, author_mapping.csv (shared with Bharat AI Atlas)
├── src/           config, collect, fetch_cards, features, build_dataset, train, explain, predict, run_pipeline, make_demo_data
├── app/           streamlit_app.py (Launch Checker)
├── tests/         leakage, split and feature tests
├── data/          raw + processed (git-ignored)
├── models/        trained model (git-ignored)
└── reports/       metrics, model comparison, SHAP, figures
```

## ⚠️ Limitations
- **One snapshot.** Card, files and license are read at the snapshot date, not at launch, so a card improved after a model became popular looks "launch-time" here. Weekly snapshots (as in Bharat AI Atlas) would fix this.
- `downloads` counts are global, and include automated and CI downloads. They measure use, not use in India. This is likely why the base "adoption" rate (62-74% by split) is much higher than Bharat AI Atlas's "ever downloaded" figures: 10 downloads in 30 days is a low bar, and a lot of that traffic is automated.
- The Hugging Face API no longer accepts `usedStorage` as an `expand` field, so repo-size-on-disk (`log_storage_mb`) cannot currently be collected and is a dead feature (always missing, median-imputed to nothing). Repo size would need to be estimated another way (e.g. per-file sizes from `repo_info`) to be useful again.
- Builder type is mapped by hand for 215 authors, so most small authors are "Unclassified".
- Language tags can be wrong (see Bharat AI Atlas), and documentation features overlap, so read SHAP for groups of related features rather than one feature at a time.
- One author, `mradermacher` (a prolific GGUF-quantization publisher), is the largest single author in both train (9.4%) and test (3.9%) and is adopted almost every time (99% in test). Excluding it from the test set only drops PR-AUC from 0.87 to 0.85, so the model is not just memorizing that one author.

## 🚀 Next steps
- **Language tag checker:** detect mislabelled models with language ID on card text.
- **Language gap forecast:** forecast dedicated models per language with uncertainty intervals.
- FastAPI endpoint, Docker image and MLflow tracking (`train.py` already logs to MLflow if it is installed).

---
**Author:** Karan More · [LinkedIn](https://www.linkedin.com/in/karan-more21) · [GitHub](https://github.com/Karan-More214) · Data: [Hugging Face Hub](https://huggingface.co)
