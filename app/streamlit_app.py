"""
Atlas-Nakshatra - Launch Checker
Will your Indian-language AI model become a star, or go unused?

    streamlit run app/streamlit_app.py
    ATLAS_DEMO=1 streamlit run app/streamlit_app.py    # use the synthetic demo model
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
from features import TASK_GROUPS  # noqa: E402
from predict import (MODEL_PATH, explain_one, features_from_draft, fetch_live,  # noqa: E402
                     load_bundle)

st.set_page_config(page_title="Atlas-Nakshatra", page_icon="✨", layout="wide")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python src/run_pipeline.py` "
             "(or `--demo` with ATLAS_DEMO=1) first.")
    st.stop()

bundle = load_bundle(str(MODEL_PATH))
languages = pd.read_csv(C.CONFIG_DIR / "languages.csv")

st.title("✨ Atlas-Nakshatra")
st.caption("Predicting which Indian-language AI models become stars — and what builders can do about it.")
if bundle.get("demo"):
    st.warning("Demo mode: this model was trained on synthetic data. Scores show how the app works, not real findings.")

m = bundle["test_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", bundle["model_family"])
c2.metric("PR-AUC (newest models)", f"{m['pr_auc']:.2f}", f"vs {m['base_rate']:.2f} guessing", delta_color="off")
c3.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
c4.metric("Adopted means", bundle["target"].replace("downloads_30d", "30-day downloads"))


def show_result(row, title):
    p, drivers, sugg = explain_one(bundle, row)
    st.subheader(title)
    left, right = st.columns([1, 2])
    with left:
        verdict = "⭐ Likely to be adopted" if p >= bundle["threshold"] else "🌑 At risk of going unused"
        st.metric("Adoption probability", f"{p:.0%}")
        st.markdown(f"**{verdict}**  \n<small>Decision threshold {bundle['threshold']:.0%}, "
                    f"chosen on validation data</small>", unsafe_allow_html=True)
    with right:
        st.markdown("**What drives this score** (SHAP, log-odds; + raises, − lowers)")
        chart = drivers.set_index("label")[["impact"]].iloc[::-1]
        st.bar_chart(chart, horizontal=True, height=280)
    st.markdown("**What would help most**")
    if len(sugg):
        show = sugg.head(5).assign(
            new_probability=lambda d: d["new_probability"].map("{:.0%}".format),
            change=lambda d: d["change"].map("{:+.0%}".format))
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Estimated by re-scoring the model with each change. These are associations "
                   "learned from past models, not guarantees.")
    else:
        st.success("No quick wins found — the launch checklist already looks strong.")
    with st.expander("All feature values"):
        st.dataframe(row.drop(columns=["card_text"]).T.rename(columns={row.index[0]: "value"}).astype(str))


tab_live, tab_draft = st.tabs(["🔎 Check a published model", "📝 Plan a new model"])

with tab_live:
    model_id = st.text_input("Hugging Face model ID", placeholder="ai4bharat/indic-bert")
    if st.button("Check model", type="primary") and model_id:
        try:
            with st.spinner("Fetching metadata and model card from Hugging Face…"):
                row = fetch_live(model_id.strip())
            show_result(row, model_id)
        except Exception as err:  # not found, gated, network
            st.error(f"Could not fetch {model_id}: {err}")

with tab_draft:
    st.markdown("Describe the model you are about to publish.")
    with st.form("draft"):
        a, b = st.columns(2)
        author = a.text_input("Your Hugging Face username", "my-lab")
        name = b.text_input("Model name", "whisper-small-marathi")
        lang = a.selectbox("Language", languages["lang_code"],
                           format_func=lambda c: languages.set_index("lang_code").loc[c, "lang_name"],
                           index=int((languages["lang_code"] == "mr").idxmax()))
        task = b.selectbox("Task", sorted(TASK_GROUPS), index=sorted(TASK_GROUPS).index("automatic-speech-recognition"))
        license_ = a.selectbox("License", ["apache-2.0", "mit", "cc-by-4.0", "cc-by-nc-4.0", "llama3.1", "gemma", ""])
        base = b.text_input("Base model (optional)", "openai/whisper-small")
        sections = st.multiselect(
            "Model-card sections you will write",
            ["usage", "training_data", "evaluation", "limitations", "citation"],
            default=["usage"],
            format_func=lambda s: s.replace("_", " ").title())
        description = st.text_area("Short description", "Whisper small fine-tuned for Marathi speech recognition.")
        x, y_, z = st.columns(3)
        safetensors = x.checkbox("Safetensors weights", True)
        english = y_.checkbox("Also supports English", False)
        prior = z.number_input("Models you've published before", 0, 10_000, 0)
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        row = features_from_draft(dict(
            author=author, name=name, lang=lang, task=task, license=license_ or None,
            base_model=base or None, sections=sections, description=description,
            safetensors=safetensors, english=english, author_prior_models=prior))
        show_result(row, f"{author}/{name} (draft)")

st.divider()
st.caption("Atlas-Nakshatra · sequel to Bharat AI Atlas · data: Hugging Face Hub · "
           f"trained {bundle['trained_at'][:10]} on models created {bundle['train_period'][0]} → {bundle['train_period'][1]}")
