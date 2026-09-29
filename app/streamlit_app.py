"""
Atlas-Nakshatra - Launch Checker
Will your Indian-language AI model become a star, or go unused?

    streamlit run app/streamlit_app.py
    ATLAS_DEMO=1 streamlit run app/streamlit_app.py    # use the synthetic demo model
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
from features import TASK_GROUPS  # noqa: E402
from predict import (MODEL_PATH, explain_one, features_from_draft, fetch_live,  # noqa: E402
                     load_bundle)

# Validated against the theme's navy surface (#0B1229) - see .streamlit/config.toml.
# Plain green/red fails colorblind-separation on hue alone, so charts below encode
# the raises/lowers sign by bar *direction* too, never by color alone.
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
NAVY_CARD = "#141B36"
INK_MUTED = "#9aa3c0"

# This model's training-set size (see README "Results") - not stored in the pickled
# bundle, so it's a display constant here; update if the model is retrained on a
# different snapshot.
TRAINED_ON_MODELS = "7,440"

st.set_page_config(page_title="Atlas-Nakshatra", page_icon="✨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* Streamlit's fixed top toolbar sits above the page - give the block container
   enough top padding that the hero/tabs never slide underneath it. */
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1100px; }

.hero { padding: 0.25rem 0 1.1rem 0; border-bottom: 1px solid rgba(245,243,238,0.10); margin-bottom: 1.3rem; }
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 0.2rem 0; letter-spacing: -0.01em; }
.hero p.tagline { font-size: 1.05rem; color: #cfd3e6; margin: 0 0 0.6rem 0; }
.hero .links a {
    color: #FF9F1C; text-decoration: none; font-weight: 600; margin-right: 1.1rem; font-size: 0.92rem;
}
.hero .links a:hover { text-decoration: underline; }

/* Streamlit >=1.30 renders st.tabs as sticky; without this the first tab's top
   border can end up clipped by the hero's bottom margin on some widths. */
div[data-testid="stTabs"] { margin-top: 0.4rem; }

.kpi-card {
    background: #141B36; border: 1px solid rgba(245,243,238,0.08); border-radius: 12px;
    padding: 1rem 1.1rem; height: 100%;
}
.kpi-card .kpi-value { font-size: 1.65rem; font-weight: 800; color: #FF9F1C; line-height: 1.15; }
.kpi-card .kpi-label { font-size: 0.82rem; color: #cfd3e6; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; margin-bottom: 0.15rem; }
.kpi-card .kpi-explain { font-size: 0.85rem; color: #9aa3c0; margin-top: 0.35rem; line-height: 1.35; }

.verdict-badge {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px; font-weight: 700;
    font-size: 0.95rem; margin-top: 0.4rem;
}
.verdict-good { background: rgba(12,163,12,0.18); color: #35d035; }
.verdict-warn { background: rgba(250,178,25,0.18); color: #fab219; }
.verdict-bad  { background: rgba(208,59,59,0.20); color: #ff6b6b; }

footer.app-footer {
    margin-top: 2.2rem; padding-top: 1.1rem; border-top: 1px solid rgba(245,243,238,0.10);
    color: #9aa3c0; font-size: 0.88rem;
}
footer.app-footer a { color: #FF9F1C; text-decoration: none; font-weight: 600; }
footer.app-footer a:hover { text-decoration: underline; }

@media (max-width: 640px) {
    .hero h1 { font-size: 1.6rem; }
    .hero p.tagline { font-size: 0.95rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python src/run_pipeline.py` "
             "(or `--demo` with ATLAS_DEMO=1) first.")
    st.stop()

bundle = load_bundle(str(MODEL_PATH))
languages = pd.read_csv(C.CONFIG_DIR / "languages.csv")
m = bundle["test_metrics"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        f"""Atlas-Nakshatra predicts, at launch, whether a Hugging Face model tagged with an
Indian language will be **adopted** — defined as **≥ {C.ADOPTION_MIN_DOWNLOADS_30D} downloads
in its most recent 30 days**.

It's trained on **{TRAINED_ON_MODELS} real Hugging Face models** (repackaged / quantized
re-uploads excluded, since automated tools download those regardless of who chose them).

**Key finding:** who builds a model (builder type, prior track record) matters more than
documentation alone — the docs-checklist baseline barely beats guessing.
""")
    st.caption("SHAP shows association, not causation - see the README for the full write-up.")

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero">
  <h1>✨ Atlas-Nakshatra</h1>
  <p class="tagline">Will your Indian-language AI model become a star?</p>
  <div class="links">
    <a href="https://github.com/Karan-More214/Atlas-Nakshatra" target="_blank">📦 GitHub repo</a>
    <a href="https://github.com/Karan-More214/Bharat-AI-Atlas" target="_blank">📊 Bharat AI Atlas (the prequel)</a>
  </div>
</div>
""", unsafe_allow_html=True)

if bundle.get("demo"):
    st.warning("Demo mode: this model was trained on synthetic data. Scores show how the app works, not real findings.")

# ------------------------------------------------------------------ KPI cards
max_lift = 1 / m["base_rate"] if m["base_rate"] else float("nan")
kpis = [
    ("Model", bundle["model_family"], "The model family chosen on validation data."),
    ("ROC-AUC", f"{m['roc_auc']:.2f}",
     "How well it ranks adopted models above non-adopted ones (0.5 = random, 1.0 = perfect)."),
    ("PR-AUC vs base rate", f"{m['pr_auc']:.2f} vs {m['base_rate']:.2f}",
     "Precision-recall score vs. just guessing the average adoption rate."),
    ("Lift, top 10%", f"{m['lift_top10pct']:.1f}×",
     f"Top-scored models are adopted this much more often than average (max possible {max_lift:.1f}×)."),
]
cols = st.columns(4)
for col, (label, value, explain) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-explain">{explain}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")


# ------------------------------------------------------------------ result rendering
def _verdict(p: float, threshold: float, band: float = 0.10):
    if p >= threshold:
        return "good", "⭐ Likely to be adopted", GOOD
    if p >= threshold - band:
        return "warn", "🌗 Borderline - near the threshold", WARNING
    return "bad", "🌑 At risk of going unused", CRITICAL


def _gauge(p: float, threshold: float):
    kind, _, color = _verdict(p, threshold)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": INK_MUTED, "tickfont": {"color": INK_MUTED}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": NAVY_CARD,
            "borderwidth": 0,
            "threshold": {"line": {"color": "#F5F3EE", "width": 3}, "thickness": 0.85,
                          "value": threshold * 100},
            "steps": [{"range": [0, 100], "color": "rgba(245,243,238,0.06)"}],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=25, r=25, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#F5F3EE"})
    return fig, kind


def _drivers_chart(drivers: pd.DataFrame):
    d = drivers.sort_values("impact", key=lambda s: s.abs()).copy()
    colors = [GOOD if v > 0 else CRITICAL for v in d["impact"]]
    fig = go.Figure(go.Bar(
        x=d["impact"], y=d["label"], orientation="h", marker_color=colors,
        customdata=d["value"],
        hovertemplate="<b>%{y}</b><br>Value: %{customdata}<br>Effect: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(245,243,238,0.35)", line_width=1)
    fig.update_layout(
        height=300, margin=dict(l=10, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F5F3EE"},
        xaxis={"title": "Effect on score (log-odds)", "gridcolor": "rgba(245,243,238,0.08)",
              "zeroline": False},
        yaxis={"automargin": True},
    )
    return fig


def show_result(row, title):
    p, drivers, sugg = explain_one(bundle, row)
    kind, verdict_text, _ = _verdict(p, bundle["threshold"])
    st.subheader(title)

    left, right = st.columns([1, 2])
    with left:
        fig, kind = _gauge(p, bundle["threshold"])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(f'<span class="verdict-badge verdict-{kind}">{verdict_text}</span>',
                   unsafe_allow_html=True)
        st.caption(f"Decision threshold {bundle['threshold']:.0%}, chosen on validation data.")
    with right:
        st.markdown("**What drives this score** (SHAP, log-odds — green raises, red lowers)")
        st.plotly_chart(_drivers_chart(drivers), width="stretch", config={"displayModeBar": False})

    st.markdown("**What would help most**")
    if len(sugg):
        show = sugg.head(5).rename(columns={
            "action": "Suggested change", "new_probability": "New probability", "change": "Change",
        })
        show["Change"] = show["Change"].map(lambda v: f":green-badge[+{v:.0%}]")
        st.dataframe(
            show, hide_index=True, width="stretch",
            column_config={
                "New probability": st.column_config.ProgressColumn(
                    "New probability", format="%.0f%%", min_value=0, max_value=1),
                "Change": st.column_config.MarkdownColumn("Change", width="small"),
            },
        )
        st.caption("Estimated by re-scoring the model with each change. These are associations "
                   "learned from past models, not guarantees.")
    else:
        st.success("No quick wins found — the launch checklist already looks strong.")
    with st.expander("All feature values"):
        st.dataframe(row.drop(columns=["card_text"]).T.rename(columns={row.index[0]: "value"}).astype(str))


# ------------------------------------------------------------------ tabs
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

st.markdown(f"""
<footer class="app-footer">
  Atlas-Nakshatra · sequel to Bharat AI Atlas · data: Hugging Face Hub · trained {bundle['trained_at'][:10]}
  on models created {bundle['train_period'][0]} → {bundle['train_period'][1]}<br>
  Built by <b>Karan More</b> ·
  <a href="https://www.linkedin.com/in/karan-more21" target="_blank">LinkedIn</a> ·
  <a href="https://github.com/Karan-More214" target="_blank">GitHub</a>
</footer>
""", unsafe_allow_html=True)
