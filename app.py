"""
Interactive demo for the document-recovery pipeline (ML-T2-022).

Runs the real Step 5 pipeline live on any sentence: render -> tear ->
OCR -> per-word damage classification -> fine-tuned RoBERTa
reconstruction -> confidence flag -> schema-validated JSON. A second tab
explores the committed 739-document held-out evaluation.

    streamlit run app.py
"""

import glob
import html
import json
import os
import random
import sys

import altair as alt
import jsonschema
import pandas as pd
import streamlit as st
from jiwer import cer, wer
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import paths  # noqa: E402
from baseline_ocr import CLEAN_TEXT, render_clean_image  # noqa: E402
from build_document_json import (  # noqa: E402
    CONFIDENCE_THRESHOLD, FUNCTION_WORDS, build_word_records, load_reconstruction_model, load_schema,
    ocr_words_from_image,
)
from reconstruction import classify_all_words  # noqa: E402
from torn_regions import apply_tear, estimate_background_color, generate_ribbon_tear  # noqa: E402
from word_targeted_masks import build_targeted_mask, get_word_boxes  # noqa: E402

st.set_page_config(page_title="Document Recovery · ML-T2-022", page_icon=":material/description:", layout="wide")

# Warm monochrome palette; colour is reserved for meaning (confident / flagged).
INK = "#2F3437"
MUTED = "#787774"
RULE = "#EAEAEA"
GREEN_BG, GREEN_FG = "#EDF3EC", "#346538"
RED_BG, RED_FG = "#FDEBEC", "#9F2F2D"
YELLOW_BG, YELLOW_FG = "#FBF3DB", "#956400"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,400..600&display=swap');

    :root { --ink:#2F3437; --muted:#787774; --rule:#EAEAEA; --bone:#F7F6F3; --paper:#FFFFFF; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
    .block-container { padding-top: 3.2rem; padding-bottom: 5rem; max-width: 1240px; }

    /* fade-in for custom blocks */
    @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
    .rise { animation: rise 600ms cubic-bezier(0.16, 1, 0.3, 1) both; }

    /* hero */
    .hero { position: relative; padding: 8px 0 30px; }
    .hero::before { content: ""; position: absolute; inset: -60px -40px auto auto; width: 520px; height: 320px;
      background: radial-gradient(closest-side, rgba(214,196,160,.16), transparent); pointer-events: none; }
    .eyebrow { font-family: 'Geist Mono', monospace; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
    .hero h1 { font-family: 'Newsreader', serif; font-weight: 500; font-size: 3.1rem; line-height: 1.08;
      letter-spacing: -0.025em; color: #111; margin: 14px 0 14px; padding: 0; max-width: 900px; }
    .hero .lede { font-size: 1.08rem; line-height: 1.6; color: #4a4f52; max-width: 720px; margin: 0; }
    .hero-meta { display: flex; flex-wrap: wrap; margin-top: 26px; border-top: 1px solid var(--rule); }
    .hero-meta > div { padding: 14px 28px 0 0; margin-right: 28px; border-right: 1px solid var(--rule); }
    .hero-meta > div:last-child { border-right: 0; }
    .hero-meta .k { font-family: 'Geist Mono', monospace; font-size: 11px; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); }
    .hero-meta .v { font-size: .95rem; color: var(--ink); margin-top: 3px; }

    /* stage bento */
    .bento { display: grid; grid-template-columns: 1.35fr 1fr 1fr 1fr 1.15fr; gap: 12px; margin: 6px 0 34px; }
    .cell { background: var(--paper); border: 1px solid var(--rule); border-radius: 10px; padding: 20px 20px 22px;
      transition: box-shadow 200ms ease; }
    .cell:hover { box-shadow: 0 2px 8px rgba(0,0,0,.04); }
    .cell .n { font-family: 'Geist Mono', monospace; font-size: 11px; color: var(--muted); letter-spacing: .06em; }
    .cell .t { font-family: 'Newsreader', serif; font-size: 1.28rem; color: #111; margin: 10px 0 6px; letter-spacing: -0.01em; }
    .cell .d { font-size: .85rem; line-height: 1.5; color: var(--muted); }

    /* section headers */
    .sec { display: grid; grid-template-columns: 48px 1fr; align-items: baseline; margin: 42px 0 16px;
      padding-top: 18px; border-top: 1px solid var(--rule); }
    .sec .i { font-family: 'Geist Mono', monospace; font-size: 12px; color: var(--muted); }
    .sec h3 { font-family: 'Newsreader', serif; font-weight: 500; font-size: 1.55rem; letter-spacing: -0.015em; color: #111; margin: 0; padding: 0; }
    .sec p { grid-column: 2; margin: 4px 0 0; color: var(--muted); font-size: .92rem; }

    /* stat tiles */
    .stats { display: grid; grid-template-columns: repeat(var(--cols), 1fr); border: 1px solid var(--rule);
      border-radius: 10px; background: var(--paper); margin: 18px 0 6px; }
    .stat { padding: 18px 20px; border-right: 1px solid var(--rule); }
    .stat:last-child { border-right: 0; }
    .stat .k { font-family: 'Geist Mono', monospace; font-size: 11px; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); }
    .stat .v { font-family: 'Newsreader', serif; font-size: 2.1rem; line-height: 1.15; color: #111; margin-top: 8px;
      font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .stat .s { font-size: .8rem; margin-top: 4px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .good { color: #346538 !important; } .bad { color: #9F2F2D !important; }

    /* recovered text */
    .legend { display: flex; gap: 18px; align-items: center; font-size: .82rem; color: var(--muted); margin-bottom: 10px; }
    .sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
    .recovered { font-family: 'Newsreader', serif; font-size: 1.4rem; line-height: 2.35rem; color: #111;
      background: var(--paper); border: 1px solid var(--rule); border-radius: 10px; padding: 22px 28px; }
    .w-ok, .w-flag { padding: 1px 6px; border-radius: 4px; }
    .w-ok { background: #EDF3EC; color: #346538; box-shadow: inset 0 -2px 0 #BCD3B8; }
    .w-flag { background: #FDEBEC; color: #9F2F2D; box-shadow: inset 0 -2px 0 #EDB9B7; }

    /* tags, notes */
    .tag { display: inline-block; font-family: 'Geist Mono', monospace; font-size: 11px; letter-spacing: .06em;
      text-transform: uppercase; padding: 4px 10px; border-radius: 9999px; }
    .note { border: 1px solid var(--rule); border-radius: 10px; background: var(--paper); padding: 18px 22px; color: var(--muted); font-size: .92rem; }
    .note b { color: var(--ink); font-weight: 600; }
    .caption-sm { font-size: .85rem; color: var(--muted); line-height: 1.55; }
    .chart-title { font-family: 'Newsreader', serif; font-size: 1.15rem; color: #111; margin: 0 0 4px; }

    /* images as paper */
    [data-testid="stImage"] img { border: 1px solid var(--rule); border-radius: 8px; background: #fff; }
    [data-testid="stImageCaption"], [data-testid="caption"] { font-family: 'Geist Mono', monospace; font-size: 11px !important;
      letter-spacing: .04em; text-transform: uppercase; color: var(--muted) !important; }

    /* buttons */
    [data-testid="stBaseButton-primary"] { background: #111 !important; border: 1px solid #111 !important; border-radius: 6px !important;
      font-weight: 500; letter-spacing: .01em; box-shadow: none !important; transition: background 150ms ease, transform 100ms ease; }
    [data-testid="stBaseButton-primary"]:hover { background: #333 !important; }
    [data-testid="stBaseButton-primary"]:active { transform: scale(0.98); }
    [data-testid="stBaseButton-secondary"] { background: #fff !important; border: 1px solid var(--rule) !important; border-radius: 6px !important; box-shadow: none !important; }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid var(--rule); }
    .stTabs [data-baseweb="tab"] { padding: 10px 0; font-size: .95rem; color: var(--muted); }
    .stTabs [aria-selected="true"] { color: #111 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: #111 !important; height: 1.5px; }

    /* sidebar record */
    [data-testid="stSidebar"] { border-right: 1px solid var(--rule); }
    .rec-title { font-family: 'Newsreader', serif; font-size: 1.35rem; color: #111; margin: 4px 0 18px; letter-spacing: -0.01em; }
    .rec { margin: 0; }
    .rec dt { font-family: 'Geist Mono', monospace; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-top: 16px; }
    .rec dd { margin: 3px 0 0; font-size: .9rem; color: var(--ink); line-height: 1.45; }
    .mono { font-family: 'Geist Mono', monospace; font-size: .82rem; background: #EFEEEA; padding: 1px 6px; border-radius: 4px; color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


def altair_theme(chart):
    return (chart.configure_axis(labelFont="Geist", titleFont="Geist", labelColor=MUTED, titleColor=MUTED,
                                 gridColor="#F0EFEB", domainColor=RULE, tickColor=RULE, labelFontSize=11,
                                 titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelFont="Geist", titleFont="Geist", labelColor=MUTED, titleColor=MUTED, labelFontSize=11)
            .configure_view(strokeWidth=0)
            .configure_text(font="Geist", color=INK))


def section(index, title, note=None):
    note_html = f"<p>{note}</p>" if note else ""
    st.markdown(f'<div class="sec rise"><span class="i">{index}</span><h3>{title}</h3>{note_html}</div>',
                unsafe_allow_html=True)


def stat_row(items):
    cells = "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s {tone or ""}">{s or "&nbsp;"}</div></div>'
        for k, v, s, tone in items
    )
    st.markdown(f'<div class="stats rise" style="--cols:{len(items)}">{cells}</div>', unsafe_allow_html=True)


def tag(text, bg, fg):
    return f'<span class="tag" style="background:{bg};color:{fg}">{text}</span>'


@st.cache_resource(show_spinner="Loading the fine-tuned RoBERTa model")
def get_model():
    return load_reconstruction_model()


@st.cache_data
def load_test_sentences(n=60):
    with open(paths.CORPUS_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    test = [r.get("text") or r.get("sentence") for r in records if r["split"] == "test"]
    return [s for s in test if 8 <= len(s.split()) <= 22][:n]


@st.cache_data
def load_summaries():
    def read(name):
        p = os.path.join(paths.RESULTS_DIR, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    return read("held_out_summary_report.json"), read("held_out_summary_report_as_published.json")


@st.cache_data
def load_held_out_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(paths.RESULTS_DIR, "held_out_documents", "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        gt = d["evaluation"]["ground_truth_text"].split(" ")
        target = random.Random(f"held_out_target_{d['document_id']}").randrange(len(d["words"]))
        w = d["words"][target]
        truth = gt[target] if len(gt) == len(d["words"]) else ""
        rows.append({
            "document": d["document_id"],
            "target word": truth,
            "reconstructed": w["reconstructed_text"],
            "confidence": w["reconstruction_confidence"],
            "flagged": w["uncertain"],
            "correct": (w["reconstructed_text"] or "").strip(",.;:").lower() == truth.strip(",.;:").lower(),
            "CER": d["evaluation"]["cer"],
            "final text": d["final_text"],
        })
    return pd.DataFrame(rows)


def overlay_mask(img, mask):
    """Damaged image with the tear tinted and outlined, so viewers can see where it is."""
    base = img.convert("RGBA")
    tinted = Image.alpha_composite(base, Image.new("RGBA", base.size, (159, 47, 45, 46)))
    out = Image.composite(tinted, base, mask)
    draw = ImageDraw.Draw(out)
    bbox = mask.getbbox()
    if bbox:
        draw.rectangle(bbox, outline=(159, 47, 45, 255), width=1)
    return out.convert("RGB")


def zoom_on_damage(img, mask, context_px=170, scale=3):
    """Crop around the tear with some surrounding text, then upscale, so the
    damage is visible on a one-line rendered page."""
    bbox = mask.getbbox() or (0, 0, img.width, img.height)
    left = max(0, bbox[0] - context_px)
    right = min(img.width, bbox[2] + context_px)
    crop = img.crop((left, 0, right, img.height))
    return crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)


def run_pipeline(text, mode, target_index, seed):
    os.makedirs(paths.OUTPUTS_DIR, exist_ok=True)
    clean_path = os.path.join(paths.OUTPUTS_DIR, "demo_clean.png")
    render_clean_image(text, clean_path)
    clean_img = Image.open(clean_path).convert("RGB")
    width, height = clean_img.size
    bg = estimate_background_color(clean_img)

    gt_words = get_word_boxes(clean_path)
    if mode == "Targeted word":
        mask = build_targeted_mask(width, height, [gt_words[min(target_index, len(gt_words) - 1)]], seed=seed)
    else:
        mask = generate_ribbon_tear(width, height, seed=seed, edge_tear=False)
    damaged = apply_tear(clean_img, mask, bg)

    classified = classify_all_words(gt_words, mask)
    ocr_words = ocr_words_from_image(damaged)
    tokenizer, model = get_model()
    records = build_word_records(classified, ocr_words, tokenizer, model)

    gt_text = " ".join(w["text"] for w in gt_words)
    ocr_only = " ".join(r["ocr_text"] for r in records if r["ocr_text"])
    final_text = " ".join(r["final_text"] for r in records if r["final_text"])
    recon = [r for r in records if r["provenance"] == "reconstructed"]
    correct = [r for r in recon
               if r["reconstructed_text"].strip(",.;:").lower() == r["_ground_truth_text"].strip(",.;:").lower()]
    content = [r for r in recon if r["_ground_truth_text"].strip(",.;:").lower() not in FUNCTION_WORDS]

    document = {
        "document_id": "demo_0001",
        "source_image_path": paths.rel(clean_path),
        "page_number": 1,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "final_text": final_text,
        "words": [{k: v for k, v in r.items() if not k.startswith("_")} for r in records],
        "evaluation": {
            "ground_truth_text": gt_text,
            "cer": round(cer(gt_text, final_text), 4),
            "wer": round(wer(gt_text, final_text), 4),
            "reconstruction_top1_accuracy": round(len(correct) / len(recon), 4) if recon else 0.0,
            "content_word_accuracy": (
                round(sum(r in correct for r in content) / len(content), 4) if content else None
            ),
        },
    }
    try:
        jsonschema.validate(instance=document, schema=load_schema())
        schema_error = None
    except jsonschema.ValidationError as e:
        schema_error = e.message

    return {
        "clean": clean_img, "damaged": damaged, "overlay": overlay_mask(damaged, mask),
        "zoom_clean": zoom_on_damage(clean_img, mask), "zoom_damaged": zoom_on_damage(damaged, mask),
        "records": records, "gt_text": gt_text, "ocr_only": ocr_only, "final_text": final_text,
        "document": document, "schema_error": schema_error,
    }


def recovered_html(records):
    parts = []
    for r in records:
        t = html.escape(r["final_text"] or "—")
        if r["provenance"] == "ocr":
            parts.append(t)
        else:
            cls = "w-flag" if r["uncertain"] else "w-ok"
            parts.append(f'<span class="{cls}" title="confidence {r["reconstruction_confidence"]:.3f}">{t}</span>')
    return '<div class="recovered rise">' + " ".join(parts) + "</div>"


def delta(value, baseline):
    d = value - baseline
    tone = "good" if d < 0 else ("bad" if d > 0 else None)
    return f"{d:+.2%} vs OCR only", tone


# ---------------------------------------------------------------- sidebar
model_ok = os.path.isfile(os.path.join(paths.FINE_TUNED_MODEL_PATH, "config.json"))

with st.sidebar:
    status = tag("Fine-tuned", GREEN_BG, GREEN_FG) if model_ok else tag("Fallback", YELLOW_BG, YELLOW_FG)
    st.markdown(
        f"""
        <div class="rec-title">Project record</div>
        <dl class="rec">
          <dt>Problem</dt><dd>ML-T2-022 · Advanced ML Internship, Track 2</dd>
          <dt>Author</dt><dd>Gyimah Ramsey Opoku<br>BSc Computer Science, KNUST</dd>
          <dt>Reconstruction model</dt><dd>RoBERTa-base, fine-tuned &nbsp;{status}</dd>
          <dt>Locked threshold</dt><dd><span class="mono">{CONFIDENCE_THRESHOLD}</span> &nbsp;chosen on validation only</dd>
          <dt>OCR</dt><dd>Tesseract <span class="mono">--psm 6</span></dd>
          <dt>Source</dt><dd>github.com/aimlin9/<br>document-recovery-ml-internship</dd>
        </dl>
        """,
        unsafe_allow_html=True,
    )
    if not model_ok:
        st.markdown('<div class="note" style="margin-top:18px">Fine-tuned weights were not found in '
                    '<span class="mono">roberta-finetuned-final/</span>. Confidence flags are not calibrated '
                    'for the fallback model.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero rise">
      <div class="eyebrow">LearnDepth Advanced ML Internship &nbsp;/&nbsp; Track 2 &nbsp;/&nbsp; ML-T2-022</div>
      <h1>Recovering structure from damaged documents</h1>
      <p class="lede">Tesseract reads a torn page, a fine-tuned RoBERTa model reconstructs the words the tear removed,
      and every reconstruction below a validation-locked confidence threshold is marked for a human to check.
      Nothing is presented as observed unless it was.</p>
      <div class="hero-meta">
        <div><div class="k">Held-out documents</div><div class="v">739 processed, 0 errors</div></div>
        <div><div class="k">Character error rate</div><div class="v">4.92% &rarr; 3.00%</div></div>
        <div><div class="k">Word error rate</div><div class="v">5.60% &rarr; 3.83%</div></div>
        <div><div class="k">Output</div><div class="v">One schema-valid JSON per page</div></div>
      </div>
    </div>
    <div class="bento rise">
      <div class="cell"><div class="n">01 / OCR</div><div class="t">Read the damaged page</div>
        <div class="d">Tesseract returns every word it can see, with a box and its own confidence.</div></div>
      <div class="cell"><div class="n">02 / DAMAGE</div><div class="t">Map the tear</div>
        <div class="d">Each word is intact, partial or missing by its overlap with the mask.</div></div>
      <div class="cell"><div class="n">03 / MODEL</div><div class="t">Reconstruct</div>
        <div class="d">RoBERTa fills damaged words, trying one to three subword tokens.</div></div>
      <div class="cell"><div class="n">04 / CONFIDENCE</div><div class="t">Flag doubt</div>
        <div class="d">Geometric-mean score against the locked 0.5442 threshold.</div></div>
      <div class="cell"><div class="n">05 / CONTRACT</div><div class="t">Validate</div>
        <div class="d">JSON Schema checks every page before it is written, with provenance per word.</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_live, tab_eval, tab_schema = st.tabs(["Live pipeline", "Held-out evaluation", "Data contract"])

# ---------------------------------------------------------------- live tab
with tab_live:
    st.write("")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        options = ["Camus sample (paper, Section 5.6.3)"] + load_test_sentences()
        choice = st.selectbox("Sentence from the held-out test split", options, index=0)
        default_text = CLEAN_TEXT if choice.startswith("Camus") else choice
        text = st.text_area("Text rendered as the page", value=default_text, height=96)
    with right:
        mode = st.radio("Damage model", ["Random ribbon tear", "Targeted word"], horizontal=True)
        n_words = len(text.split())
        c1, c2 = st.columns(2)
        seed = c1.number_input("Tear seed", min_value=0, max_value=9999, value=42, step=1)
        target_index = c2.slider("Target word index", 0, max(n_words - 1, 0), min(11, n_words - 1),
                                 disabled=(mode != "Targeted word"))
        run = st.button("Run the full pipeline", type="primary", use_container_width=True)

    if run:
        with st.spinner("Rendering, tearing, reading, reconstructing and validating"):
            st.session_state["result"] = run_pipeline(text, mode, target_index, int(seed))

    res = st.session_state.get("result")
    if res:
        section("01", "Input page and synthetic damage",
                "The page is rendered from known text, then torn with a background-matched mask.")
        st.image(res["clean"], caption="Clean page, ground truth", use_container_width=True)
        st.image(res["overlay"], caption="Damaged page, tear outlined", use_container_width=True)
        z1, z2 = st.columns(2)
        z1.image(res["zoom_clean"], caption="Detail before damage", use_container_width=True)
        z2.image(res["zoom_damaged"], caption="Detail after damage, as OCR sees it", use_container_width=True)

        section("02", "Recovered text",
                "Plain words were read by OCR. Highlighted words were inferred by the model.")
        st.markdown(
            f'<div class="legend"><span><span class="sw" style="background:{GREEN_BG};border:1px solid #BCD3B8"></span>'
            f'Reconstructed, above threshold</span><span><span class="sw" style="background:{RED_BG};'
            f'border:1px solid #EDB9B7"></span>Reconstructed, flagged for review</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(recovered_html(res["records"]), unsafe_allow_html=True)

        doc = res["document"]
        ocr_cer = cer(res["gt_text"], res["ocr_only"])
        ocr_wer = wer(res["gt_text"], res["ocr_only"])
        recon = [r for r in res["records"] if r["provenance"] == "reconstructed"]
        cer_note, cer_tone = delta(doc["evaluation"]["cer"], ocr_cer)
        wer_note, wer_tone = delta(doc["evaluation"]["wer"], ocr_wer)
        stat_row([
            ("Character error rate", f'{doc["evaluation"]["cer"]:.2%}', cer_note, cer_tone),
            ("Word error rate", f'{doc["evaluation"]["wer"]:.2%}', wer_note, wer_tone),
            ("Damaged words", len(recon), "reconstructed by the model", None),
            ("Flagged", sum(1 for r in recon if r["uncertain"]), f"at or below {CONFIDENCE_THRESHOLD}", None),
            ("Exact recoveries", f'{doc["evaluation"]["reconstruction_top1_accuracy"]:.0%}' if recon else "—",
             "against ground truth", None),
        ])

        section("03", "Word-level provenance", "Damaged positions first, then every intact word.")
        table = pd.DataFrame([{
            "#": r["word_index"],
            "ground truth": r["_ground_truth_text"],
            "OCR read": r["ocr_text"] if r["ocr_text"] is not None else "— not detected",
            "damage": r["damage_status"],
            "overlap": r["damage_fraction"],
            "reconstruction": r["reconstructed_text"] or "",
            "confidence": r["reconstruction_confidence"],
            "flagged": r["uncertain"],
            "provenance": r["provenance"],
        } for r in res["records"]])
        st.dataframe(
            pd.concat([table[table["damage"] != "intact"], table[table["damage"] == "intact"]]),
            hide_index=True, use_container_width=True, height=280,
            column_config={
                "overlap": st.column_config.ProgressColumn("mask overlap", min_value=0, max_value=1, format="%.2f"),
                "confidence": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        section("04", "Schema-validated JSON output",
                "Validated inside the pipeline, before anything is written to disk.")
        if res["schema_error"] is None:
            st.markdown(tag("Valid · schema/document_schema.json · draft 2020-12", GREEN_BG, GREEN_FG),
                        unsafe_allow_html=True)
        else:
            st.markdown(tag("Schema violation", RED_BG, RED_FG) + f' <span class="caption-sm">'
                        f'{html.escape(res["schema_error"])}</span>', unsafe_allow_html=True)
        st.write("")
        focus_words = [w for w in doc["words"] if w["provenance"] == "reconstructed"]
        preview = {**{k: v for k, v in doc.items() if k != "words"},
                   "words": focus_words + [f"... {len(doc['words']) - len(focus_words)} intact OCR words omitted in preview"]}
        j1, j2 = st.columns([1.4, 1], gap="large")
        j1.json(preview, expanded=3)
        j2.download_button("Download the full JSON document", json.dumps(doc, indent=2),
                           file_name="demo_0001.json", mime="application/json", use_container_width=True)
        j2.markdown(
            '<p class="caption-sm" style="margin-top:14px">Every reconstructed word carries '
            '<span class="mono">provenance: "reconstructed"</span>, its confidence and an '
            '<span class="mono">uncertain</span> flag. A reader can always tell an observation from a guess.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="note" style="margin-top:22px"><b>Choose a sentence and a damage model, then run the '
                    'pipeline.</b> The first run loads the fine-tuned model, which takes about ten seconds.</div>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- evaluation tab
with tab_eval:
    current, published = load_summaries()
    s = current or published
    if s is None:
        st.markdown('<div class="note">No results found in <span class="mono">results/</span>. Run '
                    '<span class="mono">python scripts/run_held_out_evaluation.py</span> first.</div>',
                    unsafe_allow_html=True)
    else:
        cl = s["corpus_level_cer_wer"]
        ra = s["reconstruction_accuracy"]
        uf = s["uncertainty_flagging_at_locked_threshold"]
        section("01", "All 739 held-out documents",
                f'{s["n_documents_succeeded"]} of {s["n_documents_attempted"]} processed, '
                f'{s["n_documents_errored"]} errors. One targeted torn word per document.')
        cer_note, cer_tone = delta(cl["final_stitched_cer"], cl["ocr_only_cer"])
        wer_note, wer_tone = delta(cl["final_stitched_wer"], cl["ocr_only_wer"])
        stat_row([
            ("Corpus CER", f'{cl["final_stitched_cer"]:.2%}', cer_note, cer_tone),
            ("Corpus WER", f'{cl["final_stitched_wer"]:.2%}', wer_note, wer_tone),
            ("Top-1 accuracy", f'{ra["top1_accuracy_overall"]:.2%}', f'content words {ra["top1_accuracy_content_words"]:.2%}', None),
            ("Error recall", f'{uf["error_recall"]:.2%}', "wrong words that were flagged", None),
            ("Flag precision", f'{uf["precision"]:.2%}', "flagged words that were wrong", None),
        ])

        st.write("")
        c1, c2 = st.columns(2, gap="large")
        err = pd.DataFrame([
            {"metric": "CER", "system": "OCR only", "rate": cl["ocr_only_cer"] * 100},
            {"metric": "CER", "system": "Full pipeline", "rate": cl["final_stitched_cer"] * 100},
            {"metric": "WER", "system": "OCR only", "rate": cl["ocr_only_wer"] * 100},
            {"metric": "WER", "system": "Full pipeline", "rate": cl["final_stitched_wer"] * 100},
        ])
        order = ["OCR only", "Full pipeline"]
        with c1:
            st.markdown('<div class="chart-title">Corpus-level error rate, %</div>', unsafe_allow_html=True)
            bars = alt.Chart(err).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=46).encode(
                x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("system:N", sort=order),
                y=alt.Y("rate:Q", title=None),
                color=alt.Color("system:N", scale=alt.Scale(domain=order, range=["#D3D1CB", INK]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["system", alt.Tooltip("rate:Q", format=".2f")],
            )
            labels = alt.Chart(err).mark_text(dy=-8, fontSize=12).encode(
                x="metric:N", xOffset=alt.XOffset("system:N", sort=order), y="rate:Q",
                text=alt.Text("rate:Q", format=".2f"),
            )
            st.altair_chart(altair_theme((bars + labels).properties(height=270)), use_container_width=True)
        flag = pd.DataFrame([
            {"metric": "Error recall", "value": uf["error_recall"] * 100},
            {"metric": "Precision", "value": uf["precision"] * 100},
            {"metric": "Flag rate", "value": uf["flag_rate"] * 100},
            {"metric": "Unflagged error rate", "value": uf["unflagged_error_rate"] * 100},
        ])
        with c2:
            st.markdown(f'<div class="chart-title">Uncertainty flag at the locked threshold ({CONFIDENCE_THRESHOLD}), %</div>',
                        unsafe_allow_html=True)
            hb = alt.Chart(flag).mark_bar(cornerRadiusEnd=3, color=INK, size=26).encode(
                y=alt.Y("metric:N", title=None, sort=None, axis=alt.Axis(labelLimit=200)),
                x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                tooltip=["metric", alt.Tooltip("value:Q", format=".2f")],
            )
            hl = alt.Chart(flag).mark_text(align="left", dx=6, fontSize=12).encode(
                y=alt.Y("metric:N", sort=None), x="value:Q", text=alt.Text("value:Q", format=".2f"),
            )
            st.altair_chart(altair_theme((hb + hl).properties(height=270)), use_container_width=True)

        section("02", "Where the confidence layer helps",
                "Wrong reconstructions cluster below the dashed threshold, which is why the flag is useful.")
        df = load_held_out_table()
        c3, c4 = st.columns([1, 1.25], gap="large")
        with c3:
            st.markdown('<div class="chart-title">Confidence of targeted reconstructions</div>', unsafe_allow_html=True)
            hist = alt.Chart(df.assign(outcome=df["correct"].map({True: "correct", False: "wrong"}))).mark_bar(
                opacity=.95, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="reconstruction confidence"),
                y=alt.Y("count():Q", stack=True, title="documents"),
                color=alt.Color("outcome:N", scale=alt.Scale(domain=["correct", "wrong"], range=["#8DB38A", "#D99A97"]),
                                legend=alt.Legend(orient="top", title=None)),
            )
            rule = alt.Chart(pd.DataFrame({"t": [CONFIDENCE_THRESHOLD]})).mark_rule(
                strokeDash=[4, 4], color="#111", strokeWidth=1).encode(x="t:Q")
            st.altair_chart(altair_theme((hist + rule).properties(height=290)), use_container_width=True)
        with c4:
            st.markdown('<div class="chart-title">Browse held-out documents</div>', unsafe_allow_html=True)
            show = st.radio("Filter", ["All", "Flagged", "Wrong but unflagged", "Correct"], horizontal=True,
                            label_visibility="collapsed")
            view = df
            if show == "Flagged":
                view = df[df["flagged"]]
            elif show == "Wrong but unflagged":
                view = df[~df["flagged"] & ~df["correct"]]
            elif show == "Correct":
                view = df[df["correct"]]
            st.dataframe(view[["document", "target word", "reconstructed", "confidence", "flagged", "correct"]],
                         hide_index=True, use_container_width=True, height=262)

        if published and current and published != current:
            section("03", "Scoring erratum", "Published aggregation compared with the corrected one.")
            st.markdown(
                '<p class="caption-sm" style="max-width:820px">The published aggregation scored the last damaged word '
                'in each document. In 17 of 739 documents the padded tear also clipped a neighbouring word, so the '
                'neighbour was scored instead of the targeted word. Per-document outputs are unchanged; only these '
                'aggregate figures move.</p>',
                unsafe_allow_html=True,
            )
            pu = published["uncertainty_flagging_at_locked_threshold"]
            pr = published["reconstruction_accuracy"]
            st.dataframe(pd.DataFrame([
                {"metric": "Top-1 accuracy", "published": pr["top1_accuracy_overall"], "corrected": ra["top1_accuracy_overall"]},
                {"metric": "Content-word accuracy", "published": pr["top1_accuracy_content_words"], "corrected": ra["top1_accuracy_content_words"]},
                {"metric": "Error recall", "published": pu["error_recall"], "corrected": uf["error_recall"]},
                {"metric": "Precision", "published": pu["precision"], "corrected": uf["precision"]},
                {"metric": "Flag rate", "published": pu["flag_rate"], "corrected": uf["flag_rate"]},
                {"metric": "Unflagged error rate", "published": pu["unflagged_error_rate"], "corrected": uf["unflagged_error_rate"]},
            ]), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- schema tab
with tab_schema:
    schema = load_schema()
    section("01", "The integration contract",
            "Every page the pipeline produces is validated against this JSON Schema before it is written.")
    a, b = st.columns([1, 1.15], gap="large")
    with a:
        st.markdown(
            '<div class="note" style="line-height:1.7"><b>Intact word</b><br>'
            '<span class="mono">provenance: "ocr"</span>, all reconstruction fields <span class="mono">null</span>.'
            '<br><br><b>Partial or missing word</b><br><span class="mono">provenance: "reconstructed"</span>, with '
            '<span class="mono">reconstructed_text</span>, <span class="mono">reconstruction_confidence</span> and '
            '<span class="mono">uncertain</span> all required.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-title" style="margin-top:28px">Try breaking it</div>', unsafe_allow_html=True)
        broken = st.checkbox("Mark an intact word as reconstructed", value=True)
        sample = json.load(open(os.path.join(paths.RESULTS_DIR, "camus_sample_001_output.json"), encoding="utf-8"))
        if broken:
            sample["words"][0]["provenance"] = "reconstructed"
        try:
            jsonschema.validate(instance=sample, schema=schema)
            st.markdown(tag("Valid document", GREEN_BG, GREEN_FG), unsafe_allow_html=True)
        except jsonschema.ValidationError as e:
            st.markdown(tag("Rejected by the schema", RED_BG, RED_FG), unsafe_allow_html=True)
            st.code(f"ValidationError: {e.message}\npath: {list(e.absolute_path)}", language="text")
    with b:
        st.markdown('<div class="chart-title">Conditional rules, <span class="mono">$defs.word.allOf</span></div>',
                    unsafe_allow_html=True)
        st.json(schema["$defs"]["word"]["allOf"], expanded=True)
        with st.expander("Full schema/document_schema.json"):
            st.json(schema, expanded=1)
