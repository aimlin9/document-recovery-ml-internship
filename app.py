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
from PIL import Image, ImageDraw, ImageOps

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

st.set_page_config(page_title="Document Recovery Pipeline", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      .hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
      .hero p {color: #5b6474; margin-top: 0; font-size: 1.02rem;}
      .stage {border: 1px solid #e4e7ec; border-radius: 10px; padding: .9rem 1rem; background: #fbfcfd; height: 100%;}
      .stage .n {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: #3b5bdb; font-weight: 700;}
      .stage .t {font-weight: 600; margin: .15rem 0 .25rem;}
      .stage .d {font-size: .85rem; color: #5b6474;}
      .recovered {font-size: 1.15rem; line-height: 2.3rem; padding: 1rem 1.2rem; border-radius: 10px;
                  background: #fff; border: 1px solid #e4e7ec;}
      .w {padding: .15rem .35rem; border-radius: 6px; margin: 0 .05rem;}
      .w-ocr {background: transparent;}
      .w-ok {background: #d3f9d8; border: 1px solid #8ce99a;}
      .w-flag {background: #ffe3e3; border: 1px solid #ffa8a8;}
      .legend span {margin-right: 1rem; font-size: .85rem;}
      .badge-ok {display:inline-block; background:#d3f9d8; color:#2b8a3e; border-radius:999px; padding:.2rem .7rem; font-weight:600; font-size:.85rem;}
      .badge-bad {display:inline-block; background:#ffe3e3; color:#c92a2a; border-radius:999px; padding:.2rem .7rem; font-weight:600; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading fine-tuned RoBERTa model...")
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
    """Damaged image with the tear outlined in red, so viewers can see where it is."""
    base = img.convert("RGBA")
    tinted = Image.alpha_composite(base, Image.new("RGBA", base.size, (224, 49, 49, 60)))
    out = Image.composite(tinted, base, mask)
    draw = ImageDraw.Draw(out)
    bbox = mask.getbbox()
    if bbox:
        draw.rectangle(bbox, outline=(224, 49, 49, 255), width=2)
    return out.convert("RGB")


def zoom_on_damage(img, mask, context_px=170, scale=3):
    """Crop around the tear with some surrounding text, then upscale, so the
    damage is visible on a one-line rendered page."""
    bbox = mask.getbbox() or (0, 0, img.width, img.height)
    left = max(0, bbox[0] - context_px)
    right = min(img.width, bbox[2] + context_px)
    crop = img.crop((left, 0, right, img.height))
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    return ImageOps.expand(crop, border=3, fill=(206, 212, 218))


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
        t = html.escape(r["final_text"] or "∅")
        if r["provenance"] == "ocr":
            parts.append(f'<span class="w w-ocr">{t}</span>')
        else:
            cls = "w-flag" if r["uncertain"] else "w-ok"
            tip = f'confidence {r["reconstruction_confidence"]:.3f}'
            parts.append(f'<span class="w {cls}" title="{tip}">{t}</span>')
    return '<div class="recovered">' + " ".join(parts) + "</div>"


# ---------------------------------------------------------------- header
tokenizer_model_ok = os.path.isfile(os.path.join(paths.FINE_TUNED_MODEL_PATH, "config.json"))

st.markdown(
    '<div class="hero"><h1>📄 Recovering Structure From Damaged Documents</h1>'
    "<p>OCR + synthetic damage modelling + fine-tuned RoBERTa reconstruction, with a calibrated "
    "confidence layer that flags uncertain words for human review.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Project")
    st.markdown(
        "**ML-T2-022** · LearnDepth Advanced ML Internship (Track 2)\n\n"
        "**Author:** Gyimah Ramsey Opoku  \n"
        "BSc Computer Science, KNUST"
    )
    st.divider()
    st.markdown("### Runtime")
    st.markdown(
        f"Model: {'✅ fine-tuned RoBERTa' if tokenizer_model_ok else '⚠️ roberta-base fallback'}  \n"
        f"Locked threshold: **{CONFIDENCE_THRESHOLD}**  \n"
        "OCR: Tesseract (`--psm 6`)"
    )
    if not tokenizer_model_ok:
        st.warning("Fine-tuned weights not found in `roberta-finetuned-final/`. "
                   "Confidence flags are not calibrated for the fallback model.")
    st.divider()
    st.caption("Repository: github.com/aimlin9/document-recovery-ml-internship")

stage_cols = st.columns(5)
stages = [
    ("Stage 1", "OCR", "Tesseract reads the damaged page"),
    ("Stage 2", "Damage map", "Each word classed intact / partial / missing"),
    ("Stage 3", "Reconstruct", "RoBERTa fills damaged words (1–3 subwords)"),
    ("Stage 4", "Confidence", "Geometric-mean score vs locked 0.5442"),
    ("Stage 5", "Contract", "Schema-validated JSON with provenance"),
]
for col, (n, t, d) in zip(stage_cols, stages):
    col.markdown(f'<div class="stage"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>',
                 unsafe_allow_html=True)

st.write("")
tab_live, tab_eval, tab_schema = st.tabs(["🔬 Live pipeline", "📊 Held-out evaluation (n = 739)", "🧾 Data contract"])

# ---------------------------------------------------------------- live tab
with tab_live:
    left, right = st.columns([1.05, 1])
    with left:
        options = ["Camus sample (paper, Section 5.6.3)"] + load_test_sentences()
        choice = st.selectbox("Sentence (held-out test split, never seen in fine-tuning)", options, index=0)
        default_text = CLEAN_TEXT if choice.startswith("Camus") else choice
        text = st.text_area("Source text to render as a page", value=default_text, height=90)
    with right:
        mode = st.radio("Damage model", ["Random ribbon tear", "Targeted word"], horizontal=True)
        n_words = len(text.split())
        c1, c2 = st.columns(2)
        seed = c1.number_input("Tear seed", min_value=0, max_value=9999, value=42, step=1)
        target_index = c2.slider("Target word index", 0, max(n_words - 1, 0), min(11, n_words - 1),
                                 disabled=(mode != "Targeted word"))
        run = st.button("▶ Run full pipeline", type="primary", use_container_width=True)

    if run:
        with st.spinner("Rendering → tearing → OCR → classifying → reconstructing → validating..."):
            st.session_state["result"] = run_pipeline(text, mode, target_index, int(seed))

    res = st.session_state.get("result")
    if res:
        st.markdown("#### 1 · Input page and synthetic damage")
        st.image(res["clean"], caption="Clean rendered page (ground truth)", use_container_width=True)
        st.image(res["overlay"], caption="Damaged page — tear region outlined in red", use_container_width=True)
        z1, z2 = st.columns(2)
        z1.image(res["zoom_clean"], caption="Zoom: before damage", use_container_width=True)
        z2.image(res["zoom_damaged"], caption="Zoom: after damage (what OCR actually sees)", use_container_width=True)

        st.markdown("#### 2 · Recovered text")
        st.markdown(
            '<div class="legend"><span>plain = OCR-observed</span>'
            '<span><span class="w w-ok">green</span> reconstructed, confident</span>'
            '<span><span class="w w-flag">red</span> reconstructed, flagged for review</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(recovered_html(res["records"]), unsafe_allow_html=True)

        doc = res["document"]
        ocr_cer = cer(res["gt_text"], res["ocr_only"])
        ocr_wer = wer(res["gt_text"], res["ocr_only"])
        recon = [r for r in res["records"] if r["provenance"] == "reconstructed"]
        m = st.columns(5)
        m[0].metric("CER", f'{doc["evaluation"]["cer"]:.2%}', f'{doc["evaluation"]["cer"] - ocr_cer:+.2%} vs OCR-only',
                    delta_color="inverse")
        m[1].metric("WER", f'{doc["evaluation"]["wer"]:.2%}', f'{doc["evaluation"]["wer"] - ocr_wer:+.2%} vs OCR-only',
                    delta_color="inverse")
        m[2].metric("Damaged words", len(recon))
        m[3].metric("Flagged uncertain", sum(1 for r in recon if r["uncertain"]))
        m[4].metric("Exact recoveries", f'{doc["evaluation"]["reconstruction_top1_accuracy"]:.0%}' if recon else "—")

        st.markdown("#### 3 · Word-level provenance")
        table = pd.DataFrame([{
            "#": r["word_index"],
            "ground truth": r["_ground_truth_text"],
            "OCR read": r["ocr_text"] if r["ocr_text"] is not None else "∅ (not detected)",
            "damage": r["damage_status"],
            "overlap": r["damage_fraction"],
            "reconstruction": r["reconstructed_text"] or "",
            "confidence": r["reconstruction_confidence"],
            "uncertain": r["uncertain"],
            "provenance": r["provenance"],
        } for r in res["records"]])
        focus = table[table["damage"] != "intact"]
        st.dataframe(
            pd.concat([focus, table[table["damage"] == "intact"]]),
            hide_index=True, use_container_width=True, height=280,
            column_config={
                "overlap": st.column_config.ProgressColumn("mask overlap", min_value=0, max_value=1, format="%.2f"),
                "confidence": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        st.markdown("#### 4 · Schema-validated JSON output")
        if res["schema_error"] is None:
            st.markdown('<span class="badge-ok">✓ Valid against schema/document_schema.json (JSON Schema 2020-12)</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-bad">✗ Schema violation: {html.escape(res["schema_error"])}</span>',
                        unsafe_allow_html=True)
        st.write("")
        focus_words = [w for w in doc["words"] if w["provenance"] == "reconstructed"]
        preview = {**{k: v for k, v in doc.items() if k != "words"},
                   "words": focus_words + [f"... {len(doc['words']) - len(focus_words)} intact OCR words omitted in preview"]}
        j1, j2 = st.columns([1.4, 1])
        j1.json(preview, expanded=3)
        j2.download_button("⬇ Download full JSON document", json.dumps(doc, indent=2),
                           file_name="demo_0001.json", mime="application/json", use_container_width=True)
        j2.markdown(
            "Every reconstructed word carries `provenance: \"reconstructed\"`, its confidence and an "
            "`uncertain` flag, so a reviewer can always tell an observation from a guess."
        )
    else:
        st.info("Choose a sentence and damage model, then press **Run full pipeline**. "
                "The first run loads the fine-tuned model (~10 s).")

# ---------------------------------------------------------------- evaluation tab
with tab_eval:
    current, published = load_summaries()
    s = current or published
    if s is None:
        st.warning("No results found in results/. Run `python scripts/run_held_out_evaluation.py` first.")
    else:
        cl = s["corpus_level_cer_wer"]
        ra = s["reconstruction_accuracy"]
        uf = s["uncertainty_flagging_at_locked_threshold"]
        st.markdown(f'**{s["n_documents_succeeded"]} / {s["n_documents_attempted"]}** held-out documents processed, '
                    f'**{s["n_documents_errored"]}** errors · one targeted torn word per document')
        k = st.columns(4)
        k[0].metric("Corpus CER", f'{cl["final_stitched_cer"]:.2%}', f'{cl["final_stitched_cer"] - cl["ocr_only_cer"]:+.2%} vs OCR',
                    delta_color="inverse")
        k[1].metric("Corpus WER", f'{cl["final_stitched_wer"]:.2%}', f'{cl["final_stitched_wer"] - cl["ocr_only_wer"]:+.2%} vs OCR',
                    delta_color="inverse")
        k[2].metric("Error recall @0.5442", f'{uf["error_recall"]:.2%}')
        k[3].metric("Flag precision", f'{uf["precision"]:.2%}')

        c1, c2 = st.columns(2)
        err = pd.DataFrame([
            {"metric": "CER", "system": "OCR only", "rate": cl["ocr_only_cer"] * 100},
            {"metric": "CER", "system": "Full pipeline", "rate": cl["final_stitched_cer"] * 100},
            {"metric": "WER", "system": "OCR only", "rate": cl["ocr_only_wer"] * 100},
            {"metric": "WER", "system": "Full pipeline", "rate": cl["final_stitched_wer"] * 100},
        ])
        c1.markdown("**Corpus-level error rate (%)**")
        c1.altair_chart(
            alt.Chart(err).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("system:N", sort=["OCR only", "Full pipeline"]),
                y=alt.Y("rate:Q", title="%"),
                color=alt.Color("system:N", scale=alt.Scale(domain=["OCR only", "Full pipeline"],
                                                            range=["#adb5bd", "#3b5bdb"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["system", alt.Tooltip("rate:Q", format=".2f")],
            ).properties(height=260)
            + alt.Chart(err).mark_text(dy=-8, fontSize=12).encode(
                x="metric:N", xOffset=alt.XOffset("system:N", sort=["OCR only", "Full pipeline"]),
                y="rate:Q", text=alt.Text("rate:Q", format=".2f"),
            ),
            use_container_width=True,
        )
        flag = pd.DataFrame([
            {"metric": "Error recall", "value": uf["error_recall"] * 100},
            {"metric": "Precision", "value": uf["precision"] * 100},
            {"metric": "Flag rate", "value": uf["flag_rate"] * 100},
            {"metric": "Unflagged error rate", "value": uf["unflagged_error_rate"] * 100},
        ])
        c2.markdown("**Uncertainty flagging at locked threshold (%)**")
        c2.altair_chart(
            alt.Chart(flag).mark_bar(cornerRadiusEnd=4, color="#3b5bdb").encode(
                y=alt.Y("metric:N", title=None, sort=None, axis=alt.Axis(labelLimit=200)),
                x=alt.X("value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                tooltip=["metric", alt.Tooltip("value:Q", format=".2f")],
            ).properties(height=260)
            + alt.Chart(flag).mark_text(align="left", dx=5, fontSize=12).encode(
                y=alt.Y("metric:N", sort=None), x="value:Q", text=alt.Text("value:Q", format=".2f"),
            ),
            use_container_width=True,
        )

        df = load_held_out_table()
        c3, c4 = st.columns([1, 1.3])
        c3.markdown("**Confidence distribution of targeted reconstructions**")
        c3.altair_chart(
            alt.Chart(df.assign(outcome=df["correct"].map({True: "correct", False: "wrong"}))).mark_bar(opacity=.85).encode(
                x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="reconstruction confidence"),
                y=alt.Y("count():Q", stack=True, title="documents"),
                color=alt.Color("outcome:N", scale=alt.Scale(domain=["correct", "wrong"], range=["#40c057", "#fa5252"])),
            ).properties(height=260)
            + alt.Chart(pd.DataFrame({"t": [CONFIDENCE_THRESHOLD]})).mark_rule(strokeDash=[5, 4], color="#212529").encode(x="t:Q"),
            use_container_width=True,
        )
        with c4:
            st.markdown("**Browse held-out documents**")
            show = st.radio("Filter", ["All", "Flagged", "Wrong but unflagged", "Correct"], horizontal=True)
            view = df
            if show == "Flagged":
                view = df[df["flagged"]]
            elif show == "Wrong but unflagged":
                view = df[~df["flagged"] & ~df["correct"]]
            elif show == "Correct":
                view = df[df["correct"]]
            st.dataframe(view[["document", "target word", "reconstructed", "confidence", "flagged", "correct"]],
                         hide_index=True, use_container_width=True, height=225)

        if published and current and published != current:
            with st.expander("Scoring erratum: published vs corrected aggregation"):
                st.markdown(
                    "The published aggregation scored the *last* damaged word in each document. In 17 of 739 "
                    "documents the padded tear also clipped a neighbouring word, so the neighbour was scored "
                    "instead of the targeted word. Per-document outputs are unchanged; only the aggregate differs."
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
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown("#### The integration contract")
        st.markdown(
            "Every document the pipeline produces is validated against this JSON Schema (draft 2020-12) "
            "*inside* the pipeline, before it is written to disk. Two conditional rules encode the core logic:"
        )
        st.markdown(
            "- **intact** word → `provenance: \"ocr\"`, reconstruction fields `null`\n"
            "- **partial / missing** word → `provenance: \"reconstructed\"`, with `reconstructed_text`, "
            "`reconstruction_confidence` and `uncertain` all required"
        )
        st.markdown("#### Try breaking it")
        broken = st.checkbox("Mark an intact word as reconstructed (should be rejected)", value=True)
        sample = json.load(open(os.path.join(paths.RESULTS_DIR, "camus_sample_001_output.json"), encoding="utf-8"))
        if broken:
            sample["words"][0]["provenance"] = "reconstructed"
        try:
            jsonschema.validate(instance=sample, schema=schema)
            st.markdown('<span class="badge-ok">✓ Document is valid</span>', unsafe_allow_html=True)
        except jsonschema.ValidationError as e:
            st.markdown('<span class="badge-bad">✗ Rejected by the schema</span>', unsafe_allow_html=True)
            st.code(f"ValidationError: {e.message}\npath: {list(e.absolute_path)}", language="text")
    with b:
        st.markdown("#### Conditional rules (`$defs.word.allOf`)")
        st.json(schema["$defs"]["word"]["allOf"], expanded=True)
        with st.expander("Full schema/document_schema.json"):
            st.json(schema, expanded=1)
