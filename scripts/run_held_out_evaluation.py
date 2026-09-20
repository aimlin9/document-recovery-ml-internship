"""
Step 5 batch validation: run the full image-based integration pipeline
(render -> targeted tear -> OCR -> damage classification -> reconstruction
-> confidence/uncertain flag -> schema-validated JSON) across every
held-out test sentence, and produce an aggregate report.

DENOMINATOR / MANIFEST: the held-out set used here is whatever
sentence_corpus.jsonl currently contains under split == "test" on this
machine, right now - not a remembered number from earlier in the
project. This script hashes that file and records the exact count into
held_out_manifest.json, so the final report cites a file-verified
number. Do not hand-edit that count anywhere else.

METHODOLOGY NOTE (read before comparing numbers to Step 4): Step 4's
1,478-example benchmark was pure text - two words masked per sentence,
no image, no OCR, no tear. This script instead renders each held-out
sentence to an image and deliberately damages exactly ONE randomly
chosen word per sentence using word_targeted_masks.build_targeted_mask
(the same targeting approach validated in Step 2/3), so that every
document contributes exactly one reconstruction attempt and none are
wasted on a random tear that happens to miss all text. That makes N
here equal to the number of held-out SENTENCES, not 1,478 - a
different, smaller denominator than Step 4's, by design. Don't compare
the two accuracy numbers as if they were the same measurement.

The confidence threshold (0.5442) is NOT re-selected here. It was
locked in Step 4 on a text-only validation set. This run only APPLIES
it and reports how it performs against a different distribution of
damage (real image + OCR noise, not pure text masking) - a genuine
generalization check, not a re-tuning.
"""

import hashlib
import json
import os
import random
import time

import pytesseract
from pytesseract import Output
from PIL import Image
from jiwer import cer, wer

from baseline_ocr import render_clean_image
from torn_regions import apply_tear, estimate_background_color
from word_targeted_masks import get_word_boxes, build_targeted_mask
from reconstruction import classify_all_words
from build_document_json import (
    load_reconstruction_model, match_ocr_word, reconstruct_word,
    CONFIDENCE_THRESHOLD, FUNCTION_WORDS,
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CORPUS_PATH = "sentence_corpus.jsonl"
OUTPUT_DIR = "held_out_documents"
TEMP_IMAGE_DIR = "held_out_temp_images"
RESULTS_JSONL = "held_out_results.jsonl"          # one line per document, written incrementally
SUMMARY_PATH = "held_out_summary_report.json"
MANIFEST_PATH = "held_out_manifest.json"

# Set to a small integer (e.g. 20) for a smoke test before committing to
# the full run. None = process every held-out sentence.
LIMIT = None


def load_held_out_sentences():
    with open(CORPUS_PATH, "rb") as f:
        raw = f.read()
    file_hash = hashlib.sha256(raw).hexdigest()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    test_records = [r for r in records if r["split"] == "test"]

    manifest = {
        "corpus_file": CORPUS_PATH,
        "corpus_file_sha256": file_hash,
        "total_records_in_file": len(records),
        "held_out_test_count": len(test_records),
        "note": "This count is read from the file at run time, not hardcoded. "
                "If this number ever differs from a previously reported count, "
                "trust this one and flag the discrepancy explicitly rather than "
                "silently averaging or picking whichever number is convenient.",
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {MANIFEST_PATH}: {len(test_records)} held-out test sentences "
          f"(corpus sha256 {file_hash[:16]}...)")
    return test_records


def error_recall_at_threshold(results, threshold):
    wrong = [r for r in results if not r["is_correct"]]
    if not wrong:
        return None
    return sum(1 for r in wrong if r["confidence"] <= threshold) / len(wrong)


def precision_at_threshold(results, threshold):
    flagged = [r for r in results if r["confidence"] <= threshold]
    if not flagged:
        return None
    return sum(1 for r in flagged if not r["is_correct"]) / len(flagged)


def unflagged_error_rate(results, threshold):
    unflagged = [r for r in results if r["confidence"] > threshold]
    if not unflagged:
        return None
    return sum(1 for r in unflagged if not r["is_correct"]) / len(unflagged)


def process_one_sentence(doc_index, sentence_record, tokenizer, model):
    text = sentence_record["text"] if "text" in sentence_record else sentence_record.get("sentence")
    document_id = f"test_{doc_index:04d}"

    clean_path = os.path.join(TEMP_IMAGE_DIR, f"{document_id}_clean.png")
    damaged_path = os.path.join(TEMP_IMAGE_DIR, f"{document_id}_damaged.png")

    render_clean_image(text, clean_path)
    clean_img = Image.open(clean_path).convert("RGB")
    width, height = clean_img.size
    bg_color = estimate_background_color(clean_img)

    ground_truth_words = get_word_boxes(clean_path)
    if len(ground_truth_words) < 2:
        raise ValueError(f"Too few words detected ({len(ground_truth_words)}) - skipping")
    ground_truth_text = " ".join(w["text"] for w in ground_truth_words)

    # Deliberately damage exactly one word, chosen reproducibly per document.
    rng = random.Random(f"held_out_target_{document_id}")
    target_index = rng.randrange(len(ground_truth_words))
    mask = build_targeted_mask(width, height, [ground_truth_words[target_index]], seed=doc_index)

    damaged_img = apply_tear(clean_img, mask, bg_color)
    damaged_img.save(damaged_path)

    classified_words = classify_all_words(ground_truth_words, mask)

    ocr_data = pytesseract.image_to_data(damaged_img, config="--psm 6", output_type=Output.DICT)
    ocr_words = []
    for i in range(len(ocr_data["text"])):
        t = ocr_data["text"][i].strip()
        conf = float(ocr_data["conf"][i]) if ocr_data["conf"][i] not in ("-1", "") else -1
        if t and conf >= 0:
            ocr_words.append({
                "text": t, "left": ocr_data["left"][i], "top": ocr_data["top"][i],
                "width": ocr_data["width"][i], "height": ocr_data["height"][i], "conf": conf,
            })

    word_records = []
    for idx, w in enumerate(classified_words):
        matched = match_ocr_word(w, ocr_words)
        word_records.append({
            "word_index": idx,
            "bounding_box": {"left": w["left"], "top": w["top"], "width": w["width"], "height": w["height"]},
            "ocr_text": matched["text"] if matched else None,
            "ocr_confidence": matched["conf"] if matched else None,
            "damage_status": w["damage_status"],
            "damage_fraction": round(w["damage_fraction"], 4),
            "reconstructed_text": None,
            "reconstruction_confidence": None,
            "uncertain": None,
            "final_text": matched["text"] if matched else "",
            "provenance": "ocr",
            "_ground_truth_text": w["text"],
        })

    context_words = [r["final_text"] or "[UNK]" for r in word_records]

    reconstruction_record = None  # the single targeted word's outcome, for aggregation
    for idx, r in enumerate(word_records):
        if r["damage_status"] == "intact":
            continue
        reconstructed_text, confidence = reconstruct_word(tokenizer, model, context_words, idx)
        r["reconstructed_text"] = reconstructed_text
        r["reconstruction_confidence"] = round(confidence, 4)
        r["uncertain"] = confidence <= CONFIDENCE_THRESHOLD
        r["final_text"] = reconstructed_text
        r["provenance"] = "reconstructed"

        gt = r["_ground_truth_text"].strip(",.;:").lower()
        pred = reconstructed_text.strip(",.;:").lower()
        reconstruction_record = {
            "document_id": document_id,
            "target_word": r["_ground_truth_text"],
            "predicted_word": reconstructed_text,
            "is_correct": pred == gt,
            "is_content_word": gt not in FUNCTION_WORDS,
            "confidence": confidence,
        }

    ocr_only_text = " ".join(r["ocr_text"] for r in word_records if r["ocr_text"])
    final_text = " ".join(r["final_text"] for r in word_records if r["final_text"])

    document = {
        "document_id": document_id,
        "source_image_path": clean_path,
        "page_number": 1,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "final_text": final_text,
        "words": [{k: v for k, v in r.items() if not k.startswith("_")} for r in word_records],
        "evaluation": {
            "ground_truth_text": ground_truth_text,
            "cer": round(cer(ground_truth_text, final_text), 4),
            "wer": round(wer(ground_truth_text, final_text), 4),
            "reconstruction_top1_accuracy": 1.0 if (reconstruction_record and reconstruction_record["is_correct"]) else 0.0,
            "content_word_accuracy": (
                (1.0 if reconstruction_record["is_correct"] else 0.0)
                if reconstruction_record and reconstruction_record["is_content_word"]
                else None
            ),
        },
    }

    return document, ground_truth_text, ocr_only_text, final_text, word_records, reconstruction_record


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

    import jsonschema
    schema = json.load(open("document_schema.json"))

    held_out = load_held_out_sentences()
    if LIMIT is not None:
        held_out = held_out[:LIMIT]
        print(f"LIMIT set: processing only the first {LIMIT} of the held-out set.")

    tokenizer, model = load_reconstruction_model()

    ground_truths, ocr_only_texts, final_texts = [], [], []
    reconstruction_results = []
    total_words, total_reconstructed = 0, 0
    errors = []

    results_file = open(RESULTS_JSONL, "w", encoding="utf-8")
    start = time.time()

    for i, sentence_record in enumerate(held_out):
        try:
            document, gt_text, ocr_text, final_text, word_records, recon_record = process_one_sentence(
                i, sentence_record, tokenizer, model
            )
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
            print(f"[{i+1}/{len(held_out)}] ERROR: {e}")
            continue

        jsonschema.validate(instance=document, schema=schema)

        with open(os.path.join(OUTPUT_DIR, f"{document['document_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
        results_file.write(json.dumps(document) + "\n")
        results_file.flush()

        ground_truths.append(gt_text)
        ocr_only_texts.append(ocr_text)
        final_texts.append(final_text)
        total_words += len(word_records)
        total_reconstructed += sum(1 for r in word_records if r["provenance"] == "reconstructed")
        if recon_record:
            reconstruction_results.append(recon_record)

        if (i + 1) % 25 == 0 or (i + 1) == len(held_out):
            elapsed = time.time() - start
            print(f"[{i+1}/{len(held_out)}] processed, {elapsed:.0f}s elapsed, {len(errors)} errors so far")

    results_file.close()

    # Corpus-level (pooled, not averaged-per-document) CER/WER.
    ocr_only_cer = cer(ground_truths, ocr_only_texts)
    ocr_only_wer = wer(ground_truths, ocr_only_texts)
    final_cer = cer(ground_truths, final_texts)
    final_wer = wer(ground_truths, final_texts)

    n_recon = len(reconstruction_results)
    top1_correct = sum(1 for r in reconstruction_results if r["is_correct"])
    content_results = [r for r in reconstruction_results if r["is_content_word"]]
    content_correct = sum(1 for r in content_results if r["is_correct"])

    recall = error_recall_at_threshold(reconstruction_results, CONFIDENCE_THRESHOLD)
    precision = precision_at_threshold(reconstruction_results, CONFIDENCE_THRESHOLD)
    unflagged_err = unflagged_error_rate(reconstruction_results, CONFIDENCE_THRESHOLD)
    flag_rate = sum(1 for r in reconstruction_results if r["confidence"] <= CONFIDENCE_THRESHOLD) / n_recon if n_recon else None

    summary = {
        "n_documents_attempted": len(held_out),
        "n_documents_succeeded": len(ground_truths),
        "n_documents_errored": len(errors),
        "errors": errors,
        "denominator_note": "N for reconstruction/flagging metrics below is n_documents_succeeded "
                             "(one targeted damaged word per document), NOT Step 4's 1,478 text-only examples.",
        "corpus_level_cer_wer": {
            "ocr_only_cer": round(ocr_only_cer, 4),
            "ocr_only_wer": round(ocr_only_wer, 4),
            "final_stitched_cer": round(final_cer, 4),
            "final_stitched_wer": round(final_wer, 4),
        },
        "reconstruction_accuracy": {
            "n": n_recon,
            "top1_accuracy_overall": round(top1_correct / n_recon, 4) if n_recon else None,
            "n_content_words": len(content_results),
            "top1_accuracy_content_words": round(content_correct / len(content_results), 4) if content_results else None,
        },
        "uncertainty_flagging_at_locked_threshold": {
            "threshold": CONFIDENCE_THRESHOLD,
            "flag_rate": round(flag_rate, 4) if flag_rate is not None else None,
            "error_recall": round(recall, 4) if recall is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
            "unflagged_error_rate": round(unflagged_err, 4) if unflagged_err is not None else None,
        },
        "provenance_counts": {
            "total_words": total_words,
            "ocr_provenance": total_words - total_reconstructed,
            "reconstructed_provenance": total_reconstructed,
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPer-document JSONs written to {OUTPUT_DIR}/")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
