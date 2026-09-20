"""
Step 5: end-to-end document reconstruction pipeline.

Stitches every prior stage into one per-page JSON output matching
document_schema.json:
  Step 1 - OCR (pytesseract) on the damaged image
  Step 2 - damage mask + per-word damage classification
  Step 3/4 - masked-word reconstruction with the fine-tuned RoBERTa model
  Step 4 - confidence scoring (geometric mean) + uncertain flag at the
           locked threshold
This file's own job is the stitching. It does not re-implement OCR,
mask generation, or damage classification - it imports and calls the
existing functions from torn_regions.py, word_targeted_masks.py, and
reconstruction.py.

SCOPE NOTE (same as document_schema.json): word positions and the
damage mask both come from the synthetic-benchmark setup, where the
clean source image and the tear are both known because this pipeline
generated them. Real deployment on a genuinely damaged scan, with no
known mask and no clean reference, needs a damage-region detector this
project has not built - that gap is not solved here, only documented.

TWO KNOWN SIMPLIFICATIONS in this specific stitching step (see inline
comments at the point each applies):
  1. Target subword length is not known in real use (no ground truth
     word to tokenize), so reconstruction tries 1-3 mask tokens and
     keeps whichever gives the highest confidence, instead of Step 4's
     benchmark evaluation, which knew the true count in advance.
  2. Adjacent damaged words are reconstructed independently; a damaged
     word's neighbor, if also damaged, contributes a placeholder rather
     than its own reconstruction as context. Every test image through
     Step 4 had exactly one damaged word per sentence, so this path has
     not been exercised.
"""

import json
import math
import os

import pytesseract
from pytesseract import Output
from PIL import Image
from jiwer import cer, wer

from torn_regions import apply_tear, estimate_background_color, generate_ribbon_tear
from word_targeted_masks import get_word_boxes
from reconstruction import classify_all_words

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Locked in Step 4 on the validation set. Do not retune here - see
# Step4_Uncertainty_Tagging_Results_Note for how it was chosen.
CONFIDENCE_THRESHOLD = 0.5442

# Point this at the folder saved by trainer.save_pretrained(...) on
# Kaggle (roberta-finetuned-final), copied down locally. Falls back to
# plain pretrained roberta-base with a loud warning, since the locked
# threshold above was calibrated against the fine-tuned model's own
# confidence distribution and is not verified for a different model.
FINE_TUNED_MODEL_PATH = "./roberta-finetuned-final"

FUNCTION_WORDS = {  # kept in sync with build_masked_examples.py by hand
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "with", "as", "by", "that", "this",
}


def load_reconstruction_model():
    from transformers import RobertaTokenizerFast, RobertaForMaskedLM
    if os.path.isdir(FINE_TUNED_MODEL_PATH):
        print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_PATH}")
        tokenizer = RobertaTokenizerFast.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = RobertaForMaskedLM.from_pretrained(FINE_TUNED_MODEL_PATH)
    else:
        print(
            f"WARNING: {FINE_TUNED_MODEL_PATH} not found locally. Falling back to "
            f"pretrained roberta-base. The locked confidence threshold "
            f"({CONFIDENCE_THRESHOLD}) was calibrated on the fine-tuned model's "
            f"confidence distribution and is NOT verified for this fallback - "
            f"'uncertain' flags below should not be trusted until the fine-tuned "
            f"model is copied down from Kaggle and this path is updated."
        )
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.eval()
    return tokenizer, model


def match_ocr_word(box, ocr_words, min_overlap=0.3):
    """Finds the OCR-detected word (on the damaged image) whose box
    overlaps `box` the most, by intersection-over-union. Returns None
    when nothing overlaps enough - expected for words OCR could not
    read at all under heavy damage."""
    bx1, by1 = box["left"], box["top"]
    bx2, by2 = bx1 + box["width"], by1 + box["height"]
    b_area = box["width"] * box["height"]

    best, best_iou = None, 0.0
    for w in ocr_words:
        wx1, wy1 = w["left"], w["top"]
        wx2, wy2 = wx1 + w["width"], wy1 + w["height"]
        ix1, iy1 = max(bx1, wx1), max(by1, wy1)
        ix2, iy2 = min(bx2, wx2), min(by2, wy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = b_area + w["width"] * w["height"] - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = w, iou

    return best if best_iou >= min_overlap else None


def reconstruct_word(tokenizer, model, context_words, target_index):
    """Iterative masked-token decoding for one word position. context_words
    is a list of strings (one per word) with the target position about to
    be blanked out; every other position uses whatever text is currently
    known for it (see simplification #2 in the module docstring).

    Tries 1, 2, and 3 mask tokens (simplification #1) and returns whichever
    reconstruction has the highest geometric-mean confidence:
    exp(mean(log p_i)) over the model's own chosen subword probabilities.
    """
    import torch

    before_text = " ".join(context_words[:target_index])
    after_text = " ".join(context_words[target_index + 1:])

    best_result = None
    for num_mask_tokens in (1, 2, 3):
        predicted_ids, log_probs = [], []
        running_before = before_text
        for step in range(num_mask_tokens):
            remaining = num_mask_tokens - step
            mask_span = " ".join([tokenizer.mask_token] * remaining)
            full_text = f"{running_before} {mask_span} {after_text}".strip()
            encoding = tokenizer(full_text, truncation=True, max_length=64, return_tensors="pt")
            input_ids = encoding["input_ids"][0]
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].tolist()
            if len(mask_positions) != remaining:
                break
            with torch.no_grad():
                outputs = model(**encoding)
            logits = outputs.logits[0]
            probs = torch.softmax(logits[mask_positions[0]], dim=-1)
            predicted_id = torch.argmax(probs).item()
            predicted_ids.append(predicted_id)
            log_probs.append(math.log(probs[predicted_id].item() + 1e-12))
            running_before = tokenizer.decode(
                tokenizer(running_before, add_special_tokens=False)["input_ids"] + [predicted_id]
            )
        if len(predicted_ids) != num_mask_tokens:
            continue
        word = tokenizer.decode(predicted_ids).strip()
        if " " in word:
            continue  # not a single word - e.g. multi-token fills that span a word boundary
        confidence = math.exp(sum(log_probs) / len(log_probs))
        if best_result is None or confidence > best_result[1]:
            best_result = (word, confidence)

    if best_result is None:
        return "[reconstruction failed]", 0.0
    return best_result


def _reconstruction_accuracy(word_records, content_only):
    attempted = [r for r in word_records if r["damage_status"] != "intact"]
    if content_only:
        attempted = [r for r in attempted if r["_ground_truth_text"].strip(",.;:").lower() not in FUNCTION_WORDS]
    if not attempted:
        return None
    correct = sum(
        1 for r in attempted
        if r["reconstructed_text"].strip(",.;:").lower() == r["_ground_truth_text"].strip(",.;:").lower()
    )
    return round(correct / len(attempted), 4)


def build_document_json(document_id, clean_image_path, seed=42):
    clean_img = Image.open(clean_image_path).convert("RGB")
    width, height = clean_img.size
    bg_color = estimate_background_color(clean_img)

    # Ground-truth word positions come from OCR on the CLEAN image, before
    # damage is introduced - see the scope note at the top of this file.
    ground_truth_words = get_word_boxes(clean_image_path)
    print("of:", repr(ground_truth_words[17]["text"]))
    print("reflection:", repr(ground_truth_words[18]["text"]))
    ground_truth_text = " ".join(w["text"] for w in ground_truth_words)

    mask = generate_ribbon_tear(width, height, seed=seed, edge_tear=False)
    damaged_img = apply_tear(clean_img, mask, bg_color)
    damaged_img.save(f"{document_id}_damaged.png")

    classified_words = classify_all_words(ground_truth_words, mask)

    ocr_data = pytesseract.image_to_data(damaged_img, config="--psm 6", output_type=Output.DICT)
    ocr_words = []
    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        conf = float(ocr_data["conf"][i]) if ocr_data["conf"][i] not in ("-1", "") else -1
        if text and conf >= 0:
            ocr_words.append({
                "text": text, "left": ocr_data["left"][i], "top": ocr_data["top"][i],
                "width": ocr_data["width"][i], "height": ocr_data["height"][i], "conf": conf,
            })

    tokenizer, model = load_reconstruction_model()

    # Pass 1: resolve every word's OCR match. Damaged words that OCR
    # could not read at all get ocr_text=None here; that's expected.
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
            "_ground_truth_text": w["text"],  # stripped before writing the final JSON; used only for evaluation
        })

    # Context for reconstruction: whatever text is currently known per
    # position. See simplification #2 - damaged neighbors contribute a
    # placeholder here, not their own (not-yet-computed) reconstruction.
    context_words = [r["final_text"] or "[UNK]" for r in word_records]

    # Pass 2: reconstruct every non-intact word.
    for idx, r in enumerate(word_records):
        if r["damage_status"] == "intact":
            continue
        reconstructed_text, confidence = reconstruct_word(tokenizer, model, context_words, idx)
        r["reconstructed_text"] = reconstructed_text
        r["reconstruction_confidence"] = round(confidence, 4)
        r["uncertain"] = confidence <= CONFIDENCE_THRESHOLD
        r["final_text"] = reconstructed_text
        r["provenance"] = "reconstructed"

    final_text = " ".join(r["final_text"] for r in word_records)

    document = {
        "document_id": document_id,
        "source_image_path": clean_image_path,
        "page_number": 1,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "final_text": final_text,
        "words": [{k: v for k, v in r.items() if not k.startswith("_")} for r in word_records],
        "evaluation": {
            "ground_truth_text": ground_truth_text,
            "cer": round(cer(ground_truth_text, final_text), 4),
            "wer": round(wer(ground_truth_text, final_text), 4),
            "reconstruction_top1_accuracy": _reconstruction_accuracy(word_records, content_only=False),
            "content_word_accuracy": _reconstruction_accuracy(word_records, content_only=True),
        },
    }
    return document


if __name__ == "__main__":
    document = build_document_json("camus_sample_001", "clean.png", seed=42)

    with open("camus_sample_001_output.json", "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    print(json.dumps(document, indent=2))

    # Fail loudly, not silently, if the output doesn't match the contract.
    import jsonschema
    schema = json.load(open("document_schema.json"))
    jsonschema.validate(instance=document, schema=schema)
    print("\nValidated against document_schema.json - OK")
