from PIL import Image
from word_targeted_masks import get_word_boxes
from torn_regions import estimate_background_color, apply_tear
from transformers import pipeline

CLEAN_IMAGE_PATH = "clean.png"


def classify_word_damage(word_box, mask, partial_threshold=0.05, missing_threshold=0.90):
    """
    Checks how much of a word's rectangle overlaps actual masked pixels
    (not just bounding-box overlap), and classifies it as intact,
    partial, or missing based on the fraction covered.
    """
    x1, y1 = word_box["left"], word_box["top"]
    x2, y2 = x1 + word_box["width"], y1 + word_box["height"]

    mask_pixels = mask.load()
    total = 0
    masked = 0
    for x in range(x1, x2):
        for y in range(y1, y2):
            total += 1
            if mask_pixels[x, y] == 255:
                masked += 1

    fraction = masked / total if total > 0 else 0.0

    if fraction < partial_threshold:
        status = "intact"
    elif fraction < missing_threshold:
        status = "partial"
    else:
        status = "missing"

    return status, fraction


def classify_all_words(words, mask):
    results = []
    for w in words:
        status, fraction = classify_word_damage(w, mask)
        results.append({**w, "damage_status": status, "damage_fraction": fraction})
    return results

def build_masked_sentence_from_ocr(ocr_words, target_index):
    """
    Same idea as build_masked_sentence, but built from OCR output on a
    DAMAGED image rather than clean ground-truth words. This means the
    surrounding context itself may contain OCR errors, a harder, more
    realistic test than masking within perfect context.
    """
    tokens = []
    for i, w in enumerate(ocr_words):
        if i == target_index:
            tokens.append("<mask>")
        else:
            tokens.append(w)
    return " ".join(tokens)


if __name__ == "__main__":
    from word_targeted_masks import build_targeted_mask

    clean_img = Image.open(CLEAN_IMAGE_PATH).convert("RGB")
    width, height = clean_img.size
    bg_color = estimate_background_color(clean_img)
    ground_truth_words = get_word_boxes(CLEAN_IMAGE_PATH)

    fill_mask = pipeline("fill-mask", model="roberta-base", top_k=5)

    results = []
    for i, target_word in enumerate(ground_truth_words):
        # Tear out this specific word from the image.
        mask = build_targeted_mask(width, height, [target_word], seed=i)
        torn_img = apply_tear(clean_img, mask, bg_color)

        # Get OCR's read of the DAMAGED image, this is the real-world
        # context you'd actually have, not the clean ground truth.
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        damaged_text = pytesseract.image_to_string(torn_img, config="--psm 6").strip()
        ocr_words = damaged_text.split()

        # The torn word usually disappears from OCR's word list entirely
        # (no box detected), so we insert the mask token at the same
        # position rather than trying to match text.
        if len(ocr_words) < len(ground_truth_words):
            ocr_words.insert(i, "<mask>")
            masked_sentence = " ".join(ocr_words)
        else:
            masked_sentence = build_masked_sentence_from_ocr(ocr_words, i)
        if i == 5:
          print(f"\n[DEBUG index=5, word='{target_word['text']}'] masked_sentence: {masked_sentence}\n")

        

        predictions = fill_mask(masked_sentence)
        correct = target_word["text"].strip(",.;:").lower()
        top_prediction = predictions[0]["token_str"].strip().lower()
        top5_words = [p["token_str"].strip().lower() for p in predictions]

        results.append({
            "word": target_word["text"],
            "ocr_context": masked_sentence,
            "top_prediction": predictions[0]["token_str"].strip(),
            "top_confidence": predictions[0]["score"],
            "exact_match_top1": top_prediction == correct,
            "in_top5": correct in top5_words,
        })

    print(f"{'WORD':<15} {'PREDICTED':<15} {'CONF':<8} {'TOP1':<8} {'TOP5'}")
    for r in results:
        print(f"{r['word']:<15} {r['top_prediction']:<15} {r['top_confidence']:.3f}    {str(r['exact_match_top1']):<8} {r['in_top5']}")

    top1_acc = sum(r["exact_match_top1"] for r in results) / len(results)
    top5_acc = sum(r["in_top5"] for r in results) / len(results)
    print(f"\nTop-1 exact match rate (OCR context): {top1_acc:.2%}")
    print(f"Top-5 exact match rate (OCR context): {top5_acc:.2%}")