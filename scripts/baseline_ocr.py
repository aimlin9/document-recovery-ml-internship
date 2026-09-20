import random
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from jiwer import cer, wer
import statistics

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CLEAN_TEXT = (
    "To doubt everything or to believe everything are two equally convenient "
    "solutions; both dispense with the necessity of reflection."
)

DEGRADATION_LEVELS = {
    "mild":     {"contrast_scale": 0.75, "contrast_offset": 0.10, "blur": 0.8, "noise": 12},
    "moderate": {"contrast_scale": 0.55, "contrast_offset": 0.25, "blur": 1.5, "noise": 25},
    "severe":   {"contrast_scale": 0.35, "contrast_offset": 0.45, "blur": 2.5, "noise": 45},
}

# How many independently-seeded samples to generate per severity level.
# More samples = a more reliable mean, but slower to run (noise loop is per-pixel).
SAMPLES_PER_LEVEL = 5


def render_clean_image(text, path="clean.png"):
    font = ImageFont.truetype("arial.ttf", 24)
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding = 40
    img = Image.new("RGB", (text_width + padding * 2, text_height + padding * 2), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, fill="black", font=font)
    img.save(path)
    return img


def degrade_image(img, contrast_scale, contrast_offset, blur, noise, seed, path="degraded.png"):
    # Seed the random generator so this exact image is reproducible later,
    # anyone re-running this script with the same seed gets the same noise pattern.
    rng = random.Random(seed)

    faded = Image.eval(img, lambda p: int(p * contrast_scale + 255 * contrast_offset))
    blurred = faded.filter(ImageFilter.GaussianBlur(radius=blur))
    pixels = blurred.load()
    width, height = blurred.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            n = rng.randint(-noise, noise)
            pixels[x, y] = (
                max(0, min(255, r + n)),
                max(0, min(255, g + n)),
                max(0, min(255, b + n)),
            )

    blurred.save(path)
    return blurred


def run_ocr_and_score(image_path, ground_truth_text):
    ocr_text = pytesseract.image_to_string(Image.open(image_path), config="--psm 6")
    return ocr_text, cer(ground_truth_text, ocr_text), wer(ground_truth_text, ocr_text)


if __name__ == "__main__":
    clean_img = render_clean_image(CLEAN_TEXT)

    all_results = {}  # level_name -> list of (seed, cer, wer, ocr_text)

    for level_name, params in DEGRADATION_LEVELS.items():
        level_results = []
        for seed in range(SAMPLES_PER_LEVEL):
            # Filename encodes level and seed, so every generated image is
            # individually saved and traceable back to exactly how it was made.
            path = f"degraded_{level_name}_seed{seed}.png"
            degrade_image(clean_img, path=path, seed=seed, **params)
            ocr_text, c, w = run_ocr_and_score(path, CLEAN_TEXT)
            level_results.append((seed, c, w, ocr_text))
            print(f"[{level_name.upper()} seed={seed}] CER={c:.3f}  WER={w:.3f}")

        all_results[level_name] = level_results
        print()

    print("=" * 60)
    print("SUMMARY (mean / min / max across samples per level)")
    print("=" * 60)
    for level_name, level_results in all_results.items():
        cers = [r[1] for r in level_results]
        wers = [r[2] for r in level_results]
        print(
            f"{level_name:10s}  "
            f"CER mean={statistics.mean(cers):.3f} min={min(cers):.3f} max={max(cers):.3f}  "
            f"WER mean={statistics.mean(wers):.3f} min={min(wers):.3f} max={max(wers):.3f}"
        )

    # Flag any individual run that produced unusually long/hallucinated output,
    # worth keeping as a qualitative example regardless of the overall mean.
    print()
    print("Flagged outputs (OCR text much longer than ground truth, possible hallucination):")
    gt_len = len(CLEAN_TEXT)
    for level_name, level_results in all_results.items():
        for seed, c, w, ocr_text in level_results:
            if len(ocr_text) > gt_len * 2:
                print(f"  {level_name} seed={seed}: OCR output length={len(ocr_text)} vs ground truth={gt_len}")