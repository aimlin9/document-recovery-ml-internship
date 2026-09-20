import random
import math
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
from jiwer import cer, wer

from torn_regions import apply_tear, estimate_background_color

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CLEAN_IMAGE_PATH = "clean.png"


def get_word_boxes(image_path, min_conf=30):
    """Runs OCR just to get word positions, not for scoring. Filters out
    low-confidence junk detections Tesseract sometimes produces on whitespace."""
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i] not in ("-1", "") else -1
        if text and conf >= min_conf:
            words.append({
                "text": text, "left": data["left"][i], "top": data["top"][i],
                "width": data["width"][i], "height": data["height"][i], "conf": conf,
            })
    return words


def build_targeted_mask(width, height, target_boxes, seed, padding_px=8, num_vertices=10, jitter=0.15):
    rng = random.Random(seed)
    left = min(b["left"] for b in target_boxes) - padding_px
    top = min(b["top"] for b in target_boxes) - padding_px
    right = max(b["left"] + b["width"] for b in target_boxes) + padding_px
    bottom = max(b["top"] + b["height"] for b in target_boxes) + padding_px

    cx, cy = (left + right) / 2, (top + bottom) / 2
    rx, ry = (right - left) / 2, (bottom - top) / 2

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    points = []
    for i in range(num_vertices):
        angle = 2 * math.pi * i / num_vertices
        jf = rng.uniform(1 - jitter, 1 + jitter)
        points.append((cx + rx * jf * math.cos(angle), cy + ry * jf * math.sin(angle)))
    draw.polygon(points, fill=255)
    return mask


def select_word_boundary_crossing(words, index):
    """Targets only the right half of one word plus the left half of the
    next, simulating a tear that clips across a boundary mid-word rather
    than cleanly covering whole words."""
    w1, w2 = words[index], words[index + 1]
    half1_left = w1["left"] + w1["width"] // 2
    half2_right = w2["left"] + w2["width"] // 2
    return [{
        "text": w1["text"] + "/" + w2["text"],
        "left": half1_left, "top": min(w1["top"], w2["top"]),
        "width": half2_right - half1_left,
        "height": max(w1["top"] + w1["height"], w2["top"] + w2["height"]) - min(w1["top"], w2["top"]),
    }]


def generate_edge_tear_over_text(width, height, seed, text_top, text_bottom, num_points=10):
    rng = random.Random(seed)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    band_top = text_top - 10
    band_bottom = text_bottom + 10

    side = rng.choice(["left", "right"])
    y = rng.uniform(text_top, text_bottom)
    x, angle = (0, rng.uniform(-15, 15)) if side == "left" else (width, rng.uniform(165, 195))

    centerline, widths = [(x, y)], [rng.uniform(10, 18)]
    for _ in range(num_points):
        angle += rng.uniform(-15, 15)  # tighter jitter, less wild drift
        step = rng.uniform(10, 18)
        x += step * math.cos(math.radians(angle))
        y += step * math.sin(math.radians(angle))
        y = max(band_top, min(band_bottom, y))  # keep the walk inside the text band
        centerline.append((x, y))
        widths.append(rng.uniform(8, 18))

    left_side, right_side = [], []
    for i in range(len(centerline)):
        if i == 0:
            dx, dy = centerline[1][0] - centerline[0][0], centerline[1][1] - centerline[0][1]
        elif i == len(centerline) - 1:
            dx, dy = centerline[i][0] - centerline[i-1][0], centerline[i][1] - centerline[i-1][1]
        else:
            dx, dy = centerline[i+1][0] - centerline[i-1][0], centerline[i+1][1] - centerline[i-1][1]
        length = math.hypot(dx, dy) or 1
        nx, ny = -dy / length, dx / length
        cx_, cy_ = centerline[i]
        w = widths[i]
        left_side.append((cx_ + nx * w, cy_ + ny * w))
        right_side.append((cx_ - nx * w, cy_ - ny * w))

    draw.polygon(left_side + right_side[::-1], fill=255)
    return mask


def check_preservation(ocr_text, all_words, target_word_texts):
    """Rough but useful check: of the words that should NOT have been
    touched, how many still show up correctly in the OCR output?"""
    ocr_words_lower = set(w.strip(",.;:").lower() for w in ocr_text.split())
    expected_intact = [w["text"] for w in all_words if w["text"] not in target_word_texts]
    preserved = [w for w in expected_intact if w.strip(",.;:").lower() in ocr_words_lower]
    return len(preserved), len(expected_intact)


if __name__ == "__main__":
    clean_img = Image.open(CLEAN_IMAGE_PATH).convert("RGB")
    width, height = clean_img.size
    bg_color = estimate_background_color(clean_img)
    words = get_word_boxes(CLEAN_IMAGE_PATH)

    print("Detected words:")
    for i, w in enumerate(words):
        print(f"  [{i}] '{w['text']}'")
    print()

    text_top = min(w["top"] for w in words)
    text_bottom = max(w["top"] + w["height"] for w in words)
    print(f"text_top={text_top}, text_bottom={text_bottom}")
    print([(w["text"], w["top"]) for w in words])
    cases = {
        "single_word": [words[2]],
        "two_words": words[5:7],
        "word_boundary": select_word_boundary_crossing(words, 8),
        "larger_span": words[10:14],
    }

    for label, targets in cases.items():
        mask = build_targeted_mask(width, height, targets, seed=hash(label) % 1000)
        torn = apply_tear(clean_img, mask, bg_color)
        torn.save(f"targeted_{label}.png")

        ocr_text = pytesseract.image_to_string(torn, config="--psm 6")
        target_texts = [t["text"] for t in targets]
        preserved, total = check_preservation(ocr_text, words, target_texts)

        print(f"[{label}] targeted: {target_texts}")
        print(f"  OCR output: {ocr_text.strip()}")
        print(f"  Preserved {preserved}/{total} untouched words correctly\n")

    # Edge tear, constrained to actually intersect the text row
    edge_mask = generate_edge_tear_over_text(width, height, seed=42, text_top=text_top, text_bottom=text_bottom)
    edge_torn = apply_tear(clean_img, edge_mask, bg_color)
    edge_torn.save("targeted_edge_tear.png")
    ocr_text = pytesseract.image_to_string(edge_torn, config="--psm 6")
    print(f"[edge_tear] OCR output: {ocr_text.strip()}")

    print(f"Edge mask bounding box: {edge_mask.getbbox()}")
    print(f"Edge mask non-zero pixel count: {sum(1 for p in edge_mask.getdata() if p == 255)}")