import random
import math
import json
from PIL import Image, ImageDraw
from collections import Counter

def estimate_background_color(img):
    """Most frequent pixel color = our best guess at the page background,
    since background pixels vastly outnumber text pixels."""
    pixels = list(img.getdata())
    most_common = Counter(pixels).most_common(1)[0][0]
    return most_common


def generate_ribbon_tear(width, height, seed, edge_tear, num_points=12):
    rng = random.Random(seed)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Same border-biased starting logic as before.
    if edge_tear:
        side = rng.choice(["left", "right", "top", "bottom"])
        if side == "left":
            x, y, angle = 0, rng.uniform(0, height), rng.uniform(-30, 30)
        elif side == "right":
            x, y, angle = width, rng.uniform(0, height), rng.uniform(150, 210)
        elif side == "top":
            x, y, angle = rng.uniform(0, width), 0, rng.uniform(60, 120)
        else:
            x, y, angle = rng.uniform(0, width), height, rng.uniform(240, 300)
    else:
        x = rng.uniform(width * 0.25, width * 0.75)
        y = rng.uniform(height * 0.25, height * 0.75)
        angle = rng.uniform(0, 360)

    # Walk the path, recording each point AND a random width at that point.
    centerline = [(x, y)]
    widths = [rng.uniform(8, 18)]
    for _ in range(num_points):
        angle += rng.uniform(-30, 30)
        step = rng.uniform(10, 20)
        x = x + step * math.cos(math.radians(angle))
        y = y + step * math.sin(math.radians(angle))
        centerline.append((x, y))
        widths.append(rng.uniform(6, 20))

    # For each point, offset perpendicular to the path's local direction
    # by that point's width, once to the left, once to the right.
    # Stitching left-side + reversed-right-side into one polygon gives
    # a continuous, varying-width ribbon along the whole path.
    left_side, right_side = [], []
    for i in range(len(centerline)):
        if i == 0:
            dx = centerline[1][0] - centerline[0][0]
            dy = centerline[1][1] - centerline[0][1]
        elif i == len(centerline) - 1:
            dx = centerline[i][0] - centerline[i - 1][0]
            dy = centerline[i][1] - centerline[i - 1][1]
        else:
            dx = centerline[i + 1][0] - centerline[i - 1][0]
            dy = centerline[i + 1][1] - centerline[i - 1][1]
        length = math.hypot(dx, dy) or 1
        nx, ny = -dy / length, dx / length  # perpendicular unit vector
        w = widths[i]
        cx, cy = centerline[i]
        left_side.append((cx + nx * w, cy + ny * w))
        right_side.append((cx - nx * w, cy - ny * w))

    polygon = left_side + right_side[::-1]
    draw.polygon(polygon, fill=255)
    return mask


def apply_tear(img, mask, background_color):
    """Erase pixels under the mask, replacing with the estimated paper color."""
    result = img.copy()
    result_pixels = result.load()
    mask_pixels = mask.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if mask_pixels[x, y] == 255:
                result_pixels[x, y] = background_color
    return result


def mask_bounding_box(mask):
    return mask.getbbox()  # (left, top, right, bottom), or None if empty


if __name__ == "__main__":
    clean_img = Image.open("clean.png").convert("RGB")
    width, height = clean_img.size
    bg_color = estimate_background_color(clean_img)

    preview_configs = [
        ("edge", True, 0),
        ("edge", True, 1),
        ("interior", False, 2),
        ("interior", False, 3),
    ]

    for label, is_edge, seed in preview_configs:
        mask = generate_ribbon_tear(width, height, seed, edge_tear=is_edge)
        torn_img = apply_tear(clean_img, mask, bg_color)

        mask_path = f"mask_{label}_seed{seed}.png"
        torn_path = f"torn_{label}_seed{seed}.png"
        mask.save(mask_path)
        torn_img.save(torn_path)

        bbox = mask_bounding_box(mask)
        area_fraction = sum(1 for p in mask.getdata() if p == 255) / (width * height)

        print(f"[{label} seed={seed}] bbox={bbox}  area_fraction={area_fraction:.3f}")