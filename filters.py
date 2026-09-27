"""
filters.py — OpenCV filter functions for face detection web app
Each filter receives the full frame + detected faces + eyes and returns a modified frame.
"""

import cv2
import numpy as np
import math
import time


def apply_filter(frame, faces, eyes_list, filter_name):
    """Route to the correct filter function."""
    if filter_name == "none":
        return _draw_face_boxes(frame, faces)
    elif filter_name == "dog":
        return _filter_dog(frame, faces)
    elif filter_name == "sunglasses":
        return _filter_sunglasses(frame, faces, eyes_list)
    elif filter_name == "crown":
        return _filter_crown(frame, faces)
    elif filter_name == "rainbow":
        return _filter_rainbow(frame, faces)
    elif filter_name == "neon":
        return _filter_neon(frame, faces)
    elif filter_name == "vintage":
        return _filter_vintage(frame, faces)
    elif filter_name == "cyberpunk":
        return _filter_cyberpunk(frame, faces)
    elif filter_name == "mirror":
        return _filter_mirror(frame, faces)
    elif filter_name == "sketch":
        return _filter_sketch(frame, faces)
    elif filter_name == "pixelate":
        return _filter_pixelate(frame, faces)
    elif filter_name == "glitch":
        return _filter_glitch(frame, faces)
    elif filter_name == "thermal":
        return _filter_thermal(frame)
    elif filter_name == "nightvision":
        return _filter_nightvision(frame, faces)
    elif filter_name == "cartoon":
        return _filter_cartoon(frame, faces)
    elif filter_name == "fisheye":
        return _filter_fisheye(frame, faces)
    elif filter_name == "disco":
        return _filter_disco(frame, faces)
    return frame


# ──────────────────────────────────────────────
# Helper: draw green boxes (no filter selected)
# ──────────────────────────────────────────────
def _draw_face_boxes(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
        # Corner accents
        length = 20
        thickness = 3
        color = (0, 255, 100)
        for px, py, dx, dy in [
            (x, y, 1, 1), (x+w, y, -1, 1),
            (x, y+h, 1, -1), (x+w, y+h, -1, -1)
        ]:
            cv2.line(frame, (px, py), (px + dx*length, py), color, thickness)
            cv2.line(frame, (px, py), (px, py + dy*length), color, thickness)
    return frame


# ──────────────────────────────────────────────
# 1. Dog filter — ears + nose
# ──────────────────────────────────────────────
def _filter_dog(frame, faces):
    overlay = frame.copy()
    for (x, y, w, h) in faces:
        # Brown floppy ears
        ear_w = w // 3
        ear_h = int(h * 0.45)
        ear_color = (40, 80, 139)   # BGR brown
        ear_dark  = (20, 50, 100)

        # Left ear (rounded floppy shape)
        left_cx = x + ear_w // 2
        left_cy = y + ear_h // 2
        cv2.ellipse(overlay, (left_cx, left_cy - 10),
                    (ear_w//2, ear_h//2), 0, 0, 360, ear_color, -1)
        cv2.ellipse(overlay, (left_cx, left_cy - 10),
                    (ear_w//4, ear_h//3), 0, 0, 360, ear_dark, -1)

        # Right ear
        right_cx = x + w - ear_w // 2
        right_cy = y + ear_h // 2
        cv2.ellipse(overlay, (right_cx, right_cy - 10),
                    (ear_w//2, ear_h//2), 0, 0, 360, ear_color, -1)
        cv2.ellipse(overlay, (right_cx, right_cy - 10),
                    (ear_w//4, ear_h//3), 0, 0, 360, ear_dark, -1)

        # Black dog nose
        nose_cx = x + w // 2
        nose_cy = y + int(h * 0.68)
        nose_rx = w // 10
        nose_ry = int(h * 0.055)
        cv2.ellipse(overlay, (nose_cx, nose_cy),
                    (nose_rx, nose_ry), 0, 0, 360, (30, 30, 30), -1)

        # Nostrils
        cv2.circle(overlay, (nose_cx - nose_rx//2, nose_cy), 3, (0, 0, 0), -1)
        cv2.circle(overlay, (nose_cx + nose_rx//2, nose_cy), 3, (0, 0, 0), -1)

        # Tongue
        tongue_cx = nose_cx
        tongue_top = nose_cy + nose_ry + 2
        tongue_w = w // 9
        tongue_h = int(h * 0.12)
        cv2.rectangle(overlay,
                      (tongue_cx - tongue_w, tongue_top),
                      (tongue_cx + tongue_w, tongue_top + tongue_h//2),
                      (80, 60, 200), -1)
        cv2.ellipse(overlay, (tongue_cx, tongue_top + tongue_h//2),
                    (tongue_w, tongue_h//2), 0, 0, 180, (80, 60, 200), -1)

        # Whisker lines
        wh_y = nose_cy
        for wdir in [-1, 1]:
            for i in range(1, 4):
                start_x = nose_cx + wdir * nose_rx
                end_x = nose_cx + wdir * (nose_rx + w // 4)
                angle_off = (i - 2) * int(h * 0.03)
                cv2.line(overlay, (start_x, wh_y + angle_off),
                         (end_x, wh_y + angle_off - wdir * angle_off),
                         (200, 200, 200), 1)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ──────────────────────────────────────────────
# 2. Sunglasses filter
# ──────────────────────────────────────────────
def _filter_sunglasses(frame, faces, eyes_list):
    overlay = frame.copy()
    for i, (x, y, w, h) in enumerate(faces):
        # Eye zone: upper-middle third of face
        eye_y = y + int(h * 0.28)
        eye_h_band = int(h * 0.22)
        glasses_y = eye_y - 4
        glasses_h = eye_h_band + 8

        # Full glasses frame spanning face width
        glasses_x = x - int(w * 0.05)
        glasses_w = int(w * 1.1)

        # Bridge area
        mid_x = glasses_x + glasses_w // 2

        # Left lens
        left_x = glasses_x
        left_w = glasses_w // 2 - 10
        cv2.rectangle(overlay,
                      (left_x, glasses_y),
                      (left_x + left_w, glasses_y + glasses_h),
                      (10, 10, 10), -1)
        cv2.rectangle(overlay,
                      (left_x, glasses_y),
                      (left_x + left_w, glasses_y + glasses_h),
                      (80, 80, 80), 2)

        # Right lens
        right_x = mid_x + 10
        right_w = glasses_w // 2 - 10
        cv2.rectangle(overlay,
                      (right_x, glasses_y),
                      (right_x + right_w, glasses_y + glasses_h),
                      (10, 10, 10), -1)
        cv2.rectangle(overlay,
                      (right_x, glasses_y),
                      (right_x + right_w, glasses_y + glasses_h),
                      (80, 80, 80), 2)

        # Bridge
        cv2.line(overlay,
                 (left_x + left_w, glasses_y + glasses_h//2),
                 (right_x, glasses_y + glasses_h//2),
                 (60, 60, 60), 3)

        # Lens tint (blue/purple gradient simulation)
        tint = overlay[glasses_y:glasses_y+glasses_h, left_x:left_x+left_w]
        tint_color = np.zeros_like(tint)
        tint_color[:, :, 0] = 80   # Blue
        tint_color[:, :, 2] = 40   # Red
        cv2.addWeighted(tint_color, 0.3, tint, 0.7, 0,
                        overlay[glasses_y:glasses_y+glasses_h, left_x:left_x+left_w])

        # Temples (arms)
        cv2.line(overlay,
                 (glasses_x, glasses_y + glasses_h//2),
                 (x - int(w*0.12), glasses_y + glasses_h//2),
                 (40, 40, 40), 3)
        cv2.line(overlay,
                 (glasses_x + glasses_w, glasses_y + glasses_h//2),
                 (x + w + int(w*0.12), glasses_y + glasses_h//2),
                 (40, 40, 40), 3)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    return frame


# ──────────────────────────────────────────────
# 3. Crown filter
# ──────────────────────────────────────────────
def _filter_crown(frame, faces):
    overlay = frame.copy()
    gold  = (0, 200, 255)   # BGR gold
    gold2 = (0, 140, 210)
    gem_colors = [(0, 0, 200), (0, 200, 0), (200, 0, 0)]

    for (x, y, w, h) in faces:
        crown_base_y = y - int(h * 0.08)
        crown_h = int(h * 0.35)
        crown_x = x + int(w * 0.05)
        crown_w = int(w * 0.9)

        # Crown base band
        cv2.rectangle(overlay,
                      (crown_x, crown_base_y - int(crown_h * 0.3)),
                      (crown_x + crown_w, crown_base_y),
                      gold2, -1)

        # Crown spikes (5 points)
        pts_x = [crown_x + crown_w * i // 4 for i in range(5)]
        spike_heights = [crown_h, crown_h*0.6, crown_h, crown_h*0.6, crown_h]

        for i in range(4):
            pts = np.array([
                [pts_x[i], crown_base_y - int(crown_h * 0.3)],
                [pts_x[i], crown_base_y - int(spike_heights[i])],
                [pts_x[i+1], crown_base_y - int(spike_heights[i+1])],
                [pts_x[i+1], crown_base_y - int(crown_h * 0.3)]
            ], np.int32)
            cv2.fillPoly(overlay, [pts], gold)

        # Gems on spikes
        for i, gx in enumerate(pts_x):
            gem_y = crown_base_y - int(spike_heights[i]) + 8
            cv2.circle(overlay, (gx, gem_y), 6,
                       gem_colors[i % len(gem_colors)], -1)
            cv2.circle(overlay, (gx, gem_y), 6, (200, 200, 200), 1)

        # Outline
        cv2.rectangle(overlay,
                      (crown_x, crown_base_y - int(crown_h * 0.3)),
                      (crown_x + crown_w, crown_base_y),
                      gold2, 2)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ──────────────────────────────────────────────
# 4. Rainbow filter — arc over face
# ──────────────────────────────────────────────
def _filter_rainbow(frame, faces):
    overlay = frame.copy()
    t = time.time()
    colors = [
        (0, 0, 255),   # Red
        (0, 128, 255), # Orange
        (0, 255, 255), # Yellow
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (255, 0, 128), # Indigo
        (255, 0, 200), # Violet
    ]
    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        base_r = int(w * 0.65)
        band_w = max(4, w // 22)
        for j, color in enumerate(colors):
            r = base_r + j * band_w
            # Animate with slight pulse
            r += int(math.sin(t * 2 + j) * 4)
            # Draw arc (top half)
            cv2.ellipse(overlay, (cx, cy), (r, r),
                        0, 180, 360, color, band_w)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame


# ──────────────────────────────────────────────
# 5. Neon filter — glowing face outline
# ──────────────────────────────────────────────
def _filter_neon(frame, faces):
    # Create glowing edges on a dark background
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Color shift: cyan
    neon = np.zeros_like(frame)
    neon[:, :, 1] = edges  # Green channel
    neon[:, :, 0] = edges  # Blue channel (cyan = G+B)

    # Blur to create glow
    glow = cv2.GaussianBlur(neon, (15, 15), 0)

    # Darken base frame
    dark = (frame * 0.4).astype(np.uint8)
    result = cv2.add(dark, glow)

    # Highlight face boxes in magenta
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.rectangle(result, (x-2, y-2), (x+w+2, y+h+2), (180, 0, 180), 1)

    return result


# ──────────────────────────────────────────────
# 6. Vintage / Sepia filter
# ──────────────────────────────────────────────
def _filter_vintage(frame, faces):
    # Convert to float for sepia math
    img_float = frame.astype(np.float32) / 255.0
    r = img_float[:, :, 2]
    g = img_float[:, :, 1]
    b = img_float[:, :, 0]

    sepia_r = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 1)
    sepia_g = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 1)
    sepia_b = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 1)

    sepia = np.stack([sepia_b, sepia_g, sepia_r], axis=2)
    sepia = (sepia * 255).astype(np.uint8)

    # Vignette overlay
    rows, cols = frame.shape[:2]
    X = cv2.getGaussianKernel(cols, cols * 0.6)
    Y = cv2.getGaussianKernel(rows, rows * 0.6)
    vignette = (Y @ X.T)
    vignette = vignette / vignette.max()
    for c in range(3):
        sepia[:, :, c] = (sepia[:, :, c] * vignette).astype(np.uint8)

    # Film grain
    grain = np.random.normal(0, 12, sepia.shape).astype(np.int16)
    sepia = np.clip(sepia.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    return sepia


# ──────────────────────────────────────────────
# 7. Cyberpunk / HUD filter
# ──────────────────────────────────────────────
def _filter_cyberpunk(frame, faces):
    result = frame.copy()
    h, w = frame.shape[:2]

    # Green tint shift
    result[:, :, 1] = np.clip(result[:, :, 1].astype(int) + 30, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2].astype(int) - 20, 0, 255)

    # Scanlines
    for row in range(0, h, 4):
        result[row, :] = (result[row, :] * 0.7).astype(np.uint8)

    # HUD border
    cv2.rectangle(result, (10, 10), (w - 10, h - 10), (0, 255, 80), 1)
    cv2.rectangle(result, (12, 12), (w - 12, h - 12), (0, 200, 60), 1)

    # Corner brackets
    bracket_len = 30
    bracket_color = (0, 255, 80)
    for bx, by, dx, dy in [(10, 10, 1, 1), (w-10, 10, -1, 1),
                            (10, h-10, 1, -1), (w-10, h-10, -1, -1)]:
        cv2.line(result, (bx, by), (bx + dx*bracket_len, by), bracket_color, 2)
        cv2.line(result, (bx, by), (bx, by + dy*bracket_len), bracket_color, 2)

    # HUD text
    ts = time.strftime("%H:%M:%S")
    cv2.putText(result, f"SYS:{ts}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)
    cv2.putText(result, "FACE-AI v2.0", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)

    # Face targeting reticle
    for (fx, fy, fw, fh) in faces:
        cx = fx + fw // 2
        cy = fy + fh // 2
        # Targeting crosshair
        cv2.line(result, (cx - 40, cy), (cx - 10, cy), (0, 255, 80), 1)
        cv2.line(result, (cx + 10, cy), (cx + 40, cy), (0, 255, 80), 1)
        cv2.line(result, (cx, cy - 40), (cx, cy - 10), (0, 255, 80), 1)
        cv2.line(result, (cx, cy + 10), (cx, cy + 40), (0, 255, 80), 1)
        cv2.circle(result, (cx, cy), 5, (0, 255, 80), 1)

        # Corner brackets on face
        bl = 15
        for bx, by, ddx, ddy in [
            (fx, fy, 1, 1), (fx+fw, fy, -1, 1),
            (fx, fy+fh, 1, -1), (fx+fw, fy+fh, -1, -1)
        ]:
            cv2.line(result, (bx, by), (bx + ddx*bl, by), (0, 255, 200), 2)
            cv2.line(result, (bx, by), (bx, by + ddy*bl), (0, 255, 200), 2)

        # Stats readout
        cv2.putText(result, "LOCKED", (fx, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 80), 1)

    return result


# ──────────────────────────────────────────────
# 8. Mirror filter — split screen flip
# ──────────────────────────────────────────────
def _filter_mirror(frame, faces):
    h, w = frame.shape[:2]
    left  = frame[:, :w//2]
    right = cv2.flip(left, 1)
    result = np.concatenate([left, right], axis=1)

    # Divider line
    cv2.line(result, (w//2, 0), (w//2, h), (200, 200, 200), 2)
    cv2.putText(result, "MIRROR", (w//2 - 30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return result


# ──────────────────────────────────────────────
# 9. Sketch / Pencil filter
# ──────────────────────────────────────────────
def _filter_sketch(frame, faces):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
    result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # Highlight face zones in blue
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (200, 100, 50), 1)

    return result


# ──────────────────────────────────────────────
# 10. Pixelate face filter
# ──────────────────────────────────────────────
def _filter_pixelate(frame, faces):
    result = frame.copy()
    for (x, y, w, h) in faces:
        roi = result[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        pixel_size = max(8, w // 15)
        small = cv2.resize(roi, (w // pixel_size, h // pixel_size),
                           interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h),
                               interpolation=cv2.INTER_NEAREST)
        result[y:y+h, x:x+w] = pixelated
        cv2.rectangle(result, (x, y), (x+w, y+h), (180, 100, 255), 2)
    return result


# ──────────────────────────────────────────────
# 11. Glitch — RGB channel split + scan bars
# ──────────────────────────────────────────────
def _filter_glitch(frame, faces):
    result = frame.copy()
    t = time.time()
    h, w = frame.shape[:2]

    # Animated channel offset (shifts every ~0.1s)
    offset = int((math.sin(t * 8) * 0.5 + 0.5) * 18) + 4

    # Split RGB channels and shift horizontally
    b, g, r = cv2.split(result)
    shift = offset
    # Shift red channel right
    r_shifted = np.roll(r, shift, axis=1)
    # Shift blue channel left
    b_shifted = np.roll(b, -shift, axis=1)

    result = cv2.merge([b_shifted, g, r_shifted])

    # Random horizontal glitch bars
    rng = np.random.default_rng(int(t * 10) % 1000)
    num_bars = rng.integers(3, 8)
    for _ in range(num_bars):
        bar_y  = rng.integers(0, h - 5)
        bar_h  = rng.integers(2, 8)
        bar_dx = rng.integers(-30, 30)
        region = result[bar_y:bar_y+bar_h, :].copy()
        result[bar_y:bar_y+bar_h, :] = np.roll(region, bar_dx, axis=1)

    # Neon face border
    for (x, y, fw, fh) in faces:
        cv2.rectangle(result, (x, y), (x+fw, y+fh), (0, 255, 255), 1)
        cv2.rectangle(result, (x+2, y+2), (x+fw-2, y+fh-2), (255, 0, 255), 1)

    # "GLITCH" watermark
    cv2.putText(result, "GLITCH", (w - 90, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return result


# ──────────────────────────────────────────────
# 12. Thermal / Heat map
# ──────────────────────────────────────────────
def _filter_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Invert so bright = hot
    gray_inv = cv2.bitwise_not(gray)
    thermal  = cv2.applyColorMap(gray_inv, cv2.COLORMAP_JET)

    # Blend lightly with original for more realism
    result = cv2.addWeighted(thermal, 0.85, frame, 0.15, 0)

    h, w = frame.shape[:2]
    # Temperature scale bar on right edge
    scale_x = w - 20
    for i in range(200):
        y_pos = h // 2 - 100 + i
        if 0 <= y_pos < h:
            heat_val = int(255 * (1 - i / 200))
            color_bar = cv2.applyColorMap(
                np.array([[heat_val]], dtype=np.uint8), cv2.COLORMAP_JET
            )[0][0].tolist()
            cv2.line(result, (scale_x, y_pos), (scale_x + 10, y_pos), color_bar, 1)

    cv2.putText(result, "HOT",  (scale_x - 5, h // 2 - 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(result, "COLD", (scale_x - 8, h // 2 + 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1)
    cv2.putText(result, "THERMAL", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return result


# ──────────────────────────────────────────────
# 13. Night Vision — green monochrome + grain
# ──────────────────────────────────────────────
def _filter_nightvision(frame, faces):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Boost brightness
    bright = cv2.convertScaleAbs(gray, alpha=1.6, beta=30)
    # Apply green tint
    green_frame = np.zeros_like(frame)
    green_frame[:, :, 1] = bright   # Green channel only

    # Subtle grain
    grain = np.random.normal(0, 8, green_frame.shape).astype(np.int16)
    result = np.clip(green_frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Vignette
    h, w = frame.shape[:2]
    X = cv2.getGaussianKernel(w, w * 0.55)
    Y = cv2.getGaussianKernel(h, h * 0.55)
    vig = (Y @ X.T)
    vig = (vig / vig.max() * 255).astype(np.uint8)
    for c in range(3):
        result[:, :, c] = (result[:, :, c] * (vig / 255.0)).astype(np.uint8)

    # Face targeting circles
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw // 2, y + fh // 2
        cv2.circle(result, (cx, cy), fw // 2 + 5, (0, 255, 0), 1)
        cv2.circle(result, (cx, cy), 8, (0, 255, 0), 1)
        cv2.line(result, (cx - 20, cy), (cx + 20, cy), (0, 200, 0), 1)
        cv2.line(result, (cx, cy - 20), (cx, cy + 20), (0, 200, 0), 1)

    cv2.putText(result, "NV-MODE", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    return result


# ──────────────────────────────────────────────
# 14. Cartoon — bilateral smooth + edge overlay
# ──────────────────────────────────────────────
def _filter_cartoon(frame, faces):
    # Multiple bilateral passes for smooth paint effect
    smooth = frame.copy()
    for _ in range(4):
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)

    # Edge mask
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=2
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine: smooth * edges
    cartoon = cv2.bitwise_and(smooth, edges_bgr)

    # Boost saturation
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    cartoon = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return cartoon


# ──────────────────────────────────────────────
# 15. Fisheye / Barrel warp on face
# ──────────────────────────────────────────────
def _filter_fisheye(frame, faces):
    result = frame.copy()
    for (x, y, fw, fh) in faces:
        # Expand ROI slightly
        pad = int(fw * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + fw + pad)
        y2 = min(frame.shape[0], y + fh + pad)
        roi = frame[y1:y2, x1:x2].copy()
        rh, rw = roi.shape[:2]
        if rw < 4 or rh < 4:
            continue

        # Build fisheye map
        cx, cy = rw / 2.0, rh / 2.0
        K = 0.45   # distortion strength
        map_x = np.zeros((rh, rw), np.float32)
        map_y = np.zeros((rh, rw), np.float32)
        for row in range(rh):
            for col in range(rw):
                nx = (col - cx) / cx
                ny = (row - cy) / cy
                r  = math.sqrt(nx*nx + ny*ny)
                if r == 0:
                    map_x[row, col] = col
                    map_y[row, col] = row
                else:
                    theta = math.atan(r * K) / (K if K != 0 else 1)
                    factor = theta / r if r > 1e-6 else 1
                    map_x[row, col] = cx + nx * factor * cx
                    map_y[row, col] = cy + ny * factor * cy

        warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        result[y1:y2, x1:x2] = warped
        cv2.ellipse(result, (x + fw//2, y + fh//2),
                    (fw//2 + pad, fh//2 + pad), 0, 0, 360, (180, 100, 255), 2)
    return result


# ──────────────────────────────────────────────
# 16. Disco / Party — animated color strobe
# ──────────────────────────────────────────────
def _filter_disco(frame, faces):
    result = frame.copy()
    t = time.time()
    h, w = frame.shape[:2]

    # Cycling hue shift on full frame
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.int32)
    hue_shift = int((t * 60) % 180)   # full hue cycle
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 60, 0, 255)  # boost saturation
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Spinning light rays from center top
    cx, cy = w // 2, 0
    num_rays = 8
    for i in range(num_rays):
        angle = math.radians((t * 120 + i * (360 / num_rays)) % 360)
        ex = int(cx + math.cos(angle) * w)
        ey = int(cy + math.sin(angle) * h)
        intensity = int(abs(math.sin(t * 3 + i)) * 60)
        ray_color = (
            int(abs(math.sin(t + i * 1.1)) * 200),
            int(abs(math.cos(t + i * 0.7)) * 200),
            intensity
        )
        overlay = result.copy()
        cv2.line(overlay, (cx, cy), (ex, ey), ray_color, 3)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

    # Flashing face highlight
    flash_alpha = abs(math.sin(t * 4)) * 0.4
    for (fx, fy, fw, fh) in faces:
        flash_overlay = result.copy()
        face_color = (
            int(abs(math.sin(t * 3.1)) * 255),
            int(abs(math.cos(t * 2.7)) * 255),
            int(abs(math.sin(t * 1.9 + 1)) * 255)
        )
        cv2.rectangle(flash_overlay, (fx, fy), (fx+fw, fy+fh), face_color, -1)
        cv2.addWeighted(flash_overlay, flash_alpha, result, 1 - flash_alpha, 0, result)
        cv2.rectangle(result, (fx, fy), (fx+fw, fy+fh), face_color, 3)

    # Disco text
    cv2.putText(result, "DISCO", (w//2 - 35, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return result
