#!/usr/bin/env python3
"""Logo3D — Upload a 2D logo, tune parameters interactively, get a 3D extruded model."""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import base64
import os
import uuid
from PIL import Image
import io

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
import time
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# In-memory session store: {id: {img_bytes, filename, timestamp}}
# Note: with multiple gunicorn workers, use --preload or -w 1 to share memory
sessions = {}
SESSION_TTL = 1800  # 30 minutes


def cleanup_sessions():
    now = time.time()
    expired = [k for k, v in sessions.items() if now - v['timestamp'] > SESSION_TTL]
    for k in expired:
        del sessions[k]


def adaptive_resample(pts, budget=300):
    pts = np.array(pts, dtype=np.float64)
    n = len(pts)
    if n <= budget:
        return pts.tolist()
    p = np.vstack([pts[-1:], pts, pts[:1]])
    d1 = p[1:-1] - p[:-2]
    d2 = p[2:] - p[1:-1]
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    seg_len = np.sqrt((d1 ** 2).sum(axis=1)) + 1e-8
    curvature = cross / (seg_len ** 2)
    weight = curvature + np.median(curvature) * 0.1
    cum_weight = np.cumsum(weight)
    cum_weight = np.insert(cum_weight, 0, 0)
    cum_weight /= cum_weight[-1]
    targets = np.linspace(0, 1, budget, endpoint=False)
    indices = np.searchsorted(cum_weight[1:], targets).clip(0, n - 1)
    result = []
    for t, idx in zip(targets, indices):
        if idx == 0:
            result.append(pts[0].tolist())
        else:
            t0 = cum_weight[idx]
            t1 = cum_weight[idx + 1]
            frac = (t - t0) / (t1 - t0) if t1 - t0 > 1e-10 else 0
            prev = idx - 1 if idx > 0 else n - 1
            p_interp = pts[prev] * (1 - frac) + pts[idx] * frac
            result.append(p_interp.tolist())
    return result


def signed_area(pts):
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return area / 2.0


def decode_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def upscale_image(img):
    h, w = img.shape[:2]
    scale = max(1, 4000 // max(w, h))
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return img, scale


AI_MODELS = ['u2net', 'isnet-general-use', 'silueta', 'u2netp', 'isnet-anime']


def extract_mask_ai_raw(img_orig, model='u2net'):
    """Use rembg to get the raw soft alpha mask (0-255). Cached per session+model."""
    if not HAS_REMBG:
        raise RuntimeError("rembg not installed")
    from rembg import new_session
    sess = new_session(model)
    pil_img = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    result = rembg_remove(pil_img, session=sess)
    return np.array(result)[:, :, 3]  # soft alpha 0-255


def apply_ai_mask(raw_alpha, alpha_thresh=127, erode_px=None, dilate_px=None):
    """Apply threshold + morphology to a raw soft alpha mask."""
    _, mask = cv2.threshold(raw_alpha, int(alpha_thresh), 255, cv2.THRESH_BINARY)
    if dilate_px and dilate_px > 0:
        dk = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, dk, iterations=1)
    if erode_px and erode_px > 0:
        ek = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        mask = cv2.erode(mask, ek, iterations=1)
    return mask


def extract_mask(img_up, sensitivity=1.0, erode_px=None):
    """Extract foreground mask. sensitivity scales the Otsu threshold, erode_px sets erosion."""
    h, w = img_up.shape[:2]
    margin = max(1, min(h, w) // 10)
    corner_pixels = np.vstack([
        img_up[:margin, :margin].reshape(-1, 3),
        img_up[:margin, -margin:].reshape(-1, 3),
        img_up[-margin:, :margin].reshape(-1, 3),
        img_up[-margin:, -margin:].reshape(-1, 3),
    ]).astype(np.float32)
    bg_color = np.median(corner_pixels, axis=0)

    diff = img_up.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adjusted = max(1, int(otsu_val * sensitivity))
    _, thresh = cv2.threshold(dist_u8, adjusted, 255, cv2.THRESH_BINARY)

    k = max(1, min(h, w) // 200)
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    if erode_px is None:
        erode_k = max(1, min(h, w) // 150)
    else:
        erode_k = erode_px
    if erode_k > 0:
        erode_kernel = np.ones((erode_k * 2 + 1, erode_k * 2 + 1), np.uint8)
        thresh = cv2.erode(thresh, erode_kernel, iterations=1)

    return thresh


def extract_contours(thresh, scale, total_budget=600):
    """Find and resample contours from threshold mask."""
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None
    hier = hierarchy[0]

    perimeters = [cv2.arcLength(c, True) / scale for c in contours]
    total_perim = sum(perimeters)
    if total_perim == 0:
        return [], hier

    simplified = []
    for c, perim in zip(contours, perimeters):
        pts = c.reshape(-1, 2).astype(np.float64)
        if scale > 1:
            pts /= scale
        budget = max(20, int(total_budget * perim / total_perim))
        resampled = adaptive_resample(pts, budget=budget)
        simplified.append(resampled)

    return simplified, hier


def build_shapes(simplified, hier, w, h):
    """Build Three.js-ready shapes with holes from contour hierarchy."""
    min_area = max(50, w * h * 0.0001)
    shapes = []
    for i in range(len(simplified)):
        if hier[i][3] != -1:
            continue
        pts = simplified[i]
        area = abs(signed_area(pts))
        if area < min_area:
            continue

        flipped = [[p[0], h - p[1]] for p in pts]
        if signed_area(flipped) < 0:
            flipped.reverse()

        shape = {'outer': flipped, 'holes': []}
        child_idx = hier[i][2]
        while child_idx != -1:
            hole_pts = simplified[child_idx]
            if abs(signed_area(hole_pts)) >= min_area:
                hole_flipped = [[p[0], h - p[1]] for p in hole_pts]
                if signed_area(hole_flipped) > 0:
                    hole_flipped.reverse()
                shape['holes'].append(hole_flipped)
            child_idx = hier[child_idx][0]
        shapes.append(shape)
    return shapes


def get_side_color(img, thresh, w, h):
    if thresh.shape[0] != h or thresh.shape[1] != w:
        mask = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        mask = thresh > 0
    fg_pixels = img[mask]
    if len(fg_pixels) > 0:
        avg_bgr = fg_pixels.mean(axis=0).astype(int)
        return f'0x{avg_bgr[2]:02x}{avg_bgr[1]:02x}{avg_bgr[0]:02x}'
    return '0x888888'


def img_to_b64png(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode()


def apply_manual_overrides(mask, sess, w, h):
    """Apply manual keep/remove overrides to a mask."""
    if 'manual_mask' not in sess:
        return mask
    mm = sess['manual_mask']
    if mm.shape != (h, w):
        return mask
    result = mask.copy()
    result[mm == 1] = 255  # forced keep
    result[mm == 2] = 0    # forced remove
    return result


# ── Routes ──

@app.route('/')
def index():
    return INDEX_HTML


@app.route('/upload', methods=['POST'])
def upload():
    cleanup_sessions()
    if 'image' not in request.files:
        return jsonify(error='No file'), 400
    f = request.files['image']
    if not f.filename:
        return jsonify(error='No file selected'), 400

    img_bytes = f.read()
    img = decode_image(img_bytes)
    if img is None:
        return jsonify(error='Could not decode image'), 400

    sid = str(uuid.uuid4())[:8]
    h, w = img.shape[:2]
    sessions[sid] = {
        'img_bytes': img_bytes,
        'filename': f.filename,
        'timestamp': time.time(),
    }

    ext = f.filename.rsplit('.', 1)[-1].lower()
    mime = 'png' if ext == 'png' else 'webp' if ext == 'webp' else 'jpeg'
    preview = f'data:image/{mime};base64,{base64.b64encode(img_bytes).decode()}'

    return jsonify(id=sid, width=w, height=h, preview=preview)


@app.route('/api/threshold', methods=['POST'])
def api_threshold():
    d = request.json
    sid = d.get('id')
    if sid not in sessions:
        return jsonify(error='Session expired'), 404

    sess = sessions[sid]
    sess['timestamp'] = time.time()
    img = decode_image(sess['img_bytes'])
    h, w = img.shape[:2]

    method = d.get('method', 'color')
    sensitivity = float(d.get('sensitivity', 1.0))
    erode_px = int(d.get('erode', -1))
    if erode_px < 0:
        erode_px = None
    dilate_px = int(d.get('dilate', 0))
    alpha_thresh = int(d.get('alpha_thresh', 127))
    ai_model = d.get('ai_model', 'u2net')

    if method == 'ai':
        # Cache raw alpha per session+model
        cache_key = f'ai_alpha_{ai_model}'
        if cache_key not in sess:
            sess[cache_key] = extract_mask_ai_raw(img, model=ai_model)
        raw_alpha = sess[cache_key]
        mask_small = apply_ai_mask(raw_alpha, alpha_thresh, erode_px, dilate_px)
    else:
        img_up, scale = upscale_image(img)
        thresh = extract_mask(img_up, sensitivity, erode_px)
        if scale > 1:
            mask_small = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_small = thresh
        thresh_for_contours = thresh

    # Overlay: red tint on removed areas (where mask is 0)
    overlay = img.copy()
    removed = mask_small == 0
    overlay[removed] = (overlay[removed] * 0.3 + np.array([0, 0, 180]) * 0.7).astype(np.uint8)

    fg_pct = (mask_small > 0).sum() / (w * h) * 100

    return jsonify(
        preview=f'data:image/png;base64,{img_to_b64png(overlay)}',
        fg_pct=round(fg_pct, 1),
    )


@app.route('/api/paint', methods=['POST'])
def api_paint():
    """Paint a brush circle to force-keep or force-remove at a point."""
    d = request.json
    sid = d.get('id')
    if sid not in sessions:
        return jsonify(error='Session expired'), 404

    sess = sessions[sid]
    sess['timestamp'] = time.time()
    img = decode_image(sess['img_bytes'])
    h, w = img.shape[:2]

    if 'manual_mask' not in sess:
        sess['manual_mask'] = np.zeros((h, w), dtype=np.uint8)

    # Percentage coords and brush size
    px = int(d.get('x', 50) * w / 100)
    py = int(d.get('y', 50) * h / 100)
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    brush = max(3, int(d.get('brush', 5) * max(w, h) / 100))
    mode = d.get('mode', 'keep')  # 'keep', 'remove', or 'clear'

    if mode == 'clear':
        cv2.circle(sess['manual_mask'], (px, py), brush, 0, -1)
    elif mode == 'remove':
        cv2.circle(sess['manual_mask'], (px, py), brush, 2, -1)
    else:
        cv2.circle(sess['manual_mask'], (px, py), brush, 1, -1)

    # Rebuild mask with overrides
    method = d.get('method', 'color')
    sensitivity = float(d.get('sensitivity', 1.0))
    erode_px = int(d.get('erode', -1))
    if erode_px < 0:
        erode_px = None
    dilate_px = int(d.get('dilate', 0))
    alpha_thresh = int(d.get('alpha_thresh', 127))
    ai_model = d.get('ai_model', 'u2net')

    if method == 'ai':
        cache_key = f'ai_alpha_{ai_model}'
        if cache_key not in sess:
            sess[cache_key] = extract_mask_ai_raw(img, model=ai_model)
        base_mask = apply_ai_mask(sess[cache_key], alpha_thresh, erode_px, dilate_px)
    else:
        img_up, scale = upscale_image(img)
        base_mask = extract_mask(img_up, sensitivity, erode_px)
        if scale > 1:
            base_mask = cv2.resize(base_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    final = apply_manual_overrides(base_mask, sess, w, h)

    overlay = img.copy()
    removed = final == 0
    overlay[removed] = (overlay[removed] * 0.3 + np.array([0, 0, 180]) * 0.7).astype(np.uint8)
    # Show brush dots: green for keep, red for remove
    mm = sess['manual_mask']
    overlay[mm == 1] = (overlay[mm == 1] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
    overlay[mm == 2] = (overlay[mm == 2] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    fg_pct = (final > 0).sum() / (w * h) * 100

    return jsonify(
        preview=f'data:image/png;base64,{img_to_b64png(overlay)}',
        fg_pct=round(fg_pct, 1),
    )


@app.route('/api/clear_paint', methods=['POST'])
def api_clear_paint():
    """Clear all manual paint overrides."""
    d = request.json
    sid = d.get('id')
    if sid not in sessions:
        return jsonify(error='Session expired'), 404
    sess = sessions[sid]
    sess['manual_mask'] = np.zeros_like(sess.get('manual_mask', np.zeros((1, 1), dtype=np.uint8)))
    return jsonify(ok=True)


@app.route('/api/upload_mask', methods=['POST'])
def api_upload_mask():
    """Accept a client-generated mask (from SAM2) as base64 PNG."""
    d = request.json
    sid = d.get('id')
    if sid not in sessions:
        return jsonify(error='Session expired'), 404

    sess = sessions[sid]
    sess['timestamp'] = time.time()
    img = decode_image(sess['img_bytes'])
    h, w = img.shape[:2]

    mask_b64 = d.get('mask', '')
    if not mask_b64:
        return jsonify(error='No mask data'), 400

    # Decode mask from base64 PNG
    mask_bytes = base64.b64decode(mask_b64)
    mask_arr = np.frombuffer(mask_bytes, np.uint8)
    mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return jsonify(error='Could not decode mask'), 400

    # Resize to match original image if needed
    if mask_img.shape != (h, w):
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    sess['sam_mask'] = mask_bin

    fg_pct = (mask_bin > 0).sum() / (w * h) * 100
    return jsonify(ok=True, fg_pct=round(fg_pct, 1))


@app.route('/api/contours', methods=['POST'])
def api_contours():
    d = request.json
    sid = d.get('id')
    if sid not in sessions:
        return jsonify(error='Session expired'), 404

    sess = sessions[sid]
    sess['timestamp'] = time.time()
    img = decode_image(sess['img_bytes'])
    h, w = img.shape[:2]

    method = d.get('method', 'color')
    sensitivity = float(d.get('sensitivity', 1.0))
    erode_px = int(d.get('erode', -1))
    if erode_px < 0:
        erode_px = None
    budget = int(d.get('budget', 600))

    dilate_px = int(d.get('dilate', 0))
    alpha_thresh = int(d.get('alpha_thresh', 127))
    ai_model = d.get('ai_model', 'u2net')

    if method == 'sam':
        if 'sam_mask' not in sess:
            return jsonify(error='No SAM mask uploaded'), 400
        thresh = sess['sam_mask']
        scale = 1
    elif method == 'ai':
        cache_key = f'ai_alpha_{ai_model}'
        if cache_key not in sess:
            sess[cache_key] = extract_mask_ai_raw(img, model=ai_model)
        raw_alpha = sess[cache_key]
        thresh = apply_ai_mask(raw_alpha, alpha_thresh, erode_px, dilate_px)
        thresh = apply_manual_overrides(thresh, sess, w, h)
        scale = 1
    else:
        img_up, scale = upscale_image(img)
        thresh = extract_mask(img_up, sensitivity, erode_px)

    simplified, hier = extract_contours(thresh, scale, budget)

    if not simplified:
        return jsonify(error='No contours found'), 400

    # Draw contours on original image
    overlay = img.copy()
    total_pts = 0
    shape_count = 0
    for i, pts in enumerate(simplified):
        np_pts = (np.array(pts)).astype(np.int32)
        total_pts += len(np_pts)
        color = (0, 255, 100) if hier[i][3] == -1 else (100, 200, 255)
        if hier[i][3] == -1:
            shape_count += 1
        cv2.polylines(overlay, [np_pts], True, color, max(1, max(w, h) // 300))

    return jsonify(
        preview=f'data:image/png;base64,{img_to_b64png(overlay)}',
        shapes=shape_count,
        total_pts=total_pts,
    )


@app.route('/api/generate', methods=['POST'])
def api_generate():
    d = request.json
    sid = d.get('id')
    if sid not in sessions:
        return jsonify(error='Session expired'), 404

    sess = sessions[sid]
    sess['timestamp'] = time.time()
    img = decode_image(sess['img_bytes'])
    h, w = img.shape[:2]

    method = d.get('method', 'color')
    sensitivity = float(d.get('sensitivity', 1.0))
    erode_px = int(d.get('erode', -1))
    if erode_px < 0:
        erode_px = None
    budget = int(d.get('budget', 600))
    depth_pct = float(d.get('depth_pct', 8))

    dilate_px = int(d.get('dilate', 0))
    alpha_thresh = int(d.get('alpha_thresh', 127))
    ai_model = d.get('ai_model', 'u2net')

    if method == 'sam':
        if 'sam_mask' not in sess:
            return jsonify(error='No SAM mask uploaded'), 400
        thresh = sess['sam_mask']
        scale = 1
    elif method == 'ai':
        cache_key = f'ai_alpha_{ai_model}'
        if cache_key not in sess:
            sess[cache_key] = extract_mask_ai_raw(img, model=ai_model)
        raw_alpha = sess[cache_key]
        thresh = apply_ai_mask(raw_alpha, alpha_thresh, erode_px, dilate_px)
        thresh = apply_manual_overrides(thresh, sess, w, h)
        scale = 1
    else:
        img_up, scale = upscale_image(img)
        thresh = extract_mask(img_up, sensitivity, erode_px)

    simplified, hier = extract_contours(thresh, scale, budget)
    if not simplified:
        return jsonify(error='No contours found'), 400

    shapes = build_shapes(simplified, hier, w, h)
    if not shapes:
        return jsonify(error='No shapes extracted'), 400

    img_b64 = base64.b64encode(sess['img_bytes']).decode()
    depth = max(w, h) * (depth_pct / 100.0)
    basename = os.path.splitext(sess['filename'])[0]
    side_color = get_side_color(img, thresh, w, h)

    ext = sess['filename'].rsplit('.', 1)[-1].lower()
    img_mime = 'png' if ext == 'png' else 'webp' if ext == 'webp' else 'jpeg'

    shapes_json = json.dumps(shapes)
    html = VIEWER_TEMPLATE \
        .replace('%%SHAPES%%', shapes_json) \
        .replace('%%IMG_W%%', str(w)) \
        .replace('%%IMG_H%%', str(h)) \
        .replace('%%DEPTH%%', f'{depth:.1f}') \
        .replace('%%IMG_B64%%', img_b64) \
        .replace('%%IMG_EXT%%', img_mime) \
        .replace('%%SIDE_COLOR%%', side_color) \
        .replace('%%BASENAME%%', basename)

    return jsonify(html=html)


# ── Templates ──

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logo3D</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }

header { text-align: center; padding: 24px 16px 0; }
header h1 { font-size: 1.8rem; color: #fff; }
header p { color: #888; font-size: 0.9rem; margin-top: 4px; }

/* Steps indicator */
.steps { display: flex; justify-content: center; gap: 4px; padding: 16px; }
.step-dot { width: 36px; height: 4px; border-radius: 2px; background: #333; transition: background 0.3s; }
.step-dot.active { background: #4a9; }
.step-dot.done { background: #2a6; }

/* Main layout */
.main { display: flex; gap: 20px; padding: 0 20px 20px; height: calc(100vh - 110px); }
.preview-pane { flex: 1; display: flex; align-items: center; justify-content: center;
                background: #151520; border-radius: 12px; overflow: hidden; position: relative; min-width: 0; }
.preview-pane img { max-width: 100%; max-height: 100%; object-fit: contain; user-select: none; -webkit-user-drag: none; }
.preview-pane.paint-mode { cursor: crosshair; }
.preview-pane.paint-mode img { pointer-events: none; }
.sam-overlay { position: absolute; pointer-events: none; }
.preview-pane iframe { width: 100%; height: 100%; border: none; }

.controls { width: 300px; flex-shrink: 0; display: flex; flex-direction: column; gap: 12px; }

/* Step panels */
.panel { display: none; flex-direction: column; gap: 14px;
         background: #1a1a28; border-radius: 10px; padding: 20px; }
.panel.active { display: flex; }
.panel h2 { font-size: 1.1rem; color: #fff; margin-bottom: 4px; }
.panel p.hint { font-size: 0.8rem; color: #777; }

/* Form elements */
label.slider-label { display: flex; justify-content: space-between; font-size: 0.85rem; color: #aaa; }
input[type=range] { width: 100%; accent-color: #4a9; }
.stat { font-size: 0.8rem; color: #888; padding: 4px 0; }

/* Drop zone */
.drop-zone { border: 2px dashed #444; border-radius: 10px; padding: 40px 16px;
             cursor: pointer; text-align: center; transition: all 0.2s; position: relative; }
.drop-zone:hover, .drop-zone.drag { border-color: #4a9; background: rgba(74,153,119,0.06); }
.drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop-zone p { color: #888; font-size: 0.95rem; }

/* Buttons */
.btn { padding: 10px 20px; border: none; border-radius: 6px; font-size: 0.9rem;
       cursor: pointer; transition: background 0.2s; font-family: inherit; }
.btn-primary { background: #2a6; color: #fff; }
.btn-primary:hover { background: #3b7; }
.btn-primary:disabled { background: #444; cursor: not-allowed; }
.btn-secondary { background: #333; color: #ccc; }
.btn-secondary:hover { background: #444; }
.btn-row { display: flex; gap: 8px; }

.spinner { display: none; text-align: center; color: #888; font-size: 0.85rem; padding: 8px; }
.spinner.show { display: block; }

.method-toggle { display: flex; gap: 4px; margin-bottom: 4px; }
.method-btn { flex: 1; padding: 6px 8px; font-size: 0.8rem; background: #333; color: #aaa;
              border: 1px solid #444; transition: all 0.2s; }
.method-btn.active { background: #2a6; color: #fff; border-color: #2a6; }

.select-input { width: 100%; padding: 6px 8px; background: #222; color: #ddd; border: 1px solid #444;
                border-radius: 4px; font-size: 0.85rem; margin-bottom: 8px; }

/* Mobile */
@media (max-width: 768px) {
    .main { flex-direction: column; height: auto; }
    .preview-pane { min-height: 300px; }
    .controls { width: 100%; }
}
</style>
</head>
<body>

<header>
    <h1>Logo3D</h1>
    <p>Upload a logo and extrude it into 3D</p>
</header>

<div class="steps">
    <div class="step-dot active" data-step="0"></div>
    <div class="step-dot" data-step="1"></div>
    <div class="step-dot" data-step="2"></div>
    <div class="step-dot" data-step="3"></div>
</div>

<div class="main">
    <div class="preview-pane" id="previewPane">
        <p style="color:#555">Upload an image to begin</p>
    </div>

    <div class="controls">
        <!-- Step 0: Upload -->
        <div class="panel active" id="step0">
            <h2>1. Upload Image</h2>
            <p class="hint">PNG, JPG, or WebP with a solid background</p>
            <div class="drop-zone" id="dropZone">
                <input type="file" id="fileInput" accept="image/*">
                <p>Drop image here or click to browse</p>
            </div>
            <button class="btn btn-primary" id="btnUploadNext" disabled>Next</button>
        </div>

        <!-- Step 1: Threshold -->
        <div class="panel" id="step1">
            <h2>2. Background Removal</h2>
            <p class="hint">Red = removed area. Adjust until only the logo is white.</p>
            <div class="method-toggle">
                <button class="btn method-btn active" data-method="color" id="methodColor">Color</button>
                <button class="btn method-btn" data-method="ai" id="methodAI">U2Net</button>
                <button class="btn method-btn" data-method="sam" id="methodSAM">SAM 2 &#x2728;</button>
            </div>
            <div id="colorControls">
                <label class="slider-label">Sensitivity <span id="sensVal">1.0</span></label>
                <input type="range" id="sensitivity" min="0.3" max="2.0" step="0.05" value="1.0">
            </div>
            <div id="aiControls" style="display:none">
                <label class="slider-label">Model</label>
                <select id="aiModel" class="select-input">
                    <option value="u2net">U2Net (general)</option>
                    <option value="isnet-general-use">ISNet (general)</option>
                    <option value="silueta">Silueta (fast)</option>
                    <option value="u2netp">U2Net-P (light)</option>
                    <option value="isnet-anime">ISNet (anime)</option>
                </select>
                <label class="slider-label">Alpha Cutoff <span id="alphaVal">127</span></label>
                <input type="range" id="alphaThresh" min="10" max="245" step="5" value="127">
                <label class="slider-label">Dilate <span id="dilateVal">0</span></label>
                <input type="range" id="dilate" min="0" max="20" step="1" value="0">
            </div>
            <label class="slider-label">Erosion <span id="erodeVal">auto</span></label>
            <input type="range" id="erode" min="-1" max="20" step="1" value="-1">
            <div id="samControls" style="display:none">
                <div class="spinner show" id="samLoadSpinner" style="display:none">Loading SAM 2 models (~165MB)...</div>
                <p class="hint" id="samHint">Left-click = keep (green), Right-click = remove (red). Points refine the mask.</p>
                <div class="method-toggle" style="margin-top:6px">
                    <button class="btn method-btn active" id="samUndo" style="background:#555;border-color:#555">Undo Last</button>
                    <button class="btn method-btn" id="samClear" style="background:#555;border-color:#555">Clear All</button>
                </div>
                <div class="stat" id="samStat"></div>
            </div>
            <div id="paintControls" style="margin-top:6px;display:none">
                <label class="slider-label" style="margin-bottom:2px">Paint Mode</label>
                <div class="method-toggle">
                    <button class="btn method-btn paint-btn active" data-mode="keep" id="paintKeep" style="background:#186818;border-color:#186818">+ Keep</button>
                    <button class="btn method-btn paint-btn" data-mode="remove" id="paintRemove">- Remove</button>
                    <button class="btn method-btn paint-btn" data-mode="clear" id="paintClear">Erase</button>
                </div>
                <label class="slider-label">Brush Size <span id="brushVal">5</span></label>
                <input type="range" id="brushSize" min="1" max="15" step="1" value="5">
                <button class="btn btn-secondary" id="btnClearPaint" style="width:100%;font-size:0.75rem;padding:4px">Reset All Paint</button>
            </div>
            <div class="stat" id="fgStat"></div>
            <div class="spinner" id="threshSpinner">Updating...</div>
            <div class="btn-row">
                <button class="btn btn-secondary" id="btnThreshBack">Back</button>
                <button class="btn btn-primary" id="btnThreshNext" style="flex:1">Next</button>
            </div>
        </div>

        <!-- Step 2: Contours -->
        <div class="panel" id="step2">
            <h2>3. Contour Extraction</h2>
            <p class="hint">Green = outer contours, blue = holes</p>
            <label class="slider-label">Point Budget <span id="budgetVal">600</span></label>
            <input type="range" id="budget" min="100" max="2000" step="50" value="600">
            <div class="stat" id="contourStat"></div>
            <div class="spinner" id="contourSpinner">Updating...</div>
            <div class="btn-row">
                <button class="btn btn-secondary" id="btnContourBack">Back</button>
                <button class="btn btn-primary" id="btnContourNext" style="flex:1">Next</button>
            </div>
        </div>

        <!-- Step 3: 3D Viewer -->
        <div class="panel" id="step3">
            <h2>4. 3D Model</h2>
            <label class="slider-label">Depth <span id="depthVal">8%</span></label>
            <input type="range" id="depth" min="1" max="20" step="0.5" value="8">
            <div class="spinner" id="genSpinner">Generating 3D model...</div>
            <button class="btn btn-primary" id="btnRegenerate" style="width:100%">Regenerate</button>
            <button class="btn btn-primary" id="btnDownload" style="width:100%;background:#07a" disabled>Download GLB</button>
            <button class="btn btn-secondary" id="btnStartOver" style="width:100%">Start Over</button>
        </div>
    </div>
</div>

<script>
const $ = s => document.querySelector(s);
const state = { id: null, step: 0, method: 'color' };
let debounceTimer = null;

// ── Step management ──

function setStep(n) {
    state.step = n;
    document.querySelectorAll('.panel').forEach((p, i) =>
        p.classList.toggle('active', i === n));
    document.querySelectorAll('.step-dot').forEach((d, i) => {
        d.classList.toggle('active', i === n);
        d.classList.toggle('done', i < n);
    });
    $('#previewPane').classList.toggle('paint-mode', n === 1);
}

// ── Upload ──

const dz = $('#dropZone'), fi = $('#fileInput');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag');
    fi.files = e.dataTransfer.files; onFileSelected(); });
fi.addEventListener('change', onFileSelected);

function onFileSelected() {
    if (!fi.files[0]) return;
    dz.querySelector('p').textContent = fi.files[0].name;
    $('#btnUploadNext').disabled = false;
}

$('#btnUploadNext').onclick = async () => {
    const fd = new FormData();
    fd.append('image', fi.files[0]);
    $('#btnUploadNext').disabled = true;
    $('#btnUploadNext').textContent = 'Uploading...';

    const r = await fetch('/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) { alert(d.error); $('#btnUploadNext').disabled = false; return; }

    state.id = d.id;
    state._origPreview = d.preview;
    showPreview(d.preview);
    samEmbedding = null; // reset SAM on new upload
    samPoints = [];
    state._samMaskData = null;
    $('#btnUploadNext').textContent = 'Next';
    setStep(1);
    fetchThreshold();
};

function showPreview(src) {
    const pane = $('#previewPane');
    pane.innerHTML = '';
    const img = document.createElement('img');
    img.src = src;
    pane.appendChild(img);
    state._lastPreview = src;
}

function showViewer(html) {
    const pane = $('#previewPane');
    pane.innerHTML = '';
    const iframe = document.createElement('iframe');
    iframe.srcdoc = html;
    pane.appendChild(iframe);
    // Enable GLB download via message from iframe
    $('#btnDownload').disabled = false;
}

// ── Method toggle ──

$('#methodColor').onclick = () => switchMethod('color');
$('#methodAI').onclick = () => switchMethod('ai');
$('#methodSAM').onclick = () => switchMethod('sam');

function switchMethod(m) {
    state.method = m;
    document.querySelectorAll('.method-btn[data-method]').forEach(b =>
        b.classList.toggle('active', b.dataset.method === m));
    $('#colorControls').style.display = m === 'color' ? '' : 'none';
    $('#aiControls').style.display = m === 'ai' ? '' : 'none';
    $('#samControls').style.display = m === 'sam' ? '' : 'none';
    $('#paintControls').style.display = (m === 'color' || m === 'ai') ? '' : 'none';
    if (m === 'sam') {
        // Show original image for SAM point-clicking
        if (state._origPreview) showPreview(state._origPreview);
        initSAM();
    } else {
        // Remove SAM overlay if switching away
        const ov = $('#previewPane .sam-overlay');
        if (ov) ov.remove();
        fetchThreshold();
    }
}

// ── Threshold ──

$('#sensitivity').oninput = () => {
    $('#sensVal').textContent = parseFloat($('#sensitivity').value).toFixed(2);
    debounceFetch(fetchThreshold);
};
$('#erode').oninput = () => {
    const v = parseInt($('#erode').value);
    $('#erodeVal').textContent = v < 0 ? 'auto' : v + 'px';
    debounceFetch(fetchThreshold);
};
$('#alphaThresh').oninput = () => {
    $('#alphaVal').textContent = $('#alphaThresh').value;
    debounceFetch(fetchThreshold);
};
$('#dilate').oninput = () => {
    $('#dilateVal').textContent = $('#dilate').value;
    debounceFetch(fetchThreshold);
};
$('#aiModel').onchange = () => {
    $('#threshSpinner').classList.add('show');
    $('#threshSpinner').textContent = 'Loading model...';
    fetchThreshold();
};

async function fetchThreshold() {
    $('#threshSpinner').classList.add('show');
    const r = await fetch('/api/threshold', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            id: state.id,
            method: state.method,
            sensitivity: parseFloat($('#sensitivity').value),
            erode: parseInt($('#erode').value),
            dilate: parseInt($('#dilate').value || 0),
            alpha_thresh: parseInt($('#alphaThresh').value || 127),
            ai_model: $('#aiModel').value,
        })
    });
    const d = await r.json();
    $('#threshSpinner').classList.remove('show');
    $('#threshSpinner').textContent = 'Updating...';
    if (d.error) return;
    showPreview(d.preview);
    $('#fgStat').textContent = `Foreground: ${d.fg_pct}% of image`;
}

$('#btnThreshBack').onclick = () => setStep(0);
$('#btnThreshNext').onclick = async () => {
    if (state.method === 'sam') {
        if (!state._samMaskData) { alert('Click on the image to create a mask first.'); return; }
        $('#threshSpinner').classList.add('show');
        $('#threshSpinner').textContent = 'Uploading mask...';
        await uploadSAMMask();
        $('#threshSpinner').classList.remove('show');
    }
    setStep(2);
    fetchContours();
};

// ── Contours ──

$('#budget').oninput = () => {
    $('#budgetVal').textContent = $('#budget').value;
    debounceFetch(fetchContours);
};

async function fetchContours() {
    $('#contourSpinner').classList.add('show');
    const r = await fetch('/api/contours', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(getParams({budget: parseInt($('#budget').value)}))
    });
    const d = await r.json();
    $('#contourSpinner').classList.remove('show');
    if (d.error) return;
    showPreview(d.preview);
    $('#contourStat').textContent = `${d.shapes} shape(s), ${d.total_pts} total points`;
}

$('#btnContourBack').onclick = () => { setStep(1); fetchThreshold(); };
$('#btnContourNext').onclick = () => { setStep(3); fetchGenerate(); };

// ── 3D Generation ──

$('#depth').oninput = () => {
    $('#depthVal').textContent = parseFloat($('#depth').value).toFixed(1) + '%';
};

$('#btnRegenerate').onclick = fetchGenerate;

async function fetchGenerate() {
    $('#genSpinner').classList.add('show');
    $('#btnDownload').disabled = true;
    const r = await fetch('/api/generate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(getParams({
            budget: parseInt($('#budget').value),
            depth_pct: parseFloat($('#depth').value),
        }))
    });
    const d = await r.json();
    $('#genSpinner').classList.remove('show');
    if (d.error) { alert(d.error); return; }
    showViewer(d.html);
}

$('#btnDownload').onclick = () => {
    const iframe = $('#previewPane iframe');
    if (iframe && iframe.contentWindow) {
        iframe.contentWindow.postMessage('download-glb', '*');
    }
};

$('#btnStartOver').onclick = () => {
    state.id = null;
    $('#previewPane').innerHTML = '<p style="color:#555">Upload an image to begin</p>';
    fi.value = '';
    dz.querySelector('p').textContent = 'Drop image here or click to browse';
    $('#btnUploadNext').disabled = true;
    $('#btnDownload').disabled = true;
    setStep(0);
};

// ── Utility ──

function getParams(extra = {}) {
    return {
        id: state.id,
        method: state.method,
        sensitivity: parseFloat($('#sensitivity').value),
        erode: parseInt($('#erode').value),
        dilate: parseInt($('#dilate').value || 0),
        alpha_thresh: parseInt($('#alphaThresh').value || 127),
        ai_model: $('#aiModel').value,
        ...extra,
    };
}

function debounceFetch(fn) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fn, 300);
}

// ── Paint keep/remove regions ──

state.paintMode = 'keep';
state.painting = false;

document.querySelectorAll('.paint-btn').forEach(b => {
    b.onclick = () => {
        state.paintMode = b.dataset.mode;
        document.querySelectorAll('.paint-btn').forEach(pb => pb.classList.remove('active'));
        b.classList.add('active');
        // Style active button
        document.querySelectorAll('.paint-btn').forEach(pb => {
            pb.style.background = ''; pb.style.borderColor = '';
        });
        if (b.dataset.mode === 'keep') { b.style.background = '#186818'; b.style.borderColor = '#186818'; }
        else if (b.dataset.mode === 'remove') { b.style.background = '#881818'; b.style.borderColor = '#881818'; }
        else { b.style.background = '#555'; b.style.borderColor = '#555'; }
    };
});

$('#brushSize').oninput = () => { $('#brushVal').textContent = $('#brushSize').value; };

$('#btnClearPaint').onclick = async () => {
    await fetch('/api/clear_paint', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ id: state.id })
    });
    fetchThreshold();
};

function getPaintCoords(e) {
    const img = $('#previewPane img');
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * 100);
    const y = Math.round((e.clientY - rect.top) / rect.height * 100);
    if (x < 0 || x > 100 || y < 0 || y > 100) return null;
    return { x, y };
}

async function paintAt(e) {
    if (state.step !== 1) return;
    const coords = getPaintCoords(e);
    if (!coords) return;

    const r = await fetch('/api/paint', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            ...coords,
            mode: state.paintMode,
            brush: parseInt($('#brushSize').value),
            ...getParams(),
        })
    });
    const d = await r.json();
    if (d.error) return;
    showPreview(d.preview);
    $('#fgStat').textContent = `Foreground: ${d.fg_pct}% of image`;
}

$('#previewPane').addEventListener('mousedown', (e) => {
    if (state.step !== 1) return;
    state.painting = true;
    paintAt(e);
});
$('#previewPane').addEventListener('mousemove', (e) => {
    if (!state.painting) return;
    paintAt(e);
});
document.addEventListener('mouseup', () => { state.painting = false; });

// ── SAM 2 Client-Side Segmentation ──

const SAM_ENCODER_URL = 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.encoder.ort';
const SAM_DECODER_URL = 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.decoder.onnx';

let samEncoder = null, samDecoder = null, samEmbedding = null;
let samPoints = [];
let samImageW = 0, samImageH = 0;

async function initSAM() {
    if (samEncoder && samDecoder) {
        // Already loaded, just re-encode if image changed
        if (!samEmbedding) await encodeSAMImage();
        renderSAMPreview();
        return;
    }

    $('#samLoadSpinner').style.display = '';
    $('#samStat').textContent = '';

    try {
        // Load onnxruntime-web if not loaded
        if (!window.ort) {
            await new Promise((resolve, reject) => {
                const s = document.createElement('script');
                s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
                s.onload = resolve;
                s.onerror = reject;
                document.head.appendChild(s);
            });
        }

        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

        $('#samStat').textContent = 'Loading encoder model (~148MB)...';
        samEncoder = await ort.InferenceSession.create(SAM_ENCODER_URL);

        $('#samStat').textContent = 'Loading decoder model (~16MB)...';
        samDecoder = await ort.InferenceSession.create(SAM_DECODER_URL);

        $('#samStat').textContent = 'Encoding image...';
        await encodeSAMImage();

        $('#samLoadSpinner').style.display = 'none';
        $('#samStat').textContent = 'Ready. Click on the image to segment.';
    } catch (e) {
        console.error('SAM init error:', e);
        $('#samLoadSpinner').style.display = 'none';
        $('#samStat').textContent = 'Error loading SAM: ' + e.message;
    }
}

async function encodeSAMImage() {
    const img = $('#previewPane img');
    if (!img) return;

    // Get original image dimensions
    samImageW = img.naturalWidth;
    samImageH = img.naturalHeight;

    // Draw to 1024x1024 canvas
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 1024, 1024);
    const imageData = ctx.getImageData(0, 0, 1024, 1024).data;

    // Normalize to [-1, 1]
    const input = new Float32Array(3 * 1024 * 1024);
    for (let i = 0; i < 1024 * 1024; i++) {
        input[i] = (imageData[i * 4] / 255.0) * 2 - 1;
        input[i + 1024 * 1024] = (imageData[i * 4 + 1] / 255.0) * 2 - 1;
        input[i + 2 * 1024 * 1024] = (imageData[i * 4 + 2] / 255.0) * 2 - 1;
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, 1024, 1024]);
    const results = await samEncoder.run({ image: tensor });
    samEmbedding = results.image_embed;
    samPoints = [];
}

async function runSAMDecoder() {
    if (!samDecoder || !samEmbedding || samPoints.length === 0) return null;

    const numPoints = samPoints.length;
    const pointCoords = new Float32Array(numPoints * 2);
    const pointLabels = new Float32Array(numPoints);

    // Scale points to 1024x1024 space
    const scaleX = 1024 / samImageW;
    const scaleY = 1024 / samImageH;

    for (let i = 0; i < numPoints; i++) {
        pointCoords[i * 2] = samPoints[i].x * scaleX;
        pointCoords[i * 2 + 1] = samPoints[i].y * scaleY;
        pointLabels[i] = samPoints[i].label;
    }

    const inputs = {
        image_embed: samEmbedding,
        point_coords: new ort.Tensor('float32', pointCoords, [1, numPoints, 2]),
        point_labels: new ort.Tensor('float32', pointLabels, [1, numPoints]),
        mask_input: new ort.Tensor('float32', new Float32Array(1 * 1 * 256 * 256), [1, 1, 256, 256]),
        has_mask_input: new ort.Tensor('float32', new Float32Array([0.0]), [1]),
        high_res_feats_0: new ort.Tensor('float32', new Float32Array(1 * 32 * 256 * 256), [1, 32, 256, 256]),
        high_res_feats_1: new ort.Tensor('float32', new Float32Array(1 * 64 * 128 * 128), [1, 64, 128, 128]),
    };

    const results = await samDecoder.run(inputs);
    return results;
}

function renderSAMPreview() {
    const pane = $('#previewPane');
    // Ensure we have the base image + overlay canvas
    let img = pane.querySelector('img');
    let overlayCanvas = pane.querySelector('.sam-overlay');
    if (!img) {
        // Show original image
        const sess_preview = state._lastPreview;
        if (sess_preview) {
            img = document.createElement('img');
            img.src = sess_preview;
            pane.innerHTML = '';
            pane.appendChild(img);
        }
        return;
    }

    if (!overlayCanvas) {
        overlayCanvas = document.createElement('canvas');
        overlayCanvas.className = 'sam-overlay';
        overlayCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;';
        pane.appendChild(overlayCanvas);
    }

    // Size overlay to match image display size
    const rect = img.getBoundingClientRect();
    const paneRect = pane.getBoundingClientRect();
    overlayCanvas.width = rect.width;
    overlayCanvas.height = rect.height;
    overlayCanvas.style.left = (rect.left - paneRect.left) + 'px';
    overlayCanvas.style.top = (rect.top - paneRect.top) + 'px';
    overlayCanvas.style.width = rect.width + 'px';
    overlayCanvas.style.height = rect.height + 'px';

    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Draw point markers
    for (const pt of samPoints) {
        const cx = pt.x / samImageW * rect.width;
        const cy = pt.y / samImageH * rect.height;
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = pt.label === 1 ? '#00ff00' : '#ff0000';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

async function renderSAMMask(results) {
    if (!results) return;

    const pane = $('#previewPane');
    const img = pane.querySelector('img');
    if (!img) return;

    let overlayCanvas = pane.querySelector('.sam-overlay');
    if (!overlayCanvas) {
        overlayCanvas = document.createElement('canvas');
        overlayCanvas.className = 'sam-overlay';
        overlayCanvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
        pane.appendChild(overlayCanvas);
    }

    const rect = img.getBoundingClientRect();
    const paneRect = pane.getBoundingClientRect();
    overlayCanvas.width = rect.width;
    overlayCanvas.height = rect.height;
    overlayCanvas.style.left = (rect.left - paneRect.left) + 'px';
    overlayCanvas.style.top = (rect.top - paneRect.top) + 'px';
    overlayCanvas.style.width = rect.width + 'px';
    overlayCanvas.style.height = rect.height + 'px';

    const ctx = overlayCanvas.getContext('2d');

    // Get mask data (256x256)
    const maskData = results.masks?.data || results.low_res_masks?.data;
    if (!maskData) return;

    // Render mask to a small canvas then scale
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = 256;
    maskCanvas.height = 256;
    const maskCtx = maskCanvas.getContext('2d');
    const maskImgData = maskCtx.createImageData(256, 256);

    // Use first mask (best)
    for (let i = 0; i < 256 * 256; i++) {
        const v = maskData[i]; // logit value
        if (v > 0) {
            maskImgData.data[i * 4] = 0;
            maskImgData.data[i * 4 + 1] = 180;
            maskImgData.data[i * 4 + 2] = 0;
            maskImgData.data[i * 4 + 3] = 80;
        } else {
            maskImgData.data[i * 4] = 180;
            maskImgData.data[i * 4 + 1] = 0;
            maskImgData.data[i * 4 + 2] = 0;
            maskImgData.data[i * 4 + 3] = 60;
        }
    }
    maskCtx.putImageData(maskImgData, 0, 0);

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    ctx.drawImage(maskCanvas, 0, 0, overlayCanvas.width, overlayCanvas.height);

    // Draw points on top
    for (const pt of samPoints) {
        const cx = pt.x / samImageW * rect.width;
        const cy = pt.y / samImageH * rect.height;
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = pt.label === 1 ? '#00ff00' : '#ff0000';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    // Store mask for upload
    state._samMaskData = maskData;
    const fg = Array.from(maskData).filter(v => v > 0).length;
    $('#samStat').textContent = `${samPoints.length} point(s), ${(fg / (256*256) * 100).toFixed(1)}% foreground`;
}

// Upload SAM mask to server before proceeding to contours
async function uploadSAMMask() {
    if (!state._samMaskData) return false;

    // Render binary mask to canvas at image resolution
    const canvas = document.createElement('canvas');
    canvas.width = samImageW;
    canvas.height = samImageH;
    const ctx = canvas.getContext('2d');

    // First render 256x256 binary mask
    const mask256 = document.createElement('canvas');
    mask256.width = 256;
    mask256.height = 256;
    const m256ctx = mask256.getContext('2d');
    const imgData = m256ctx.createImageData(256, 256);
    for (let i = 0; i < 256 * 256; i++) {
        const v = state._samMaskData[i] > 0 ? 255 : 0;
        imgData.data[i * 4] = v;
        imgData.data[i * 4 + 1] = v;
        imgData.data[i * 4 + 2] = v;
        imgData.data[i * 4 + 3] = 255;
    }
    m256ctx.putImageData(imgData, 0, 0);

    // Scale to full resolution
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(mask256, 0, 0, samImageW, samImageH);

    // Export as PNG base64
    const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
    const buf = await blob.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));

    const r = await fetch('/api/upload_mask', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ id: state.id, mask: b64 })
    });
    const d = await r.json();
    return !d.error;
}

// SAM click handler
$('#previewPane').addEventListener('click', async (e) => {
    if (state.step !== 1 || state.method !== 'sam') return;
    if (!samDecoder || !samEmbedding) return;

    const img = $('#previewPane img');
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * samImageW;
    const y = (e.clientY - rect.top) / rect.height * samImageH;
    if (x < 0 || x > samImageW || y < 0 || y > samImageH) return;

    // Left click = keep (1), will use contextmenu for remove
    samPoints.push({ x, y, label: 1 });
    renderSAMPreview();

    const results = await runSAMDecoder();
    await renderSAMMask(results);
});

$('#previewPane').addEventListener('contextmenu', async (e) => {
    if (state.step !== 1 || state.method !== 'sam') return;
    if (!samDecoder || !samEmbedding) return;
    e.preventDefault();

    const img = $('#previewPane img');
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * samImageW;
    const y = (e.clientY - rect.top) / rect.height * samImageH;
    if (x < 0 || x > samImageW || y < 0 || y > samImageH) return;

    samPoints.push({ x, y, label: 0 });
    renderSAMPreview();

    const results = await runSAMDecoder();
    await renderSAMMask(results);
});

$('#samUndo').onclick = async () => {
    samPoints.pop();
    if (samPoints.length > 0) {
        const results = await runSAMDecoder();
        await renderSAMMask(results);
    } else {
        // Clear overlay
        const ov = $('#previewPane .sam-overlay');
        if (ov) ov.getContext('2d').clearRect(0, 0, ov.width, ov.height);
        state._samMaskData = null;
        $('#samStat').textContent = 'Click on the image to segment.';
    }
};

$('#samClear').onclick = () => {
    samPoints = [];
    const ov = $('#previewPane .sam-overlay');
    if (ov) ov.getContext('2d').clearRect(0, 0, ov.width, ov.height);
    state._samMaskData = null;
    $('#samStat').textContent = 'Click on the image to segment.';
};
</script>
</body>
</html>'''


VIEWER_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>body{margin:0;overflow:hidden;background:#1a1a2e}canvas{display:block}</style>
</head>
<body>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

const SHAPES = %%SHAPES%%;
const IMG_W = %%IMG_W%%;
const IMG_H = %%IMG_H%%;
const DEPTH = %%DEPTH%%;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 1, 100000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const d1 = new THREE.DirectionalLight(0xffffff, 1.0);
d1.position.set(500,800,1500); scene.add(d1);
const d2 = new THREE.DirectionalLight(0xffffff, 0.4);
d2.position.set(-800,-400,-500); scene.add(d2);

const loader = new THREE.TextureLoader();
loader.load('data:image/%%IMG_EXT%%;base64,%%IMG_B64%%', (texture) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    buildMesh(texture);
    animate();
});

function buildMesh(texture) {
    const uvGen = {
        generateTopUV(g,v,a,b,c) {
            return [new THREE.Vector2(v[a*3]/IMG_W,v[a*3+1]/IMG_H),
                    new THREE.Vector2(v[b*3]/IMG_W,v[b*3+1]/IMG_H),
                    new THREE.Vector2(v[c*3]/IMG_W,v[c*3+1]/IMG_H)];
        },
        generateSideWallUV(g,v,a,b,c,d) {
            return [new THREE.Vector2(0,0),new THREE.Vector2(1,0),
                    new THREE.Vector2(1,1),new THREE.Vector2(0,1)];
        }
    };
    const sideMat = new THREE.MeshStandardMaterial({color:%%SIDE_COLOR%%,roughness:0.4,metalness:0.1});
    const capMat = new THREE.MeshStandardMaterial({map:texture});
    const group = new THREE.Group();

    for (const sd of SHAPES) {
        const shape = new THREE.Shape();
        const o = sd.outer;
        shape.moveTo(o[0][0],o[0][1]);
        for (let i=1;i<o.length;i++) shape.lineTo(o[i][0],o[i][1]);
        for (const hp of sd.holes) {
            const hole = new THREE.Path();
            hole.moveTo(hp[0][0],hp[0][1]);
            for (let i=1;i<hp.length;i++) hole.lineTo(hp[i][0],hp[i][1]);
            shape.holes.push(hole);
        }
        const geo = new THREE.ExtrudeGeometry(shape, {
            depth:DEPTH, bevelEnabled:true,
            bevelThickness:DEPTH*0.06, bevelSize:DEPTH*0.06,
            bevelSegments:3, UVGenerator:uvGen
        });
        group.add(new THREE.Mesh(geo,[capMat,sideMat]));
    }

    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    group.position.sub(center);
    scene.add(group);

    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x,size.y,size.z);
    camera.position.set(maxDim*0.2,maxDim*0.1,-maxDim*2.5);
    controls.target.set(0,0,0);
    controls.update();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
}

// Listen for download message from parent
window.addEventListener('message', (e) => {
    if (e.data === 'download-glb') {
        const exporter = new GLTFExporter();
        exporter.parse(scene, (glb) => {
            const blob = new Blob([glb],{type:'model/gltf-binary'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = '%%BASENAME%%.glb';
            a.click(); URL.revokeObjectURL(url);
        }, (err) => console.error(err), {binary:true});
    }
});

addEventListener('resize', () => {
    camera.aspect = innerWidth/innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth,innerHeight);
});
</script>
</body>
</html>'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8015, debug=True)
