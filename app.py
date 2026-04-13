#!/usr/bin/env python3
"""Logo3D — Upload a 2D logo, tune parameters interactively, get a 3D extruded model."""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import base64
import os
import uuid
import time
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# In-memory session store: {id: {img_bytes, filename, timestamp}}
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

    img_up, scale = upscale_image(img)
    sensitivity = float(d.get('sensitivity', 1.0))
    erode_px = int(d.get('erode', -1))
    if erode_px < 0:
        erode_px = None

    thresh = extract_mask(img_up, sensitivity, erode_px)

    # Downscale mask for preview
    if scale > 1:
        mask_small = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_small = thresh

    # Overlay: red tint on removed areas (where mask is 0)
    overlay = img.copy()
    removed = mask_small == 0
    overlay[removed] = (overlay[removed] * 0.3 + np.array([0, 0, 180]) * 0.7).astype(np.uint8)

    fg_pct = (mask_small > 0).sum() / (w * h) * 100

    return jsonify(
        preview=f'data:image/png;base64,{img_to_b64png(overlay)}',
        fg_pct=round(fg_pct, 1),
    )


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

    img_up, scale = upscale_image(img)
    sensitivity = float(d.get('sensitivity', 1.0))
    erode_px = int(d.get('erode', -1))
    if erode_px < 0:
        erode_px = None
    budget = int(d.get('budget', 600))

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

    img_up, scale = upscale_image(img)
    sensitivity = float(d.get('sensitivity', 1.0))
    erode_px = int(d.get('erode', -1))
    if erode_px < 0:
        erode_px = None
    budget = int(d.get('budget', 600))
    depth_pct = float(d.get('depth_pct', 8))

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
.preview-pane img { max-width: 100%; max-height: 100%; object-fit: contain; }
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
            <label class="slider-label">Sensitivity <span id="sensVal">1.0</span></label>
            <input type="range" id="sensitivity" min="0.3" max="2.0" step="0.05" value="1.0">
            <label class="slider-label">Erosion <span id="erodeVal">auto</span></label>
            <input type="range" id="erode" min="-1" max="20" step="1" value="-1">
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
const state = { id: null, step: 0 };
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
    showPreview(d.preview);
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

async function fetchThreshold() {
    $('#threshSpinner').classList.add('show');
    const r = await fetch('/api/threshold', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            id: state.id,
            sensitivity: parseFloat($('#sensitivity').value),
            erode: parseInt($('#erode').value),
        })
    });
    const d = await r.json();
    $('#threshSpinner').classList.remove('show');
    if (d.error) return;
    showPreview(d.preview);
    $('#fgStat').textContent = `Foreground: ${d.fg_pct}% of image`;
}

$('#btnThreshBack').onclick = () => setStep(0);
$('#btnThreshNext').onclick = () => { setStep(2); fetchContours(); };

// ── Contours ──

$('#budget').oninput = () => {
    $('#budgetVal').textContent = $('#budget').value;
    debounceFetch(fetchContours);
};

async function fetchContours() {
    $('#contourSpinner').classList.add('show');
    const r = await fetch('/api/contours', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            id: state.id,
            sensitivity: parseFloat($('#sensitivity').value),
            erode: parseInt($('#erode').value),
            budget: parseInt($('#budget').value),
        })
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
        body: JSON.stringify({
            id: state.id,
            sensitivity: parseFloat($('#sensitivity').value),
            erode: parseInt($('#erode').value),
            budget: parseInt($('#budget').value),
            depth_pct: parseFloat($('#depth').value),
        })
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

function debounceFetch(fn) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fn, 300);
}
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
