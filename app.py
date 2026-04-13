#!/usr/bin/env python3
"""Logo3D — Upload a 2D logo, get an interactive 3D extruded model."""

from flask import Flask, request, render_template_string, jsonify
import cv2
import numpy as np
import json
import base64
import os
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload


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


def process_image(img_bytes, filename):
    """Process uploaded image bytes and return viewer HTML."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image"

    h, w = img.shape[:2]
    orig_img, orig_w, orig_h = img, w, h

    # Upscale for smoother contours
    scale = max(1, 4000 // max(w, h))
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    # Background subtraction
    margin = max(1, min(h, w) // 10)
    corner_pixels = np.vstack([
        img[:margin, :margin].reshape(-1, 3),
        img[:margin, -margin:].reshape(-1, 3),
        img[-margin:, :margin].reshape(-1, 3),
        img[-margin:, -margin:].reshape(-1, 3),
    ]).astype(np.float32)
    bg_color = np.median(corner_pixels, axis=0)

    diff = img.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(1, min(h, w) // 200)
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    erode_k = max(1, min(h, w) // 150)
    erode_kernel = np.ones((erode_k * 2 + 1, erode_k * 2 + 1), np.uint8)
    thresh = cv2.erode(thresh, erode_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No contours found in image"
    hier = hierarchy[0]

    # Adaptive resample
    total_budget = 600
    perimeters = [cv2.arcLength(c, True) / scale for c in contours]
    total_perim = sum(perimeters)
    simplified = []
    for c, perim in zip(contours, perimeters):
        pts = c.reshape(-1, 2).astype(np.float64)
        if scale > 1:
            pts /= scale
        budget = max(20, int(total_budget * perim / total_perim))
        resampled = adaptive_resample(pts, budget=budget)
        simplified.append(resampled)

    w, h, img = orig_w, orig_h, orig_img

    # Build shapes
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

    if not shapes:
        return None, "No shapes extracted from image"

    img_b64 = base64.b64encode(img_bytes).decode()
    depth = max(w, h) * 0.08
    basename = os.path.splitext(filename)[0]

    # Side color
    if thresh.shape[0] != h or thresh.shape[1] != w:
        mask = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        mask = thresh > 0
    fg_pixels = img[mask]
    if len(fg_pixels) > 0:
        avg_bgr = fg_pixels.mean(axis=0).astype(int)
        side_color = f'0x{avg_bgr[2]:02x}{avg_bgr[1]:02x}{avg_bgr[0]:02x}'
    else:
        side_color = '0x888888'

    ext = filename.rsplit('.', 1)[-1].lower()
    img_mime = 'png' if ext == 'png' else 'webp' if ext == 'webp' else 'jpeg'

    return generate_viewer(shapes, img_b64, w, h, depth, side_color, basename, img_mime), None


def generate_viewer(shapes, img_b64, img_w, img_h, depth, side_color, basename, img_mime):
    shapes_json = json.dumps(shapes)
    return VIEWER_TEMPLATE.replace('%%SHAPES%%', shapes_json) \
        .replace('%%IMG_W%%', str(img_w)) \
        .replace('%%IMG_H%%', str(img_h)) \
        .replace('%%DEPTH%%', f'{depth:.1f}') \
        .replace('%%IMG_B64%%', img_b64) \
        .replace('%%IMG_EXT%%', img_mime) \
        .replace('%%SIDE_COLOR%%', side_color) \
        .replace('%%BASENAME%%', basename)


UPLOAD_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logo3D</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f1a; color: #e0e0e0; min-height: 100vh;
       display: flex; align-items: center; justify-content: center; }
.container { max-width: 520px; width: 100%; padding: 40px 24px; text-align: center; }
h1 { font-size: 2.2rem; margin-bottom: 8px; color: #fff; }
.sub { color: #888; margin-bottom: 32px; font-size: 0.95rem; }
.drop-zone { border: 2px dashed #444; border-radius: 12px; padding: 48px 24px;
             cursor: pointer; transition: all 0.2s; position: relative; }
.drop-zone:hover, .drop-zone.drag { border-color: #4a9; background: rgba(74,153,119,0.08); }
.drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop-zone p { font-size: 1.1rem; color: #aaa; }
.drop-zone .icon { font-size: 2.5rem; margin-bottom: 12px; display: block; }
.preview { margin-top: 20px; }
.preview img { max-width: 200px; max-height: 200px; border-radius: 8px; border: 1px solid #333; }
.btn { display: inline-block; margin-top: 20px; padding: 12px 32px;
       background: #2a6; color: #fff; border: none; border-radius: 6px;
       font-size: 1rem; cursor: pointer; transition: background 0.2s; }
.btn:hover { background: #3b7; }
.btn:disabled { background: #555; cursor: not-allowed; }
.status { margin-top: 16px; color: #aaa; font-size: 0.9rem; }
.error { color: #e55; }
</style>
</head>
<body>
<div class="container">
    <h1>Logo3D</h1>
    <p class="sub">Upload a logo image to extrude into a 3D model</p>
    <form id="form" action="/process" method="post" enctype="multipart/form-data">
        <div class="drop-zone" id="dropZone">
            <input type="file" name="image" id="fileInput" accept="image/*" required>
            <span class="icon">&#128196;</span>
            <p>Drop an image here or click to browse</p>
        </div>
        <div class="preview" id="preview" style="display:none">
            <img id="previewImg">
        </div>
        <button class="btn" type="submit" id="submitBtn" disabled>Generate 3D Model</button>
    </form>
    <p class="status" id="status"></p>
</div>
<script>
const dz = document.getElementById('dropZone');
const fi = document.getElementById('fileInput');
const pv = document.getElementById('preview');
const pi = document.getElementById('previewImg');
const btn = document.getElementById('submitBtn');
const st = document.getElementById('status');

dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag');
    fi.files = e.dataTransfer.files; showPreview(); });
fi.addEventListener('change', showPreview);

function showPreview() {
    if (!fi.files[0]) return;
    const r = new FileReader();
    r.onload = e => { pi.src = e.target.result; pv.style.display = ''; btn.disabled = false;
        dz.querySelector('p').textContent = fi.files[0].name; };
    r.readAsDataURL(fi.files[0]);
}

document.getElementById('form').addEventListener('submit', () => {
    btn.disabled = true;
    st.textContent = 'Processing... this may take a few seconds';
    st.className = 'status';
});
</script>
</body>
</html>'''


VIEWER_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>%%BASENAME%% - Logo3D</title>
<style>
body { margin:0; overflow:hidden; background:#1a1a2e; }
canvas { display:block; }
#ui { position:absolute; top:10px; left:10px; right:10px; display:flex; justify-content:space-between; pointer-events:none; }
#ui > * { pointer-events:auto; }
#info { color:#ccc; font:13px/1.4 monospace; background:rgba(0,0,0,.5);
        padding:8px 12px; border-radius:4px; }
.btn { padding:8px 16px; border:none; border-radius:4px; cursor:pointer; font:13px monospace; }
#dlBtn { background:#2a6; color:#fff; }
#backBtn { background:#555; color:#fff; text-decoration:none; }
</style>
</head>
<body>
<div id="ui">
    <div>
        <a href="/" class="btn" id="backBtn">&larr; New</a>
        <span id="info" style="margin-left:8px;">Drag to rotate &middot; Scroll to zoom</span>
    </div>
    <button id="dlBtn" class="btn">Download GLB</button>
</div>

<script type="importmap">
{
    "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/"
    }
}
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
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 1, 100000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const d1 = new THREE.DirectionalLight(0xffffff, 1.0);
d1.position.set(500, 800, 1500); scene.add(d1);
const d2 = new THREE.DirectionalLight(0xffffff, 0.4);
d2.position.set(-800, -400, -500); scene.add(d2);

const loader = new THREE.TextureLoader();
loader.load('data:image/%%IMG_EXT%%;base64,%%IMG_B64%%', (texture) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    buildMesh(texture);
    animate();
});

function buildMesh(texture) {
    const uvGen = {
        generateTopUV(geometry, vertices, iA, iB, iC) {
            return [
                new THREE.Vector2(vertices[iA*3] / IMG_W, vertices[iA*3+1] / IMG_H),
                new THREE.Vector2(vertices[iB*3] / IMG_W, vertices[iB*3+1] / IMG_H),
                new THREE.Vector2(vertices[iC*3] / IMG_W, vertices[iC*3+1] / IMG_H)
            ];
        },
        generateSideWallUV(geometry, vertices, iA, iB, iC, iD) {
            return [
                new THREE.Vector2(0, 0), new THREE.Vector2(1, 0),
                new THREE.Vector2(1, 1), new THREE.Vector2(0, 1)
            ];
        }
    };

    const sideMat = new THREE.MeshStandardMaterial({
        color: %%SIDE_COLOR%%, roughness: 0.4, metalness: 0.1
    });
    const capMat = new THREE.MeshStandardMaterial({ map: texture });
    const group = new THREE.Group();

    for (const sd of SHAPES) {
        const shape = new THREE.Shape();
        const o = sd.outer;
        shape.moveTo(o[0][0], o[0][1]);
        for (let i = 1; i < o.length; i++) shape.lineTo(o[i][0], o[i][1]);
        for (const hp of sd.holes) {
            const hole = new THREE.Path();
            hole.moveTo(hp[0][0], hp[0][1]);
            for (let i = 1; i < hp.length; i++) hole.lineTo(hp[i][0], hp[i][1]);
            shape.holes.push(hole);
        }
        const geo = new THREE.ExtrudeGeometry(shape, {
            depth: DEPTH, bevelEnabled: true,
            bevelThickness: DEPTH * 0.06, bevelSize: DEPTH * 0.06,
            bevelSegments: 3, UVGenerator: uvGen
        });
        group.add(new THREE.Mesh(geo, [capMat, sideMat]));
    }

    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    group.position.sub(center);
    scene.add(group);

    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(maxDim * 0.2, maxDim * 0.1, -maxDim * 2.5);
    controls.target.set(0, 0, 0);
    controls.update();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

document.getElementById('dlBtn').onclick = () => {
    const exporter = new GLTFExporter();
    exporter.parse(scene, (glb) => {
        const blob = new Blob([glb], {type: 'model/gltf-binary'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = '%%BASENAME%%.glb';
        a.click(); URL.revokeObjectURL(url);
    }, (err) => console.error(err), {binary: true});
};

addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>'''


@app.route('/')
def index():
    return UPLOAD_PAGE


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return 'No file uploaded', 400
    f = request.files['image']
    if not f.filename:
        return 'No file selected', 400

    img_bytes = f.read()
    html, error = process_image(img_bytes, f.filename)
    if error:
        return f'<h2>Error: {error}</h2><p><a href="/">Try again</a></p>', 400
    return html


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8015, debug=True)
