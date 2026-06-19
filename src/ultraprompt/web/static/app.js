const $ = (id) => document.getElementById(id);
const body = document.body;
const annotateWorkarea = $('annotateWorkarea');
const canvasShell = $('canvasShell');
const canvasWrap = $('canvasWrap');
const imageCanvas = $('imageCanvas');
const labelCanvas = $('labelCanvas');
const vectorCanvas = $('vectorCanvas');
const imageCtx = imageCanvas.getContext('2d', { willReadFrequently: false });
const labelCtx = labelCanvas.getContext('2d', { willReadFrequently: true });
const vectorCtx = vectorCanvas.getContext('2d');
const inferWorkarea = $('inferWorkarea');
const inferCanvasShell = $('inferCanvasShell');
const inferCanvasWrap = $('inferCanvasWrap');
const inferImageCanvas = $('inferImageCanvas');
const inferOverlayCanvas = $('inferOverlayCanvas');
const inferImageCtx = inferImageCanvas.getContext('2d', { willReadFrequently: false });
const inferOverlayCtx = inferOverlayCanvas.getContext('2d');

const colors = [
  [230, 57, 70], [42, 157, 143], [69, 123, 157], [244, 162, 97],
  [131, 56, 236], [255, 202, 58], [6, 214, 160], [239, 71, 111],
  [58, 134, 255], [138, 201, 38], [255, 127, 17], [91, 192, 190]
];
const MAX_UNDO = 30;
const SEGMENT_MODELS = ['yolo26n-seg', 'yolo26s-seg', 'yolo26m-seg', 'yolo26l-seg', 'yolo26x-seg', 'yolov8n-seg', 'yolov8m-seg', 'yolov8x-seg'];
const SEMANTIC_MODELS = ['yolo26n-sem', 'yolo26s-sem', 'yolo26m-sem', 'yolo26l-sem', 'yolo26x-sem'];

let state = { count: 0, images: [], classes: ['object'], filters: [] };
let currentIndex = 0;
let currentImageInfo = null;
let mode = 'points';
let width = 0;
let height = 0;
let labelMap = new Uint16Array(0);
let points = [];
let boxes = [];
let drawingBox = null;
let brushing = false;
let erasing = false;
let redrawPending = false;
let zoomLevel = 1;
let autoFitZoom = true;
let yoloLogOffset = 0;
let yoloPoll = null;
let lastYoloResults = null;
let picker = { target: null, purpose: 'image_dir', path: '' };
const imageSessions = new Map();
let inferState = { count: 0, images: [], task: 'segment' };
let inferCatalog = [];
let inferHasResults = false;
let inferModelReady = false;
let inferIndex = 0;
let inferWidth = 0;
let inferHeight = 0;
let inferZoomLevel = 1;
let inferAutoFitZoom = true;

function setStatus(text) { $('status').textContent = text; }
function pixelScale() { return Number($('pixelScale').value || 1); }
function filterName() { return $('filterName').value || 'none'; }
function classId() { return Math.max(0, $('classSelect').selectedIndex); }
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
function escapeAttr(text) { return escapeHtml(text); }

function applyTheme(theme) {
  body.dataset.theme = theme;
  localStorage.setItem('ultraprompt-theme', theme);
  $('themeToggle').textContent = theme === 'dark' ? 'Light' : 'Dark';
}

function pickerTitle(purpose) {
  if (purpose === 'image_dir') return 'Select Image Folder';
  if (purpose === 'out_dir') return 'Select Output Folder';
  if (purpose === 'classes') return 'Select classes.txt';
  if (purpose === 'weights') return 'Select Weights';
  if (purpose === 'yaml') return 'Select data.yaml';
  if (purpose === 'csv') return 'Select results.csv';
  return 'Browse VM Files';
}

function classColor(cls) {
  if (cls < colors.length) return colors[cls];
  const hue = (cls * 137) % 360;
  const c = 0.75;
  const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
  const m = 0.15;
  let r = 0, g = 0, b = 0;
  if (hue < 60) [r, g, b] = [c, x, 0];
  else if (hue < 120) [r, g, b] = [x, c, 0];
  else if (hue < 180) [r, g, b] = [0, c, x];
  else if (hue < 240) [r, g, b] = [0, x, c];
  else if (hue < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let msg = await res.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

function currentImageKey(index = currentIndex) {
  const imageRoot = $('imageDir').value || state.image_dir || '';
  const outputRoot = $('outDir').value || state.out_dir || '';
  const imageName = state.images && state.images[index] ? state.images[index] : String(index);
  return `${imageRoot}|${outputRoot}|${imageName}|${pixelScale()}|${filterName()}`;
}

function clonePoints(src = points) {
  return src.map((p) => ({ x: p.x, y: p.y, label: p.label }));
}

function cloneBoxes(src = boxes) {
  return src.map((b) => ({ x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, class_id: b.class_id }));
}

function ensureSession(key, w = width, h = height) {
  let session = imageSessions.get(key);
  if (!session || session.width !== w || session.height !== h) {
    session = {
      width: w,
      height: h,
      points: [],
      boxes: [],
      labelMap: new Uint16Array(Math.max(0, w * h)),
      undo: [],
    };
    imageSessions.set(key, session);
  }
  return session;
}

function saveCurrentSession() {
  if (!state.count || !width || !height) return;
  const session = ensureSession(currentImageKey(), width, height);
  session.points = clonePoints();
  session.boxes = cloneBoxes();
  session.labelMap = labelMap.slice();
}

function loadCurrentSession() {
  const session = ensureSession(currentImageKey(), width, height);
  points = clonePoints(session.points);
  boxes = cloneBoxes(session.boxes);
  labelMap = session.labelMap.slice();
  drawingBox = null;
  drawVectors();
  drawLabels();
  updateUndoState();
}

function pushUndo(includeLabels) {
  if (!width || !height) return;
  const session = ensureSession(currentImageKey(), width, height);
  session.undo.push({
    points: clonePoints(),
    boxes: cloneBoxes(),
    labelMap: includeLabels ? labelMap.slice() : null,
  });
  if (session.undo.length > MAX_UNDO) session.undo.shift();
  updateUndoState();
}

function updateUndoState() {
  const disabled = !width || !height || ensureSession(currentImageKey(), width, height).undo.length === 0;
  $('undoAction').disabled = disabled;
}

function hasAnyLabels() {
  for (let i = 0; i < labelMap.length; i++) {
    if (labelMap[i] !== 0) return true;
  }
  return false;
}

function undoLast() {
  if (!width || !height) return;
  const session = ensureSession(currentImageKey(), width, height);
  const snapshot = session.undo.pop();
  if (!snapshot) {
    setStatus('Nothing to undo');
    return;
  }
  points = clonePoints(snapshot.points);
  boxes = cloneBoxes(snapshot.boxes);
  if (snapshot.labelMap) labelMap = snapshot.labelMap.slice();
  drawingBox = null;
  drawVectors();
  drawLabels();
  saveCurrentSession();
  updateUndoState();
  setStatus('Undid last action');
}

async function refreshState() {
  state = await api('/api/state');
  $('imageSelect').innerHTML = state.images.map((name, i) => `<option value="${i}">${escapeHtml(name)}</option>`).join('');
  $('classSelect').innerHTML = state.classes.map((name) => `<option>${escapeHtml(name)}</option>`).join('');
  $('classesText').value = state.classes.join('\n');
  $('yoloClasses').value = state.classes.join(', ');
  if (state.count) {
    currentIndex = Math.min(currentIndex, state.count - 1);
    $('imageSelect').value = String(currentIndex);
  }
}

function applyZoom() {
  if (!width || !height) return;
  canvasWrap.style.transform = `scale(${zoomLevel})`;
  canvasShell.style.width = `${Math.round(width * zoomLevel)}px`;
  canvasShell.style.height = `${Math.round(height * zoomLevel)}px`;
  $('zoomReadout').textContent = `${Math.round(zoomLevel * 100)}%`;
}

function fitZoom() {
  if (!width || !height) return;
  const rect = annotateWorkarea.getBoundingClientRect();
  const scale = clamp(Math.min((rect.width - 28) / width, (rect.height - 28) / height), 0.25, 16);
  zoomLevel = scale;
  autoFitZoom = true;
  applyZoom();
}

function setZoom(zoom, manual = true) {
  zoomLevel = clamp(zoom, 0.25, 16);
  if (manual) autoFitZoom = false;
  applyZoom();
}

function stepZoom(direction) {
  const factor = direction > 0 ? 1.25 : 0.8;
  setZoom(zoomLevel * factor, true);
}

function clearPointsOnly() {
  if (!points.length) return;
  pushUndo(false);
  points = [];
  drawingBox = null;
  drawVectors();
  saveCurrentSession();
  setStatus('Points cleared');
}

function clearBoxesOnly() {
  if (!boxes.length && !drawingBox) return;
  pushUndo(false);
  boxes = [];
  drawingBox = null;
  drawVectors();
  saveCurrentSession();
  setStatus('Boxes cleared');
}

function clearPromptsOnly() {
  if (!points.length && !boxes.length && !drawingBox) return;
  pushUndo(false);
  points = [];
  boxes = [];
  drawingBox = null;
  drawVectors();
  saveCurrentSession();
  setStatus('Prompts cleared');
}

function clearLabelsOnly() {
  if (!hasAnyLabels()) return;
  pushUndo(true);
  labelMap.fill(0);
  drawLabels();
  saveCurrentSession();
  setStatus('Accepted masks cleared');
}

function labelMapHasData(map) {
  for (let i = 0; i < map.length; i++) {
    if (map[i] !== 0) return true;
  }
  return false;
}

async function decodeLabelMapPng(dataUrl) {
  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = dataUrl;
  });
  const tmp = document.createElement('canvas');
  tmp.width = img.naturalWidth;
  tmp.height = img.naturalHeight;
  const ctx = tmp.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, tmp.width, tmp.height).data;
  const out = new Uint16Array(tmp.width * tmp.height);
  for (let i = 0; i < out.length; i++) out[i] = data[i * 4];
  return out;
}

async function fetchSavedLabelState(index) {
  const qs = new URLSearchParams({ pixel_scale: pixelScale(), filter_name: filterName(), t: Date.now() });
  const result = await api(`/api/image/${index}/saved-label?${qs}`);
  const saved = result.exists ? await decodeLabelMapPng(result.label_png) : new Uint16Array(width * height);
  return { labelMap: saved, source: result.source || null };
}

function clearAllState() {
  if (!points.length && !boxes.length && !drawingBox && !hasAnyLabels()) return;
  pushUndo(true);
  points = [];
  boxes = [];
  drawingBox = null;
  labelMap.fill(0);
  drawVectors();
  drawLabels();
  saveCurrentSession();
  setStatus('Cleared prompts and labels');
}

async function loadImage(index = currentIndex) {
  if (!state.count) return;
  if (width && height) saveCurrentSession();
  currentIndex = Math.max(0, Math.min(index, state.count - 1));
  $('imageSelect').value = String(currentIndex);
  const qs = new URLSearchParams({ pixel_scale: pixelScale(), filter_name: filterName(), t: Date.now() });
  currentImageInfo = await api(`/api/image/${currentIndex}/info?${qs}`);
  const img = new Image();
  img.onload = async () => {
    width = img.naturalWidth;
    height = img.naturalHeight;
    for (const canvas of [imageCanvas, labelCanvas, vectorCanvas]) {
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    canvasWrap.style.width = `${width}px`;
    canvasWrap.style.height = `${height}px`;
    imageCtx.clearRect(0, 0, width, height);
    imageCtx.drawImage(img, 0, 0);
    canvasShell.style.aspectRatio = `${width} / ${height}`;

    const key = currentImageKey();
    let session = imageSessions.get(key);
    let restoredSource = null;
    if (!session || session.width !== width || session.height !== height) {
      const saved = await fetchSavedLabelState(currentIndex);
      restoredSource = saved.source;
      session = {
        width,
        height,
        points: [],
        boxes: [],
        labelMap: saved.labelMap,
        undo: [],
      };
      imageSessions.set(key, session);
    }
    loadCurrentSession();
    if (autoFitZoom || zoomLevel === 1) fitZoom();
    else applyZoom();
    const restoredText = restoredSource && labelMapHasData(labelMap) ? ` | restored ${restoredSource}` : '';
    setStatus(`${currentImageInfo.name} | view ${width}x${height} | source ${currentImageInfo.source_size[0]}x${currentImageInfo.source_size[1]}${restoredText}`);
  };
  img.src = `/api/image/${currentIndex}/png?${qs}`;
}

function canvasPos(evt) {
  const rect = vectorCanvas.getBoundingClientRect();
  return {
    x: clamp((evt.clientX - rect.left) * width / rect.width, 0, width - 1),
    y: clamp((evt.clientY - rect.top) * height / rect.height, 0, height - 1),
  };
}

function scheduleDrawLabels() {
  if (redrawPending) return;
  redrawPending = true;
  requestAnimationFrame(() => {
    redrawPending = false;
    drawLabels();
  });
}

function drawLabels() {
  if (!width || !height) return;
  const img = labelCtx.createImageData(width, height);
  const data = img.data;
  for (let i = 0; i < labelMap.length; i++) {
    const value = labelMap[i];
    if (!value) continue;
    const [r, g, b] = classColor(value - 1);
    const j = i * 4;
    data[j] = r;
    data[j + 1] = g;
    data[j + 2] = b;
    data[j + 3] = 115;
  }
  labelCtx.putImageData(img, 0, 0);
}

function drawBox(box, transient) {
  const [r, g, b] = classColor((box.class_id != null ? box.class_id : classId()));
  vectorCtx.strokeStyle = `rgb(${r},${g},${b})`;
  vectorCtx.setLineDash(transient ? [5, 4] : []);
  vectorCtx.lineWidth = 2;
  vectorCtx.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
  vectorCtx.setLineDash([]);
}

function drawVectors() {
  vectorCtx.clearRect(0, 0, width, height);
  vectorCtx.lineWidth = 2;
  for (const point of points) {
    vectorCtx.beginPath();
    vectorCtx.arc(point.x, point.y, 5, 0, Math.PI * 2);
    vectorCtx.fillStyle = point.label ? '#16a34a' : '#dc2626';
    vectorCtx.fill();
    vectorCtx.strokeStyle = '#ffffff';
    vectorCtx.stroke();
  }
  for (const box of boxes) drawBox(box, false);
  if (drawingBox) drawBox(drawingBox, true);
}

function paintBrush(x, y, erase) {
  if (!width || !height) return;
  const radius = Number($('brushSize').value || 12);
  const value = classId() + 1;
  const cx = Math.round(x);
  const cy = Math.round(y);
  const radius2 = radius * radius;
  const x0 = Math.max(0, cx - radius);
  const x1 = Math.min(width - 1, cx + radius);
  const y0 = Math.max(0, cy - radius);
  const y1 = Math.min(height - 1, cy + radius);
  for (let yy = y0; yy <= y1; yy++) {
    const dy = yy - cy;
    const row = yy * width;
    for (let xx = x0; xx <= x1; xx++) {
      const dx = xx - cx;
      if (dx * dx + dy * dy <= radius2) labelMap[row + xx] = erase ? 0 : value;
    }
  }
  scheduleDrawLabels();
}

function clearClass(cls) {
  const value = cls + 1;
  for (let i = 0; i < labelMap.length; i++) {
    if (labelMap[i] === value) labelMap[i] = 0;
  }
}

async function applyMask(mask) {
  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = `data:image/png;base64,${mask.png}`;
  });
  const tmp = document.createElement('canvas');
  tmp.width = width;
  tmp.height = height;
  const ctx = tmp.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;
  const value = Number(mask.class_id) + 1;
  for (let i = 0; i < labelMap.length; i++) {
    if (data[i * 4] > 127 && labelMap[i] === 0) labelMap[i] = value;
  }
}

async function runSam() {
  if (!state.count) return;
  setStatus('Running SAM...');
  const payload = {
    index: currentIndex,
    pixel_scale: pixelScale(),
    filter_name: filterName(),
    mode,
    class_id: classId(),
    concept_text: $('conceptText').value,
    points,
    boxes,
  };
  const result = await api('/api/segment', { method: 'POST', body: JSON.stringify(payload) });
  if (!result.masks.length) {
    setStatus(mode === 'points' ? 'No mask returned; existing labels kept.' : 'No masks returned');
    return;
  }
  pushUndo(true);
  if (mode === 'points') clearClass(classId());
  for (const mask of result.masks) await applyMask(mask);
  drawLabels();
  saveCurrentSession();
  setStatus(`Accepted ${result.masks.length} mask(s)`);
}

function labelPngDataUrl() {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < labelMap.length; i++) {
    const value = Math.min(255, labelMap[i]);
    const j = i * 4;
    img.data[j] = value;
    img.data[j + 1] = value;
    img.data[j + 2] = value;
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}

async function saveYolo() {
  if (!state.count || !width || !height) return;
  setStatus('Saving labels...');
  saveCurrentSession();
  const result = await api('/api/save-yolo', {
    method: 'POST',
    body: JSON.stringify({
      index: currentIndex,
      pixel_scale: pixelScale(),
      filter_name: filterName(),
      label_png: labelPngDataUrl(),
    }),
  });
  setStatus(`Saved ${result.polygons} polygon(s) + semantic mask: ${result.mask_path}`);
}

async function openPicker(target, purpose, startPath = '') {
  picker = { target, purpose, path: startPath || '' };
  $('fileDialogTitle').textContent = pickerTitle(purpose);
  $('fileDialog').classList.remove('hidden');
  await loadPickerPath(picker.path);
}

function renderBreadcrumbs(path, roots) {
  const rootPaths = roots.map((root) => root.path).sort((a, b) => b.length - a.length);
  const base = rootPaths.find((root) => path.startsWith(root)) || path;
  const remainder = path.slice(base.length).split('/').filter(Boolean);
  const parts = [{ name: base, path: base }];
  let current = base;
  for (const part of remainder) {
    current = current.endsWith('/') ? `${current}${part}` : `${current}/${part}`;
    parts.push({ name: part, path: current });
  }
  $('fileBreadcrumbs').innerHTML = parts.map((part) => `<button type="button" data-path="${escapeAttr(part.path)}">${escapeHtml(part.name)}</button>`).join('');
}

async function loadPickerPath(path) {
  const qs = new URLSearchParams({ purpose: picker.purpose });
  if (path) qs.set('path', path);
  const data = await api(`/api/browse?${qs}`);
  picker.path = data.path;
  $('fileDialogPath').textContent = data.path;
  $('fileParent').disabled = !data.parent;
  $('fileParent').dataset.path = data.parent || '';
  $('selectCurrent').disabled = !data.current_selectable;
  $('selectCurrent').style.display = ['classes', 'weights', 'yaml', 'csv'].includes(picker.purpose) ? 'none' : '';
  renderBreadcrumbs(data.path, data.roots);
  $('fileRoots').innerHTML = data.roots.map((root) => `<button type="button" class="${data.path.startsWith(root.path) ? 'active' : ''}" data-root="${escapeAttr(root.path)}">${escapeHtml(root.name)}</button>`).join('');
  $('fileEntries').innerHTML = data.entries.map((entry) => {
    const kind = entry.is_dir ? 'Folder' : 'File';
    const action = entry.is_dir ? 'Open' : (entry.selectable ? 'Select' : 'View');
    return `<button type="button" class="file-entry ${entry.selectable ? 'selectable' : ''}" data-path="${escapeAttr(entry.path)}" data-dir="${entry.is_dir ? '1' : '0'}" data-selectable="${entry.selectable ? '1' : '0'}"><span class="name">${escapeHtml(entry.name)}</span><span class="type">${kind}</span><span class="tag">${action}</span></button>`;
  }).join('');
}

function closePicker() {
  $('fileDialog').classList.add('hidden');
}

function clearDraftSessions(message) {
  imageSessions.clear();
  points = [];
  boxes = [];
  drawingBox = null;
  if (width && height) {
    labelMap = new Uint16Array(width * height);
    drawVectors();
    drawLabels();
  } else {
    labelMap = new Uint16Array(0);
  }
  updateUndoState();
  setStatus(message);
}

async function createFolderInPicker() {
  const name = window.prompt('New folder name');
  if (name == null) return;
  const trimmed = name.trim();
  if (!trimmed) return;
  const result = await api('/api/mkdir', {
    method: 'POST',
    body: JSON.stringify({ parent_dir: picker.path, name: trimmed }),
  });
  await loadPickerPath(result.path);
  setStatus(`Created folder ${result.name}`);
}

async function choosePickerPath(path) {
  if (picker.target === 'classesPath' || picker.target === 'yoloClassesPath') {
    await api('/api/classes', { method: 'POST', body: JSON.stringify({ classes_path: path }) });
    if ($('yoloClassesPath')) $('yoloClassesPath').value = path;
    await refreshState();
    setStatus(`Loaded classes from ${path}`);
    setYoloStatus(`Loaded classes from ${path}`);
  } else {
    $(picker.target).value = path;
    if (picker.target === 'imageDir') {
      clearDraftSessions('Image folder changed. Click Open to load that folder.');
    } else if (picker.target === 'outDir') {
      clearDraftSessions('Output folder changed. Click Open to reload saved labels from that folder.');
    } else if (picker.target === 'inferImagesDir') {
      scanInferImages().catch((e) => setInferStatus(e.message));
      setInferStatus(`Selected ${path}`);
    } else if (picker.target === 'inferWeightsPath') {
      initializeInferModel().catch((e) => setInferStatus(`Model init failed: ${e.message}`));
      setInferStatus(`Selected ${path}`);
    } else {
      setStatus(`Selected ${path}`);
    }
  }
  closePicker();
}

function showPage(name) {
  const annotate = name === 'annotate';
  const yolo = name === 'yolo';
  const infer = name === 'infer';
  $('annotatePage').classList.toggle('hidden', !annotate);
  $('yoloPage').classList.toggle('hidden', !yolo);
  $('inferPage').classList.toggle('hidden', !infer);
  $('tabAnnotate').classList.toggle('active', annotate);
  $('tabYolo').classList.toggle('active', yolo);
  $('tabInfer').classList.toggle('active', infer);
}

async function refreshYoloDevices() {
  const data = await api('/api/yolo/devices');
  $('yoloDevice').innerHTML = data.devices.map((dev) => `<option value="${escapeAttr(dev.split(' ')[0])}">${escapeHtml(dev)}</option>`).join('');
  setYoloStatus(`Devices refreshed | ${data.platform}`);
}

function setYoloStatus(text) { $('yoloStatus').textContent = text; }
function yoloNumber(id) { return Number($(id).value); }
function yoloChecked(id) { return $(id).checked; }
function yoloTask() { return $('yoloTask').value || 'segment'; }

function setDisabled(ids, disabled) {
  ids.forEach((id) => {
    const el = $(id);
    if (el) el.disabled = disabled;
  });
}

function syncYoloTaskUi() {
  const task = yoloTask();
  const models = task === 'semantic' ? SEMANTIC_MODELS : SEGMENT_MODELS;
  const current = $('yoloModel').value;
  const labelDir = $('yoloLabelsDir');
  const outDir = $('outDir').value || '';
  $('yoloModel').innerHTML = models.map((model) => `<option value="${escapeAttr(model)}">${escapeHtml(model)}</option>`).join('');
  $('yoloModel').value = models.includes(current) ? current : models[0];
  $('yoloAnnoLabel').textContent = task === 'semantic' ? 'Dataset Masks' : 'Dataset Labels';
  labelDir.placeholder = task === 'semantic' ? 'masks folder (.png)' : 'labels folder (.txt)';
  if (!labelDir.value || /\/(labels|masks)$/.test(labelDir.value)) {
    if (outDir) labelDir.value = `${outDir}/${task === 'semantic' ? 'masks' : 'labels'}`;
  }
  $('buildYoloDataset').textContent = task === 'semantic' ? 'Build Semantic Dataset + data.yaml' : 'Build Segment Dataset + data.yaml';
  setDisabled(['yoloCls', 'yoloConf', 'yoloDropout', 'yoloMaskRatio', 'yoloBox'], task === 'semantic');
  syncYoloOptimizeUi();
}

function syncYoloOptimizeUi() {
  const auto = yoloChecked('yoloAutoOptimize');
  setDisabled(['yoloOptimizer', 'yoloLr0'], auto);
}

function syncYoloSplitUi() {
  const useTest = yoloChecked('yoloUseTestSplit');
  $('yoloTrainPct').max = useTest ? '90' : '100';
  $('yoloValPct').min = useTest ? '5' : '0';
  const train = yoloNumber('yoloTrainPct');
  const val = yoloNumber('yoloValPct');
  if (!useTest && train + val !== 100) {
    $('yoloValPct').value = String(Math.max(0, 100 - train));
  }
}

function inferTask() { return $('inferTask').value || 'segment'; }
function setInferStatus(text) { $('inferStatus').textContent = text; }
function inferNumber(id) { return Number($(id).value); }
function inferChecked(id) { return $(id).checked; }

function syncInferTaskUi() {
  const task = inferTask();
  const models = task === 'semantic' ? SEMANTIC_MODELS : SEGMENT_MODELS;
  const current = $('inferModel').value;
  $('inferModel').innerHTML = models.map((model) => `<option value="${escapeAttr(model)}">${escapeHtml(model)}</option>`).join('');
  $('inferModel').value = models.includes(current) ? current : models[0];
  $('inferSavePredictionsText').textContent = task === 'semantic' ? 'Save masks (*.png)' : 'Save labels + boxes';
  inferModelReady = false;
}

async function initializeInferModel() {
  if (!$('inferAutoWeights').checked && !$('inferWeightsPath').value.trim()) {
    inferModelReady = false;
    setInferStatus('Select custom weights to initialize inference');
    return;
  }
  setInferStatus('Initializing model...');
  const result = await api('/api/yolo/infer/init', {
    method: 'POST',
    body: JSON.stringify({
      task: inferTask(),
      model: $('inferModel').value,
      auto_weights: inferChecked('inferAutoWeights'),
      custom_weights: $('inferWeightsPath').value || null,
      device: $('inferDevice').value,
    }),
  });
  inferModelReady = true;
  setInferStatus(result.loaded ? `Model initialized | ${result.device}` : `Model cached | ${result.device}`);
}

async function refreshInferDevices() {
  const data = await api('/api/yolo/devices');
  $('inferDevice').innerHTML = data.devices.map((dev) => `<option value="${escapeAttr(dev.split(' ')[0])}">${escapeHtml(dev)}</option>`).join('');
  setInferStatus(`Devices refreshed | ${data.platform}`);
}

function fitInferZoom() {
  if (!inferWidth || !inferHeight) return;
  const rect = inferWorkarea.getBoundingClientRect();
  const scale = clamp(Math.min((rect.width - 28) / inferWidth, (rect.height - 28) / inferHeight), 0.25, 16);
  inferZoomLevel = scale;
  inferAutoFitZoom = true;
  applyInferZoom();
}

function applyInferZoom() {
  if (!inferWidth || !inferHeight) return;
  inferCanvasWrap.style.transform = `scale(${inferZoomLevel})`;
  inferCanvasShell.style.width = `${Math.round(inferWidth * inferZoomLevel)}px`;
  inferCanvasShell.style.height = `${Math.round(inferHeight * inferZoomLevel)}px`;
  $('inferZoomReadout').textContent = `${Math.round(inferZoomLevel * 100)}%`;
}

function setInferZoom(zoom, manual = true) {
  inferZoomLevel = clamp(zoom, 0.25, 16);
  if (manual) inferAutoFitZoom = false;
  applyInferZoom();
}

function stepInferZoom(direction) {
  const factor = direction > 0 ? 1.25 : 0.8;
  setInferZoom(inferZoomLevel * factor, true);
}

function updateInferSelector() {
  $('inferImageSelect').innerHTML = inferCatalog.map((name, i) => `<option value="${i}">${escapeHtml(name)}</option>`).join('');
  const disabled = inferCatalog.length === 0;
  $('inferImageSelect').disabled = disabled;
  $('inferPrevImage').disabled = disabled;
  $('inferNextImage').disabled = disabled;
  if (!disabled) $('inferImageSelect').value = String(Math.max(0, Math.min(inferIndex, inferCatalog.length - 1)));
}

function inferCanvasImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function loadInferState() {
  inferState = await api('/api/yolo/infer/state');
  inferCatalog = inferState.images.slice();
  inferHasResults = inferState.count > 0;
  inferModelReady = !!inferState.model_ready;
  inferIndex = 0;
  updateInferSelector();
}

function drawInferCanvases(base, overlay, meta) {
  inferWidth = base.naturalWidth;
  inferHeight = base.naturalHeight;
  for (const canvas of [inferImageCanvas, inferOverlayCanvas]) {
    canvas.width = inferWidth;
    canvas.height = inferHeight;
    canvas.style.width = `${inferWidth}px`;
    canvas.style.height = `${inferHeight}px`;
  }
  inferCanvasWrap.style.width = `${inferWidth}px`;
  inferCanvasWrap.style.height = `${inferHeight}px`;
  inferImageCtx.clearRect(0, 0, inferWidth, inferHeight);
  inferOverlayCtx.clearRect(0, 0, inferWidth, inferHeight);
  inferImageCtx.drawImage(base, 0, 0);
  if (overlay) inferOverlayCtx.drawImage(overlay, 0, 0);
  inferCanvasShell.style.aspectRatio = `${inferWidth} / ${inferHeight}`;
  if (inferAutoFitZoom || inferZoomLevel === 1) fitInferZoom();
  else applyInferZoom();
  const lines = [meta.name, meta.summary || '', `Task: ${meta.task}`, `Boxes: ${meta.boxes}`, `Masks: ${meta.masks}`];
  if (meta.classes && meta.classes.length) lines.push(`Classes: ${meta.classes.join(', ')}`);
  if (meta.saved_overlay) lines.push(`Overlay: ${meta.saved_overlay}`);
  if (meta.saved_prediction) lines.push(`Prediction: ${meta.saved_prediction}`);
  if (meta.saved_boxes) lines.push(`Boxes: ${meta.saved_boxes}`);
  $('inferSummary').textContent = lines.filter(Boolean).join('\n');
}

async function scanInferImages() {
  const dir = $('inferImagesDir').value.trim();
  if (!dir) {
    inferCatalog = [];
    inferHasResults = false;
    inferIndex = 0;
    updateInferSelector();
    return;
  }
  const qs = new URLSearchParams({ image_dir: dir });
  const data = await api(`/api/yolo/infer/scan?${qs}`);
  inferCatalog = data.images || [];
  inferHasResults = false;
  inferIndex = 0;
  updateInferSelector();
  setInferStatus(`Loaded ${data.count} image(s) for inference selection`);
  if (inferCatalog.length) await loadInferImage(0);
}

async function loadInferSourceImage(index = inferIndex) {
  if (!inferCatalog.length) return;
  inferIndex = Math.max(0, Math.min(index, inferCatalog.length - 1));
  $('inferImageSelect').value = String(inferIndex);
  const imageName = inferCatalog[inferIndex];
  const qs = new URLSearchParams({ image_dir: $('inferImagesDir').value.trim(), image_name: imageName, t: Date.now() });
  const [base, meta] = await Promise.all([
    inferCanvasImage(`/api/yolo/infer/source/png?${qs}`),
    api(`/api/yolo/infer/source/meta?${qs}`),
  ]);
  drawInferCanvases(base, null, meta);
  setInferStatus(meta.summary || 'Source image loaded');
}

async function loadInferImage(index = inferIndex) {
  if (!inferHasResults || !inferState.count) {
    await loadInferSourceImage(index);
    return;
  }
  inferIndex = Math.max(0, Math.min(index, inferState.count - 1));
  $('inferImageSelect').value = String(inferIndex);
  const stamp = Date.now();
  const [base, overlay, meta] = await Promise.all([
    inferCanvasImage(`/api/yolo/infer/image/${inferIndex}/png?t=${stamp}`),
    inferCanvasImage(`/api/yolo/infer/image/${inferIndex}/overlay?t=${stamp}`),
    api(`/api/yolo/infer/image/${inferIndex}/meta?t=${stamp}`),
  ]);
  drawInferCanvases(base, overlay, meta);
  setInferStatus(meta.summary || 'Inference loaded');
}

async function runInfer() {
  const scope = $('inferScope').value || 'all';
  const selectedName = inferCatalog[inferIndex] || null;
  if (!inferModelReady) await initializeInferModel();
  setInferStatus('Running inference...');
  inferState = await api('/api/yolo/infer/run', {
    method: 'POST',
    body: JSON.stringify({
      image_dir: $('inferImagesDir').value,
      output_dir: $('inferOutputDir').value || null,
      task: inferTask(),
      model: $('inferModel').value,
      auto_weights: inferChecked('inferAutoWeights'),
      custom_weights: $('inferWeightsPath').value || null,
      device: $('inferDevice').value,
      conf: inferNumber('inferConf'),
      imgsz: inferNumber('inferImgsz'),
      scope,
      image_name: scope === 'current' ? selectedName : null,
      save_overlays: inferChecked('inferSaveOverlays'),
      save_predictions: inferChecked('inferSavePredictions'),
    }),
  });
  inferCatalog = inferState.images.slice();
  inferHasResults = inferState.count > 0;
  inferIndex = 0;
  updateInferSelector();
  if (inferState.count) {
    await loadInferImage(0);
    setInferStatus(`Inference complete | ${inferState.count} image(s)`);
  } else {
    $('inferSummary').textContent = 'No inference results.';
    setInferStatus('No inference results');
  }
}

function yoloPayload() {
  return {
    data_yaml: $('yoloDataYaml').value,
    output_dir: $('yoloOutputDir').value,
    run_name: $('yoloRunName').value,
    task: yoloTask(),
    model: $('yoloModel').value,
    auto_weights: yoloChecked('yoloAutoWeights'),
    custom_weights: $('yoloWeightsPath').value || null,
    device: $('yoloDevice').value,
    epochs: yoloNumber('yoloEpochs'),
    imgsz: yoloNumber('yoloImgsz'),
    batch: yoloNumber('yoloBatch'),
    patience: yoloNumber('yoloPatience'),
    cls: yoloNumber('yoloCls'),
    conf: yoloNumber('yoloConf'),
    dropout: yoloNumber('yoloDropout'),
    mask_ratio: yoloNumber('yoloMaskRatio'),
    auto_optimize: yoloChecked('yoloAutoOptimize'),
    optimizer: $('yoloOptimizer').value,
    lr0: yoloNumber('yoloLr0'),
    box: yoloNumber('yoloBox'),
    fliplr: yoloNumber('yoloFlipLR'),
    flipud: yoloNumber('yoloFlipUD'),
    scale: yoloNumber('yoloScale'),
    translate: yoloNumber('yoloTranslate'),
    mosaic: yoloNumber('yoloMosaic'),
    mixup: yoloNumber('yoloMixup'),
    workers: yoloNumber('yoloWorkers'),
    seed: yoloNumber('yoloSeed'),
    save: yoloChecked('yoloSave'),
    plots: yoloChecked('yoloPlots'),
    amp: yoloChecked('yoloAmp'),
    verbose: yoloChecked('yoloVerbose'),
    save_period: yoloChecked('yoloSavePeriod'),
  };
}

async function buildYoloDataset() {
  setYoloStatus('Building dataset...');
  const result = await api('/api/yolo/build-dataset', {
    method: 'POST',
    body: JSON.stringify({
      image_dir: $('yoloImagesDir').value,
      labels_dir: $('yoloLabelsDir').value,
      output_dir: $('yoloDatasetOut').value,
      task: yoloTask(),
      classes: $('yoloClasses').value.split(',').map((value) => value.trim()).filter(Boolean),
      train_pct: yoloNumber('yoloTrainPct'),
      val_pct: yoloNumber('yoloValPct'),
      use_test_split: yoloChecked('yoloUseTestSplit'),
      seed: yoloNumber('yoloDatasetSeed'),
    }),
  });
  $('yoloDataYaml').value = result.yaml_path;
  if (!$('yoloOutputDir').value) $('yoloOutputDir').value = $('yoloDatasetOut').value;
  const normalizedText = result.normalized_images ? ` | normalized ${result.normalized_images} TIFFs to PNG` : '';
  const splitText = result.use_test_split
    ? `train ${result.counts.train}, val ${result.counts.val}, test ${result.counts.test}`
    : `train ${result.counts.train}, val ${result.counts.val}`;
  setYoloStatus(`${result.task} dataset built: ${result.total} pairs | ${splitText}${normalizedText}`);
}

async function startYoloTraining() {
  yoloLogOffset = 0;
  $('yoloLog').textContent = '';
  setYoloStatus('Starting training...');
  await api('/api/yolo/start', { method: 'POST', body: JSON.stringify(yoloPayload()) });
  startYoloPolling();
}

async function stopYoloTraining() {
  await api('/api/yolo/stop', { method: 'POST', body: '{}' });
  await pollYoloStatus();
}

async function loadYoloResults() {
  await api('/api/yolo/load-results', { method: 'POST', body: JSON.stringify({ results_csv: $('yoloResultsPath').value }) });
  yoloLogOffset = 0;
  await pollYoloStatus();
}

function startYoloPolling() {
  if (yoloPoll) clearInterval(yoloPoll);
  yoloPoll = setInterval(() => pollYoloStatus().catch((e) => setYoloStatus(`Status failed: ${e.message}`)), 2500);
  pollYoloStatus().catch((e) => setYoloStatus(`Status failed: ${e.message}`));
}

async function pollYoloStatus() {
  const data = await api(`/api/yolo/status?offset=${yoloLogOffset}`);
  yoloLogOffset = data.log_offset;
  if (data.log && data.log.length) {
    $('yoloLog').textContent += data.log.join('\n') + '\n';
    $('yoloLog').scrollTop = $('yoloLog').scrollHeight;
  }
  $('yoloEpoch').textContent = data.epoch || 'Epoch: -';
  setYoloStatus(data.error ? `${data.status}: ${data.error}` : data.status);
  renderYoloResults(data.results || {});
  if (data.confusion_matrix) {
    $('yoloConfusion').src = `${data.confusion_matrix}&t=${Date.now()}`;
    $('yoloConfusion').classList.remove('hidden');
  } else {
    $('yoloConfusion').classList.add('hidden');
  }
  if (!data.running && yoloPoll && data.status !== 'running') {
    clearInterval(yoloPoll);
    yoloPoll = null;
  }
}

function renderYoloResults(results) {
  if (!results.exists) return;
  lastYoloResults = results;
  const metrics = results.metrics || {};
  const series = results.series || {};
  const semantic = Object.prototype.hasOwnProperty.call(metrics, 'metrics/mIoU') || Object.prototype.hasOwnProperty.call(series, 'metrics/mIoU');
  const lines = [`Rows: ${results.rows}`, results.path || '', ''];
  for (const [key, value] of Object.entries(metrics)) {
    lines.push(`${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`);
  }
  $('yoloMetrics').textContent = lines.join('\n');
  if (semantic) {
    drawChart($('yoloLossChart'), series, [
      ['train/ce_loss', '#2563eb'], ['train/dice_loss', '#16a34a'], ['train/aux_loss', '#f97316'],
      ['val/ce_loss', '#dc2626'], ['val/dice_loss', '#22c55e'], ['val/aux_loss', '#60a5fa'],
    ], { title: 'Semantic Loss', yLabel: 'Loss', xLabel: 'Epoch', logY: true });
    drawChart($('yoloMapChart'), series, [
      ['metrics/mIoU', '#16a34a'], ['metrics/pixel_acc', '#60a5fa'],
    ], { title: 'Semantic Metrics', yLabel: 'Score', xLabel: 'Epoch' });
  } else {
    drawChart($('yoloLossChart'), series, [
      ['train/box_loss', '#2563eb'], ['train/seg_loss', '#16a34a'], ['val/box_loss', '#dc2626'], ['val/seg_loss', '#f97316'],
    ], { title: 'Loss', yLabel: 'Loss', xLabel: 'Epoch', logY: true });
    drawChart($('yoloMapChart'), series, [
      ['metrics/mAP50(M)', '#16a34a'], ['metrics/mAP50-95(M)', '#22c55e'], ['metrics/mAP50(B)', '#2563eb'], ['metrics/mAP50-95(B)', '#60a5fa'],
    ], { title: 'mAP', yLabel: 'Score', xLabel: 'Epoch' });
  }
}

function fitChartCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.round(rect.width || canvas.clientWidth || 520));
  const height = Math.max(220, Math.round(rect.height || canvas.clientHeight || 240));
  const targetWidth = Math.round(width * dpr);
  const targetHeight = Math.round(height * dpr);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function formatChartValue(value) {
  if (!Number.isFinite(value)) return '';
  if (Math.abs(value) >= 100 || Math.abs(value - Math.round(value)) < 1e-6) return `${Math.round(value)}`;
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function drawChart(canvas, series, keys, options) {
  const { title, xLabel, yLabel, logY = false } = options;
  const { ctx, width: w, height: h } = fitChartCanvas(canvas);
  const dark = body.dataset.theme === 'dark';
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = dark ? '#121b2c' : '#ffffff';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = dark ? '#24324a' : '#d1d5db';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  const filtered = [];
  const values = [];
  let maxLen = 0;
  for (const [key, color] of keys) {
    const seriesValues = (series[key] || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
    if (!seriesValues.length) continue;
    filtered.push([key, color, seriesValues]);
    values.push(...seriesValues);
    maxLen = Math.max(maxLen, seriesValues.length);
  }

  ctx.fillStyle = dark ? '#edf3ff' : '#111827';
  ctx.font = '13px system-ui';
  ctx.fillText(title, 12, 20);
  if (!filtered.length || maxLen < 1) {
    ctx.fillStyle = dark ? '#9fb0c7' : '#6b7280';
    ctx.fillText('No results yet', 12, 46);
    return;
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (logY) {
    const positive = values.filter((value) => value > 0);
    if (!positive.length) {
      ctx.fillStyle = dark ? '#9fb0c7' : '#6b7280';
      ctx.fillText('No positive loss values yet', 12, 46);
      return;
    }
    min = Math.min(...positive);
    max = Math.max(...positive);
    if (max <= min) max = min * 1.5;
  } else {
    if (max <= min) max = min + 1;
    const pad = (max - min) * 0.08;
    min -= pad;
    max += pad;
  }

  const legendHeight = 18 * Math.ceil(filtered.length / 2);
  const left = 54;
  const right = 14;
  const top = 38 + legendHeight;
  const bottom = 42;
  const plotW = Math.max(10, w - left - right);
  const plotH = Math.max(10, h - top - bottom);

  ctx.fillStyle = dark ? '#9fb0c7' : '#4b5563';
  ctx.font = '11px system-ui';
  filtered.forEach(([key, color], idx) => {
    const col = idx % 2;
    const row = Math.floor(idx / 2);
    const lx = 12 + col * Math.max(130, (w - 24) / 2);
    const ly = 34 + row * 16;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 8, 10, 10);
    ctx.fillStyle = dark ? '#cbd5e1' : '#475569';
    ctx.fillText(key, lx + 16, ly);
  });

  ctx.strokeStyle = dark ? '#24324a' : '#e5e7eb';
  ctx.fillStyle = dark ? '#9fb0c7' : '#6b7280';
  ctx.font = '11px system-ui';
  const yTicks = 4;
  if (logY) {
    const minLog = Math.log10(min);
    const maxLog = Math.log10(max);
    for (let i = 0; i <= yTicks; i++) {
      const t = i / yTicks;
      const y = top + (1 - t) * plotH;
      const val = 10 ** (minLog + t * (maxLog - minLog));
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(w - right, y);
      ctx.stroke();
      ctx.fillText(formatChartValue(val), 6, y + 4);
    }
  } else {
    for (let i = 0; i <= yTicks; i++) {
      const t = i / yTicks;
      const y = top + (1 - t) * plotH;
      const val = min + t * (max - min);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(w - right, y);
      ctx.stroke();
      ctx.fillText(formatChartValue(val), 6, y + 4);
    }
  }

  const xTicks = Math.min(4, Math.max(1, maxLen - 1));
  for (let i = 0; i <= xTicks; i++) {
    const t = xTicks === 0 ? 0 : i / xTicks;
    const x = left + t * plotW;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + plotH);
    ctx.stroke();
    const epoch = maxLen === 1 ? 1 : Math.round(1 + t * (maxLen - 1));
    ctx.fillText(String(epoch), x - 5, h - 20);
  }

  ctx.strokeStyle = dark ? '#64748b' : '#94a3b8';
  ctx.beginPath();
  ctx.moveTo(left, top + plotH);
  ctx.lineTo(w - right, top + plotH);
  ctx.moveTo(left, top);
  ctx.lineTo(left, top + plotH);
  ctx.stroke();

  filtered.forEach(([key, color, seriesValues]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    seriesValues.forEach((value, i) => {
      if (logY && value <= 0) return;
      const x = left + ((maxLen <= 1 ? 0 : i / (maxLen - 1)) * plotW);
      const y = logY
        ? top + plotH - ((Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min))) * plotH
        : top + plotH - ((value - min) / (max - min)) * plotH;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    if (started) ctx.stroke();
  });

  ctx.save();
  ctx.fillStyle = dark ? '#9fb0c7' : '#6b7280';
  ctx.font = '11px system-ui';
  ctx.fillText(xLabel, left + plotW / 2 - 16, h - 6);
  ctx.translate(12, top + plotH / 2 + 16);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

for (const button of document.querySelectorAll('.segmented button')) {
  button.addEventListener('click', () => {
    mode = button.dataset.mode;
    document.querySelectorAll('.segmented button').forEach((item) => item.classList.toggle('active', item === button));
    setStatus(`Mode: ${mode}`);
  });
}

$('openFolder').addEventListener('click', async () => {
  try {
    const result = await api('/api/open-folder', { method: 'POST', body: JSON.stringify({ image_dir: $('imageDir').value, out_dir: $('outDir').value || null }) });
    imageSessions.clear();
    state = result;
    points = [];
    boxes = [];
    drawingBox = null;
    labelMap = new Uint16Array(0);
    if (!$('yoloImagesDir').value) $('yoloImagesDir').value = $('imageDir').value;
    if (!$('yoloOutputDir').value && $('outDir').value) $('yoloOutputDir').value = $('outDir').value;
    syncYoloTaskUi();
    await refreshState();
    await loadImage(0);
    const restored = result.existing_outputs ? ` | restored ${result.existing_outputs} saved annotation set(s)` : '';
    setStatus(`Opened ${state.count} image(s); converted ${result.converted || 0}${restored}`);
  } catch (e) {
    setStatus(`Open failed: ${e.message}`);
  }
});

$('loadWeights').addEventListener('click', async () => {
  try {
    await api('/api/load-weights', { method: 'POST', body: JSON.stringify({ weights: $('weightsPath').value, device: $('device').value }) });
    setStatus('SAM weights loaded');
  } catch (e) {
    setStatus(`Weights failed: ${e.message}`);
  }
});

$('setClasses').addEventListener('click', async () => {
  try {
    await api('/api/classes', { method: 'POST', body: JSON.stringify({ classes: $('classesText').value.split(/\r?\n/) }) });
    await refreshState();
    setStatus('Classes updated');
  } catch (e) {
    setStatus(`Classes failed: ${e.message}`);
  }
});

$('imageSelect').addEventListener('change', () => loadImage(Number($('imageSelect').value)));
$('prevImage').addEventListener('click', () => loadImage(currentIndex - 1));
$('nextImage').addEventListener('click', () => loadImage(currentIndex + 1));
$('pixelScale').addEventListener('change', () => loadImage(currentIndex));
$('filterName').addEventListener('change', () => loadImage(currentIndex));
$('zoomIn').addEventListener('click', () => stepZoom(1));
$('zoomOut').addEventListener('click', () => stepZoom(-1));
$('zoomFit').addEventListener('click', () => fitZoom());
$('zoomReset').addEventListener('click', () => setZoom(1, true));
$('undoAction').addEventListener('click', undoLast);
$('clearPoints').addEventListener('click', clearPointsOnly);
$('clearBoxes').addEventListener('click', clearBoxesOnly);
$('clearPrompts').addEventListener('click', clearPromptsOnly);
$('clearLabels').addEventListener('click', clearLabelsOnly);
$('clearAll').addEventListener('click', clearAllState);
$('runSam').addEventListener('click', () => runSam().catch((e) => setStatus(`SAM failed: ${e.message}`)));
$('saveYolo').addEventListener('click', () => saveYolo().catch((e) => setStatus(`Save failed: ${e.message}`)));
$('themeToggle').addEventListener('click', () => applyTheme(body.dataset.theme === 'dark' ? 'light' : 'dark'));

$('browseImages').addEventListener('click', () => openPicker('imageDir', 'image_dir', $('imageDir').value).catch((e) => setStatus(`Browse failed: ${e.message}`)));
$('browseOutput').addEventListener('click', () => openPicker('outDir', 'out_dir', $('outDir').value || $('imageDir').value).catch((e) => setStatus(`Browse failed: ${e.message}`)));
$('imageDir').addEventListener('change', () => clearDraftSessions('Image folder changed. Click Open to load that folder.'));
$('outDir').addEventListener('change', () => clearDraftSessions('Output folder changed. Click Open to reload saved labels from that folder.'));
$('browseWeights').addEventListener('click', () => openPicker('weightsPath', 'weights', $('weightsPath').value).catch((e) => setStatus(`Browse failed: ${e.message}`)));
$('browseClasses').addEventListener('click', () => openPicker('classesPath', 'classes', $('imageDir').value || $('outDir').value).catch((e) => setStatus(`Browse failed: ${e.message}`)));
$('closeFileDialog').addEventListener('click', closePicker);
$('fileDialog').addEventListener('click', (evt) => { if (evt.target === $('fileDialog')) closePicker(); });
$('fileParent').addEventListener('click', () => { if ($('fileParent').dataset.path) loadPickerPath($('fileParent').dataset.path).catch((e) => setStatus(`Browse failed: ${e.message}`)); });
$('selectCurrent').addEventListener('click', () => choosePickerPath(picker.path).catch((e) => setStatus(`Select failed: ${e.message}`)));
$('newFolder').addEventListener('click', () => createFolderInPicker().catch((e) => setStatus(`Create folder failed: ${e.message}`)));
$('fileRoots').addEventListener('click', (evt) => {
  const button = evt.target.closest('button[data-root]');
  if (button) loadPickerPath(button.dataset.root).catch((e) => setStatus(`Browse failed: ${e.message}`));
});
$('fileBreadcrumbs').addEventListener('click', (evt) => {
  const button = evt.target.closest('button[data-path]');
  if (button) loadPickerPath(button.dataset.path).catch((e) => setStatus(`Browse failed: ${e.message}`));
});
$('fileEntries').addEventListener('click', (evt) => {
  const button = evt.target.closest('.file-entry');
  if (!button) return;
  const path = button.dataset.path;
  const isDir = button.dataset.dir === '1';
  const selectable = button.dataset.selectable === '1';
  if (isDir) loadPickerPath(path).catch((e) => setStatus(`Browse failed: ${e.message}`));
  else if (selectable) choosePickerPath(path).catch((e) => setStatus(`Select failed: ${e.message}`));
});

vectorCanvas.addEventListener('contextmenu', (evt) => evt.preventDefault());
vectorCanvas.addEventListener('pointerdown', (evt) => {
  if (!width || !height) return;
  const pos = canvasPos(evt);
  if (mode === 'brush') {
    if (![0, 2].includes(evt.button)) return;
    vectorCanvas.setPointerCapture(evt.pointerId);
    pushUndo(true);
    brushing = true;
    erasing = evt.button === 2;
    paintBrush(pos.x, pos.y, erasing);
    return;
  }
  if (mode === 'points') {
    if (![0, 2].includes(evt.button)) return;
    pushUndo(false);
    points.push({ x: pos.x, y: pos.y, label: evt.button === 2 ? 0 : 1 });
    drawVectors();
    saveCurrentSession();
    updateUndoState();
    return;
  }
  if (mode === 'boxes' || mode === 'concept') {
    if (evt.button !== 0) return;
    vectorCanvas.setPointerCapture(evt.pointerId);
    drawingBox = { x0: pos.x, y0: pos.y, x1: pos.x, y1: pos.y, class_id: classId() };
    drawVectors();
  }
});

vectorCanvas.addEventListener('pointermove', (evt) => {
  if (!width || !height) return;
  const pos = canvasPos(evt);
  if (mode === 'brush' && brushing) {
    paintBrush(pos.x, pos.y, erasing);
    return;
  }
  if ((mode === 'boxes' || mode === 'concept') && drawingBox) {
    drawingBox.x1 = pos.x;
    drawingBox.y1 = pos.y;
    drawVectors();
  }
});

function finishBrush() {
  if (!brushing) return;
  brushing = false;
  erasing = false;
  drawLabels();
  saveCurrentSession();
  updateUndoState();
}

vectorCanvas.addEventListener('pointerup', () => {
  if (mode === 'brush') {
    finishBrush();
    return;
  }
  if ((mode === 'boxes' || mode === 'concept') && drawingBox) {
    const x0 = Math.min(drawingBox.x0, drawingBox.x1);
    const y0 = Math.min(drawingBox.y0, drawingBox.y1);
    const x1 = Math.max(drawingBox.x0, drawingBox.x1);
    const y1 = Math.max(drawingBox.y0, drawingBox.y1);
    if (x1 - x0 >= 2 && y1 - y0 >= 2) {
      pushUndo(false);
      boxes.push({ x0, y0, x1, y1, class_id: drawingBox.class_id });
      saveCurrentSession();
    }
    drawingBox = null;
    drawVectors();
    updateUndoState();
  }
});
vectorCanvas.addEventListener('pointercancel', finishBrush);
vectorCanvas.addEventListener('pointerleave', () => { if (brushing) finishBrush(); });

annotateWorkarea.addEventListener('wheel', (evt) => {
  if (!width || !height) return;
  if (!evt.ctrlKey && evt.target !== vectorCanvas && evt.target !== labelCanvas && evt.target !== imageCanvas) return;
  evt.preventDefault();
  stepZoom(evt.deltaY < 0 ? 1 : -1);
}, { passive: false });

window.addEventListener('resize', () => {
  if (autoFitZoom && width && height) fitZoom();
  if (inferAutoFitZoom && inferWidth && inferHeight) fitInferZoom();
  if (lastYoloResults) renderYoloResults(lastYoloResults);
});
window.addEventListener('keydown', (evt) => {
  if (evt.key === 'Escape' && !$('fileDialog').classList.contains('hidden')) closePicker();
  if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === 'z') {
    evt.preventDefault();
    undoLast();
  }
});

$('tabAnnotate').addEventListener('click', () => showPage('annotate'));
$('tabYolo').addEventListener('click', () => showPage('yolo'));
$('tabInfer').addEventListener('click', () => showPage('infer'));
$('refreshYoloDevices').addEventListener('click', () => refreshYoloDevices().catch((e) => setYoloStatus(e.message)));
$('refreshInferDevices').addEventListener('click', () => refreshInferDevices().catch((e) => setInferStatus(e.message)));
$('yoloTask').addEventListener('change', syncYoloTaskUi);
$('yoloAutoOptimize').addEventListener('change', syncYoloOptimizeUi);
$('yoloUseTestSplit').addEventListener('change', syncYoloSplitUi);
$('yoloTrainPct').addEventListener('change', syncYoloSplitUi);
$('yoloValPct').addEventListener('change', syncYoloSplitUi);
$('buildYoloDataset').addEventListener('click', () => buildYoloDataset().catch((e) => setYoloStatus(`Dataset failed: ${e.message}`)));
$('startYoloTraining').addEventListener('click', () => startYoloTraining().catch((e) => setYoloStatus(`Start failed: ${e.message}`)));
$('stopYoloTraining').addEventListener('click', () => stopYoloTraining().catch((e) => setYoloStatus(`Stop failed: ${e.message}`)));
$('loadYoloResults').addEventListener('click', () => loadYoloResults().catch((e) => setYoloStatus(`Load failed: ${e.message}`)));
$('yoloAutoWeights').addEventListener('change', () => {
  const custom = !$('yoloAutoWeights').checked;
  $('yoloWeightsPath').disabled = !custom;
  $('browseYoloWeights').disabled = !custom;
});
$('inferTask').addEventListener('change', () => {
  syncInferTaskUi();
  initializeInferModel().catch((e) => setInferStatus(`Model init failed: ${e.message}`));
});
$('inferModel').addEventListener('change', () => initializeInferModel().catch((e) => setInferStatus(`Model init failed: ${e.message}`)));
$('inferDevice').addEventListener('change', () => initializeInferModel().catch((e) => setInferStatus(`Model init failed: ${e.message}`)));
$('inferAutoWeights').addEventListener('change', () => {
  const custom = !$('inferAutoWeights').checked;
  $('inferWeightsPath').disabled = !custom;
  $('browseInferWeights').disabled = !custom;
  if (custom) {
    inferModelReady = false;
    setInferStatus('Select custom weights to initialize inference');
  } else {
    initializeInferModel().catch((e) => setInferStatus(`Model init failed: ${e.message}`));
  }
});
$('inferWeightsPath').addEventListener('change', () => initializeInferModel().catch((e) => setInferStatus(`Model init failed: ${e.message}`)));
$('runInfer').addEventListener('click', () => runInfer().catch((e) => setInferStatus(`Inference failed: ${e.message}`)));
$('inferPrevImage').addEventListener('click', () => loadInferImage(inferIndex - 1).catch((e) => setInferStatus(e.message)));
$('inferNextImage').addEventListener('click', () => loadInferImage(inferIndex + 1).catch((e) => setInferStatus(e.message)));
$('inferZoomIn').addEventListener('click', () => stepInferZoom(1));
$('inferZoomOut').addEventListener('click', () => stepInferZoom(-1));
$('inferZoomFit').addEventListener('click', () => fitInferZoom());
$('inferZoomReset').addEventListener('click', () => setInferZoom(1, true));
$('browseYoloImages').addEventListener('click', () => openPicker('yoloImagesDir', 'image_dir', $('yoloImagesDir').value || $('imageDir').value).catch((e) => setYoloStatus(e.message)));
$('browseYoloLabels').addEventListener('click', () => openPicker('yoloLabelsDir', 'out_dir', $('yoloLabelsDir').value || $('outDir').value || $('imageDir').value).catch((e) => setYoloStatus(e.message)));
$('browseYoloClasses').addEventListener('click', () => openPicker('yoloClassesPath', 'classes', $('yoloClassesPath').value || $('yoloLabelsDir').value || $('outDir').value || $('imageDir').value).catch((e) => setYoloStatus(e.message)));
$('browseYoloDatasetOut').addEventListener('click', () => openPicker('yoloDatasetOut', 'out_dir', $('yoloDatasetOut').value || $('outDir').value).catch((e) => setYoloStatus(e.message)));
$('browseYoloOutput').addEventListener('click', () => openPicker('yoloOutputDir', 'out_dir', $('yoloOutputDir').value || $('yoloDatasetOut').value).catch((e) => setYoloStatus(e.message)));
$('browseYoloWeights').addEventListener('click', () => openPicker('yoloWeightsPath', 'weights', $('yoloWeightsPath').value).catch((e) => setYoloStatus(e.message)));
$('browseYoloYaml').addEventListener('click', () => openPicker('yoloDataYaml', 'yaml', $('yoloDataYaml').value || $('yoloDatasetOut').value).catch((e) => setYoloStatus(e.message)));
$('browseYoloResults').addEventListener('click', () => openPicker('yoloResultsPath', 'csv', $('yoloResultsPath').value || $('yoloOutputDir').value).catch((e) => setYoloStatus(e.message)));
$('inferImagesDir').addEventListener('change', () => scanInferImages().catch((e) => setInferStatus(e.message)));
$('inferImageSelect').addEventListener('change', () => {
  inferIndex = Number($('inferImageSelect').value);
  loadInferImage(inferIndex).catch((e) => setInferStatus(e.message));
});
$('browseInferImages').addEventListener('click', () => openPicker('inferImagesDir', 'image_dir', $('inferImagesDir').value || $('imageDir').value).catch((e) => setInferStatus(e.message)));
$('browseInferOutput').addEventListener('click', () => openPicker('inferOutputDir', 'out_dir', $('inferOutputDir').value || $('outDir').value || $('inferImagesDir').value).catch((e) => setInferStatus(e.message)));
$('browseInferWeights').addEventListener('click', () => openPicker('inferWeightsPath', 'weights', $('inferWeightsPath').value).catch((e) => setInferStatus(e.message)));

applyTheme(localStorage.getItem('ultraprompt-theme') || body.dataset.theme || 'dark');
syncYoloTaskUi();
syncYoloSplitUi();
syncInferTaskUi();
refreshState().then(() => {
  if (state.count) loadImage(0);
}).catch((e) => setStatus(`Startup failed: ${e.message}`));
loadInferState().catch(() => {});
refreshYoloDevices().catch(() => {});
refreshInferDevices().then(() => initializeInferModel().catch(() => {})).catch(() => {});
