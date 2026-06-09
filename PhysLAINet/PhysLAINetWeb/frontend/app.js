const bands = [
  ["red", "Red"],
  ["green", "Green"],
  ["blue", "Blue"],
  ["rededge", "RedEdge"],
  ["nir", "NIR"],
];

const defaultTransform = () => ({
  tx: 0,
  ty: 0,
  scale: 1,
  rotation: 0,
});

const state = {
  jobId: null,
  mode: "idle",
  editMode: "adjust",
  editTarget: "lag",
  images: { current: null, lag: null },
  transforms: { current: defaultTransform(), lag: defaultTransform() },
  rois: [],
  results: [],
  drawing: null,
  dragging: null,
  activeRect: null,
  panelIndex: 0,
  panelItems: [],
  panelRois: { current: {}, lag: {} },
};

const $ = (id) => document.getElementById(id);
const canvas = $("imageCanvas");
const ctx = canvas.getContext("2d");

function initInputs() {
  document.querySelectorAll(".grid-inputs").forEach((container) => {
    const period = container.dataset.period;
    const periodLabel = period === "current" ? "Current" : "Lag";
    bands.forEach(([key, label]) => {
      container.insertAdjacentHTML(
        "beforeend",
        `<label class="file-row">
          <span>${periodLabel} ${label}</span>
          <input id="${period}_${key}" type="file" accept=".tif,.tiff,image/tiff" />
        </label>
        <label class="file-row">
          <span>${periodLabel} Radiometric Calibration ${label}</span>
          <input id="${period}_panel_${key}" type="file" accept=".tif,.tiff,image/tiff" />
        </label>`,
      );
    });
  });
  document.querySelectorAll(".channel-map").forEach((container) => {
    const period = container.dataset.period;
    const defaults = { red: 1, green: 2, blue: 3, rededge: 4, nir: 5 };
    bands.forEach(([key, label]) => {
      const options = Array.from({ length: 6 }, (_, index) => {
        const value = index + 1;
        return `<option value="${value}" ${value === defaults[key] ? "selected" : ""}>Band ${value}</option>`;
      }).join("");
      container.insertAdjacentHTML(
        "beforeend",
        `<label class="file-row">
          <span>${label}</span>
          <select id="${period}_map_${key}">${options}</select>
        </label>`,
      );
    });
  });
}

function apiBase() {
  return $("apiBase").value.replace(/\/$/, "");
}

function requireFile(id, formData) {
  const input = $(id);
  if (!input.files.length) {
    throw new Error(`Please select file: ${id}`);
  }
  formData.append(id, input.files[0]);
}

function appendChannelMap(period, formData) {
  bands.forEach(([key]) => {
    formData.append(`${period}_${key}`, $(`${period}_map_${key}`).value);
  });
}

function updateUploadMode() {
  const calibrated = $("uploadMode").value === "calibrated";
  $("rawUploadFields").hidden = calibrated;
  $("calibratedUploadFields").hidden = !calibrated;
  $("panelStatus").textContent = calibrated
    ? "Upload calibrated multiband TIFs and select channel mapping"
    : "Select the whiteboard area one by one after uploading";
}

function updateProgress(progress, step) {
  $("progressBar").style.width = `${progress}%`;
  $("progressText").textContent = `${progress}%`;
  $("stepText").textContent = step;
}

function buildPanelItems() {
  return ["current", "lag"].flatMap((period) =>
    bands.map(([band, label]) => ({
      period,
      band,
      label: `${period === "current" ? "Current" : "Lag"} Radiometric Calibration ${label}`,
    })),
  );
}

function setCanvasVisible(visible) {
  canvas.style.display = visible ? "block" : "none";
  $("emptyState").style.display = visible ? "none" : "block";
}

function setRegistrationControls(enabled) {
  ["editTarget", "modeAdjust", "modeRoi", "resetTransform", "scaleControl", "rotationControl"].forEach((id) => {
    $(id).disabled = !enabled;
  });
}

function setEditMode(mode) {
  state.editMode = mode;
  $("modeAdjust").classList.toggle("primary", mode === "adjust");
  $("modeRoi").classList.toggle("primary", mode === "roi");
  canvas.classList.toggle("move-cursor", mode === "adjust");
  canvas.classList.toggle("draw-cursor", mode === "roi");
  $("panelStatus").textContent =
    mode === "adjust"
      ? "Drag the selected image, or use scale/rotate/opacity to align both periods"
      : "Draw one ROI on the aligned overlay; existing ROIs can be dragged or deleted";
}

function syncTransformControls() {
  const transform = state.transforms[state.editTarget];
  $("scaleControl").value = transform.scale;
  $("rotationControl").value = transform.rotation;
}

async function submitJob() {
  const formData = new FormData();
  const calibrated = $("uploadMode").value === "calibrated";
  try {
    if (calibrated) {
      requireFile("current_calibrated", formData);
      requireFile("lag_calibrated", formData);
      appendChannelMap("current", formData);
      appendChannelMap("lag", formData);
    } else {
      ["current", "lag"].forEach((period) => {
        bands.forEach(([key]) => {
          requireFile(`${period}_${key}`, formData);
          requireFile(`${period}_panel_${key}`, formData);
        });
      });
    }
    requireFile("phenology", formData);
  } catch (error) {
    alert(error.message);
    return;
  }

  $("submitJob").disabled = true;
  $("predict").disabled = true;
  $("savePanelRoi").hidden = true;
  state.rois = [];
  state.results = [];
  state.transforms = { current: defaultTransform(), lag: defaultTransform() };
  state.panelRois = { current: {}, lag: {} };
  resetDownloads();
  setRegistrationControls(false);
  updateProgress(5, calibrated ? "Upload Calibrated Multiband Images" : "Upload File");

  try {
    const response = await fetch(`${apiBase()}${calibrated ? "/api/jobs/calibrated" : "/api/jobs"}`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error(await response.text());
    const job = await response.json();
    state.jobId = job.id;
    updateProgress(job.progress, job.step);
    if (calibrated) {
      pollJob();
    } else {
      startPanelSelection();
    }
  } catch (error) {
    $("submitJob").disabled = false;
    updateProgress(0, "Upload failed");
    alert(error.message);
  }
}

function startPanelSelection() {
  state.mode = "panel";
  state.panelIndex = 0;
  state.panelItems = buildPanelItems();
  state.activeRect = null;
  $("savePanelRoi").hidden = false;
  $("savePanelRoi").disabled = true;
  $("clearRois").disabled = true;
  $("predict").disabled = true;
  setRegistrationControls(false);
  loadCurrentPanel();
}

async function loadCurrentPanel() {
  const item = state.panelItems[state.panelIndex];
  if (!item) {
    await startProcessing();
    return;
  }
  state.activeRect = null;
  const count = `${state.panelIndex + 1}/${state.panelItems.length}`;
  $("panelStatus").textContent = `Draw ${count}: ${item.label}`;
  updateProgress(12, `Draw Radiometric Calibration ROI ${count}`);
  const image = await loadImageElement(`${apiBase()}/api/jobs/${state.jobId}/panel-preview/${item.period}/${item.band}`);
  state.images.current = image;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  setCanvasVisible(true);
  draw();
}

async function startProcessing() {
  $("savePanelRoi").hidden = true;
  $("clearRois").disabled = false;
  $("panelStatus").textContent = "Radiometric Calibration ROI completed, processing images...";
  updateProgress(14, "Start Processing");
  const response = await fetch(`${apiBase()}/api/jobs/${state.jobId}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.panelRois),
  });
  if (!response.ok) {
    alert(await response.text());
    $("submitJob").disabled = false;
    return;
  }
  pollJob();
}

async function pollJob() {
  if (!state.jobId) return;
  const response = await fetch(`${apiBase()}/api/jobs/${state.jobId}`);
  const job = await response.json();
  updateProgress(job.progress, job.step);

  if (job.status === "failed") {
    $("submitJob").disabled = false;
    alert(job.message || "Process failed");
    return;
  }
  if (job.status !== "finished" || !job.preview_url) {
    setTimeout(pollJob, 1200);
    return;
  }

  state.mode = "roi";
  state.rois = [];
  state.results = [];
  state.transforms = { current: defaultTransform(), lag: defaultTransform() };
  await Promise.all([
    loadPreviewImage("current", `${apiBase()}/api/jobs/${state.jobId}/preview/current`),
    loadPreviewImage("lag", `${apiBase()}/api/jobs/${state.jobId}/preview/lag`),
  ]);
  const current = state.images.current;
  const lag = state.images.lag;
  canvas.width = Math.max(current.naturalWidth, lag.naturalWidth);
  canvas.height = Math.max(current.naturalHeight, lag.naturalHeight);
  state.transforms.current.tx = (canvas.width - current.naturalWidth) / 2;
  state.transforms.current.ty = (canvas.height - current.naturalHeight) / 2;
  state.transforms.lag.tx = (canvas.width - lag.naturalWidth) / 2;
  state.transforms.lag.ty = (canvas.height - lag.naturalHeight) / 2;
  state.editTarget = "lag";
  $("editTarget").value = "lag";
  setCanvasVisible(true);
  setRegistrationControls(true);
  setEditMode("adjust");
  syncTransformControls();
  $("submitJob").disabled = false;
  $("predict").disabled = false;
  draw();
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = `${url}?t=${Date.now()}`;
  });
}

async function loadPreviewImage(period, url) {
  state.images[period] = await loadImageElement(url);
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function normalizeRect(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  return {
    x,
    y,
    width: Math.max(1, Math.abs(end.x - start.x)),
    height: Math.max(1, Math.abs(end.y - start.y)),
  };
}

function getImageCenter(period) {
  const image = state.images[period];
  return { x: image.naturalWidth / 2, y: image.naturalHeight / 2 };
}

function imageToCanvas(period, point) {
  const transform = state.transforms[period];
  const center = getImageCenter(period);
  const angle = (transform.rotation * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const dx = (point.x - center.x) * transform.scale;
  const dy = (point.y - center.y) * transform.scale;
  return {
    x: center.x + transform.tx + dx * cos - dy * sin,
    y: center.y + transform.ty + dx * sin + dy * cos,
  };
}

function canvasToImage(period, point) {
  const transform = state.transforms[period];
  const center = getImageCenter(period);
  const angle = (-transform.rotation * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const dx = point.x - center.x - transform.tx;
  const dy = point.y - center.y - transform.ty;
  return {
    x: center.x + (dx * cos - dy * sin) / transform.scale,
    y: center.y + (dx * sin + dy * cos) / transform.scale,
  };
}

function roiCorners(rect) {
  return [
    { x: rect.x, y: rect.y },
    { x: rect.x + rect.width, y: rect.y },
    { x: rect.x + rect.width, y: rect.y + rect.height },
    { x: rect.x, y: rect.y + rect.height },
  ];
}

function clampPointToImage(period, point) {
  const image = state.images[period];
  return {
    x: Math.max(0, Math.min(image.naturalWidth - 1, point.x)),
    y: Math.max(0, Math.min(image.naturalHeight - 1, point.y)),
  };
}

function buildRoiPayload(rect) {
  return {
    display: {
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    },
    current: roiCorners(rect).map((point) => clampPointToImage("current", canvasToImage("current", point))),
    lag: roiCorners(rect).map((point) => clampPointToImage("lag", canvasToImage("lag", point))),
  };
}

function hitTest(rect, point) {
  return point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y && point.y <= rect.y + rect.height;
}

function findRoiAt(point) {
  for (let index = state.rois.length - 1; index >= 0; index -= 1) {
    if (hitTest(state.rois[index], point)) return index;
  }
  return -1;
}

function colorForLai(lai, alpha) {
  const value = Math.max(0, Math.min(1, lai / 6));
  let r;
  let g;
  let b;
  if (value < 0.5) {
    const t = value / 0.5;
    r = 66 + (244 - 66) * t;
    g = 165 + (208 - 165) * t;
    b = 245 + (63 - 245) * t;
  } else {
    const t = (value - 0.5) / 0.5;
    r = 244 + (231 - 244) * t;
    g = 208 + (76 - 208) * t;
    b = 63 + (60 - 63) * t;
  }
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
}

function drawImageLayer(period, opacity) {
  const image = state.images[period];
  if (!image) return;
  const transform = state.transforms[period];
  const center = getImageCenter(period);
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.translate(center.x + transform.tx, center.y + transform.ty);
  ctx.rotate((transform.rotation * Math.PI) / 180);
  ctx.scale(transform.scale, transform.scale);
  ctx.drawImage(image, -center.x, -center.y);
  ctx.restore();
}

function drawRect(rect, options = {}) {
  ctx.fillStyle = options.fill || "rgba(22, 124, 128, 0.18)";
  ctx.strokeStyle = options.stroke || "#20c4c8";
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 900));
  ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
  ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
}

function drawLabel(rect, label) {
  ctx.font = `${Math.max(14, Math.round(canvas.width / 70))}px Arial`;
  const metrics = ctx.measureText(label);
  ctx.fillStyle = "rgba(16,24,32,0.78)";
  ctx.fillRect(rect.x, Math.max(rect.y - 26, 0), metrics.width + 12, 24);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, rect.x + 6, Math.max(rect.y - 8, 18));
}

function draw() {
  if (!state.images.current) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (state.mode === "panel") {
    ctx.drawImage(state.images.current, 0, 0);
    if (state.activeRect) {
      drawRect(state.activeRect, { fill: "rgba(255, 255, 255, 0.18)", stroke: "#ffffff" });
      drawLabel(state.activeRect, "Radiometric Calibration ROI");
    }
  } else {
    drawImageLayer("current", Number($("currentOpacity").value));
    drawImageLayer("lag", Number($("lagOpacity").value));
    state.rois.forEach((roi, index) => {
      const match = state.results.find((item) => item.roi_id === index + 1);
      drawRect(roi, {
        fill: match ? colorForLai(match.lai, 0.45) : "rgba(22, 124, 128, 0.18)",
        stroke: match ? "rgba(255,255,255,0.95)" : "#20c4c8",
      });
      const label = match && $("showValues").checked ? `ROI ${index + 1}: ${match.lai.toFixed(2)}` : `ROI ${index + 1}`;
      drawLabel(roi, label);
    });
  }

  if (state.drawing) {
    const roi = normalizeRect(state.drawing.start, state.drawing.end);
    ctx.strokeStyle = "#ffffff";
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
    ctx.setLineDash([]);
  }
  $("roiCount").textContent = `ROI: ${state.rois.length}`;
  renderResults();
}

function beginPointer(event) {
  if (state.mode === "idle" || !state.images.current) return;
  const point = canvasPoint(event);

  if (state.mode === "roi" && state.editMode === "adjust") {
    state.dragging = {
      type: "image",
      start: point,
      tx: state.transforms[state.editTarget].tx,
      ty: state.transforms[state.editTarget].ty,
    };
    return;
  }

  if (state.mode === "roi" && state.editMode === "roi") {
    const index = findRoiAt(point);
    if (index >= 0) {
      const rect = state.rois[index];
      state.dragging = {
        type: "roi",
        index,
        offsetX: point.x - rect.x,
        offsetY: point.y - rect.y,
      };
      return;
    }
  }

  state.drawing = { start: point, end: point };
  draw();
}

function updatePointer(event) {
  const point = canvasPoint(event);
  if (state.dragging?.type === "image") {
    const transform = state.transforms[state.editTarget];
    transform.tx = state.dragging.tx + point.x - state.dragging.start.x;
    transform.ty = state.dragging.ty + point.y - state.dragging.start.y;
    resetDownloads();
    draw();
    return;
  }
  if (state.dragging?.type === "roi") {
    const roi = state.rois[state.dragging.index];
    roi.x = Math.max(0, Math.min(point.x - state.dragging.offsetX, canvas.width - roi.width));
    roi.y = Math.max(0, Math.min(point.y - state.dragging.offsetY, canvas.height - roi.height));
    state.results = [];
    resetDownloads();
    draw();
    return;
  }
  if (!state.drawing) return;
  state.drawing.end = point;
  draw();
}

function finishPointer() {
  if (state.dragging) {
    state.dragging = null;
    return;
  }
  if (!state.drawing) return;

  const roi = normalizeRect(state.drawing.start, state.drawing.end);
  if (roi.width > 4 && roi.height > 4) {
    if (state.mode === "panel") {
      state.activeRect = {
        x: Math.round(roi.x),
        y: Math.round(roi.y),
        width: Math.round(roi.width),
        height: Math.round(roi.height),
      };
      $("savePanelRoi").disabled = false;
    } else if (state.mode === "roi" && state.editMode === "roi") {
      state.rois.push(roi);
      state.results = [];
      resetDownloads();
    }
  }
  state.drawing = null;
  draw();
}

function bindCanvas() {
  canvas.addEventListener("mousedown", beginPointer);
  canvas.addEventListener("mousemove", updatePointer);
  window.addEventListener("mouseup", finishPointer);
}

function savePanelRoi() {
  if (!state.activeRect) return;
  const item = state.panelItems[state.panelIndex];
  state.panelRois[item.period][item.band] = state.activeRect;
  state.panelIndex += 1;
  $("savePanelRoi").disabled = true;
  loadCurrentPanel();
}

function resetDownloads() {
  $("excelLink").classList.add("disabled");
  $("csvLink").classList.add("disabled");
  $("downloadPng").disabled = true;
  $("resultTable").querySelector("tbody").innerHTML = "";
}

async function predict() {
  if (!state.jobId) {
    alert("Please complete image processing first");
    return;
  }
  if (!state.rois.length) {
    alert("Please draw at least one ROI");
    return;
  }
  $("predict").disabled = true;
  updateProgress(88, "Feature Calculation and Model Inference");
  try {
    const payloadRois = state.rois.map(buildRoiPayload);
    const response = await fetch(`${apiBase()}/api/jobs/${state.jobId}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rois: payloadRois }),
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    state.results = payload.results;
    renderResults();
    $("excelLink").href = `${apiBase()}${payload.excel_url}`;
    $("csvLink").href = `${apiBase()}${payload.csv_url}`;
    $("excelLink").classList.remove("disabled");
    $("csvLink").classList.remove("disabled");
    $("downloadPng").disabled = false;
    updateProgress(100, "Model inference completed");
    draw();
  } catch (error) {
    alert(error.message);
  } finally {
    $("predict").disabled = false;
  }
}

function renderResults() {
  const body = $("resultTable").querySelector("tbody");
  body.innerHTML = "";
  state.rois.forEach((roi, index) => {
    const row = state.results.find((item) => item.roi_id === index + 1);
    body.insertAdjacentHTML(
      "beforeend",
      `<tr>
        <td>${index + 1}</td>
        <td>${Math.round(roi.x)}</td>
        <td>${Math.round(roi.y)}</td>
        <td>${Math.round(roi.width)}</td>
        <td>${Math.round(roi.height)}</td>
        <td>${row ? Number(row.lai).toFixed(3) : "-"}</td>
        <td><button class="delete-roi" data-index="${index}">Delete</button></td>
      </tr>`,
    );
  });
}

function deleteRoi(index) {
  state.rois.splice(index, 1);
  state.results = [];
  resetDownloads();
  draw();
}

function downloadPng() {
  draw();
  const link = document.createElement("a");
  link.download = `lai_overlay_${state.jobId || "preview"}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

function resetSelectedTransform() {
  const image = state.images[state.editTarget];
  state.transforms[state.editTarget] = defaultTransform();
  state.transforms[state.editTarget].tx = (canvas.width - image.naturalWidth) / 2;
  state.transforms[state.editTarget].ty = (canvas.height - image.naturalHeight) / 2;
  state.results = [];
  resetDownloads();
  syncTransformControls();
  draw();
}

function initEvents() {
  $("submitJob").addEventListener("click", submitJob);
  $("uploadMode").addEventListener("change", updateUploadMode);
  $("savePanelRoi").addEventListener("click", savePanelRoi);
  $("predict").addEventListener("click", predict);
  $("clearRois").addEventListener("click", () => {
    if (state.mode !== "roi") return;
    state.rois = [];
    state.results = [];
    resetDownloads();
    draw();
  });
  $("editTarget").addEventListener("change", () => {
    state.editTarget = $("editTarget").value;
    syncTransformControls();
  });
  $("modeAdjust").addEventListener("click", () => setEditMode("adjust"));
  $("modeRoi").addEventListener("click", () => setEditMode("roi"));
  $("resetTransform").addEventListener("click", resetSelectedTransform);
  $("currentOpacity").addEventListener("input", draw);
  $("lagOpacity").addEventListener("input", draw);
  $("scaleControl").addEventListener("input", () => {
    state.transforms[state.editTarget].scale = Number($("scaleControl").value);
    state.results = [];
    resetDownloads();
    draw();
  });
  $("rotationControl").addEventListener("input", () => {
    state.transforms[state.editTarget].rotation = Number($("rotationControl").value);
    state.results = [];
    resetDownloads();
    draw();
  });
  $("showValues").addEventListener("change", draw);
  $("downloadPng").addEventListener("click", downloadPng);
  $("resultTable").addEventListener("click", (event) => {
    if (!event.target.classList.contains("delete-roi")) return;
    deleteRoi(Number(event.target.dataset.index));
  });
}

initInputs();
bindCanvas();
initEvents();
updateUploadMode();
setCanvasVisible(false);
