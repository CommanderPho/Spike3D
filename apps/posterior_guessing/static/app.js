/* 2D posterior guessing frontend */

const state = {
  bundles: [],
  bundleId: null,
  summary: null,
  binIndex: 0,
  nX: 0,
  nY: 0,
  weights: null,
  revealed: false,
  painting: false,
};

const els = {
  bundleSelect: document.getElementById('bundleSelect'),
  binSlider: document.getElementById('binSlider'),
  prevBin: document.getElementById('prevBin'),
  nextBin: document.getElementById('nextBin'),
  binLabel: document.getElementById('binLabel'),
  spikeLabel: document.getElementById('spikeLabel'),
  paintCanvas: document.getElementById('paintCanvas'),
  truthCanvas: document.getElementById('truthCanvas'),
  clearPaint: document.getElementById('clearPaint'),
  revealBtn: document.getElementById('revealBtn'),
  brushSize: document.getElementById('brushSize'),
  brushStrength: document.getElementById('brushStrength'),
  tuningGrid: document.getElementById('tuningGrid'),
  emptyCells: document.getElementById('emptyCells'),
  scoreBox: document.getElementById('scoreBox'),
  scorePrimary: document.getElementById('scorePrimary'),
  scoreSecondary: document.getElementById('scoreSecondary'),
};

const paintCtx = els.paintCanvas.getContext('2d');
const truthCtx = els.truthCanvas.getContext('2d');

function ylOrRd(t) {
  // Approximate YlOrRd
  const stops = [
    [1.0, 1.0, 0.8],
    [0.996, 0.878, 0.545],
    [0.992, 0.682, 0.38],
    [0.89, 0.29, 0.2],
    [0.55, 0.02, 0.15],
  ];
  return sampleStops(stops, t);
}

function blues(t) {
  const stops = [
    [0.97, 0.98, 1.0],
    [0.73, 0.85, 0.92],
    [0.42, 0.68, 0.84],
    [0.19, 0.45, 0.69],
    [0.03, 0.19, 0.42],
  ];
  return sampleStops(stops, t);
}

function sampleStops(stops, t) {
  const x = Math.min(1, Math.max(0, t));
  const scaled = x * (stops.length - 1);
  const i = Math.floor(scaled);
  const f = scaled - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

function renormalize(weights) {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[0].length; j++) {
      if (weights[i][j] < 0) weights[i][j] = 0;
      sum += weights[i][j];
    }
  }
  if (sum <= 0) return null;
  const out = weights.map((row) => row.map((v) => v / sum));
  return out;
}

function zeros2d(nX, nY) {
  return Array.from({ length: nX }, () => Array(nY).fill(0));
}

function paintMass(weights) {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[0].length; j++) sum += Math.max(0, weights[i][j]);
  }
  return sum;
}

function drawMap(ctx, canvas, map, cmap, { blankIfNull = true } = {}) {
  const nX = state.nX;
  const nY = state.nY;
  if (!nX || !nY) return;
  canvas.width = nX;
  canvas.height = nY;
  const img = ctx.createImageData(nX, nY);
  if (!map) {
    if (blankIfNull) {
      for (let k = 0; k < img.data.length; k += 4) {
        img.data[k] = 20;
        img.data[k + 1] = 24;
        img.data[k + 2] = 28;
        img.data[k + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }
    return;
  }
  let maxV = 0;
  for (let i = 0; i < nX; i++) {
    for (let j = 0; j < nY; j++) maxV = Math.max(maxV, map[i][j]);
  }
  if (maxV <= 0) maxV = 1;
  // ImageData is row-major with y downward; our arrays are [x][y]
  for (let iy = 0; iy < nY; iy++) {
    for (let ix = 0; ix < nX; ix++) {
      const t = map[ix][iy] / maxV;
      const [r, g, b] = cmap(t);
      // Flip y for display so low y is at bottom
      const dy = nY - 1 - iy;
      const k = (dy * nX + ix) * 4;
      img.data[k] = Math.round(r * 255);
      img.data[k + 1] = Math.round(g * 255);
      img.data[k + 2] = Math.round(b * 255);
      img.data[k + 3] = map[ix][iy] > 0 ? 230 : 40;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function redrawPaint() {
  const norm = renormalize(state.weights.map((r) => r.slice()));
  drawMap(paintCtx, els.paintCanvas, norm || null, ylOrRd);
  els.revealBtn.disabled = paintMass(state.weights) <= 0 || state.revealed;
}

function eventToBin(evt) {
  const rect = els.paintCanvas.getBoundingClientRect();
  const px = (evt.clientX - rect.left) / rect.width;
  const py = (evt.clientY - rect.top) / rect.height;
  const ix = Math.min(state.nX - 1, Math.max(0, Math.floor(px * state.nX)));
  // Flip y back from display
  const iyDisp = Math.min(state.nY - 1, Math.max(0, Math.floor(py * state.nY)));
  const iy = state.nY - 1 - iyDisp;
  return { ix, iy };
}

function stampBrush(ix, iy) {
  if (state.revealed) return;
  const radius = Number(els.brushSize.value);
  const strength = Number(els.brushStrength.value) / 10;
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dy = -radius; dy <= radius; dy++) {
      const x = ix + dx;
      const y = iy + dy;
      if (x < 0 || y < 0 || x >= state.nX || y >= state.nY) continue;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > radius) continue;
      const w = strength * Math.exp(-0.5 * (dist / Math.max(0.5, radius * 0.55)) ** 2);
      state.weights[x][y] += w;
    }
  }
  redrawPaint();
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function renderTuning(activeCells) {
  els.tuningGrid.innerHTML = '';
  if (!activeCells.length) {
    els.emptyCells.hidden = false;
    return;
  }
  els.emptyCells.hidden = true;
  activeCells.forEach((cell, idx) => {
    const card = document.createElement('div');
    card.className = 'cell-card';
    const label = document.createElement('div');
    label.className = 'label';
    label.innerHTML = `Unit <strong>${cell.neuron_id}</strong> · spikes <strong>${cell.spike_count}</strong>`;
    const plot = document.createElement('div');
    plot.id = `tune-${idx}`;
    plot.style.height = '110px';
    card.appendChild(label);
    card.appendChild(plot);
    els.tuningGrid.appendChild(card);

    const z = cell.tuning_curve;
    // Plotly heatmap expects z as [y][x]; our tuning is [x][y]
    const zT = [];
    for (let y = 0; y < state.nY; y++) {
      const row = [];
      for (let x = 0; x < state.nX; x++) row.push(z[x][y]);
      zT.push(row);
    }
    Plotly.newPlot(
      plot,
      [{
        z: zT,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: false,
      }],
      {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { visible: false },
        yaxis: { visible: false },
      },
      { displayModeBar: false, staticPlot: true, responsive: true },
    );
  });
}

async function loadBin(binIndex) {
  state.binIndex = binIndex;
  state.revealed = false;
  els.truthCanvas.hidden = true;
  els.scoreBox.hidden = true;
  const data = await api(`/api/bundles/${encodeURIComponent(state.bundleId)}/bins/${binIndex}`);
  state.nX = data.n_x;
  state.nY = data.n_y;
  state.weights = zeros2d(state.nX, state.nY);
  els.binSlider.value = String(binIndex);
  els.binLabel.textContent = `Bin ${binIndex} / ${state.summary.n_time - 1} · t=${data.time_bin_center.toFixed(3)}s`;
  els.spikeLabel.textContent = `Spikes ${data.total_spikes} · ${data.active_cells.length} cells`;
  redrawPaint();
  renderTuning(data.active_cells);
}

async function selectBundle(bundleId) {
  state.bundleId = bundleId;
  state.summary = await api(`/api/bundles/${encodeURIComponent(bundleId)}`);
  els.binSlider.min = '0';
  els.binSlider.max = String(Math.max(0, state.summary.n_time - 1));
  await loadBin(0);
}

async function reveal() {
  if (state.revealed || paintMass(state.weights) <= 0) return;
  const body = {
    user_weights: state.weights,
    save: true,
  };
  const result = await api(`/api/bundles/${encodeURIComponent(state.bundleId)}/bins/${state.binIndex}/reveal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  state.revealed = true;
  els.revealBtn.disabled = true;
  drawMap(paintCtx, els.paintCanvas, result.user_posterior, ylOrRd);
  drawMap(truthCtx, els.truthCanvas, result.true_posterior, blues);
  els.truthCanvas.hidden = false;
  els.scoreBox.hidden = false;
  els.scorePrimary.textContent = `Hellinger affinity ${result.scores.hellinger_affinity.toFixed(3)}`;
  els.scoreSecondary.textContent = `Cosine ${result.scores.cosine_similarity.toFixed(3)}`;
}

async function init() {
  const listing = await api('/api/bundles');
  state.bundles = listing.bundles.filter((b) => !b.error);
  els.bundleSelect.innerHTML = '';
  if (!state.bundles.length) {
    els.bundleSelect.innerHTML = '<option value="">No bundles found</option>';
    els.binLabel.textContent = 'Run: python scripts/export_posterior_guessing_bundle.py --synthetic';
    return;
  }
  state.bundles.forEach((b) => {
    const opt = document.createElement('option');
    opt.value = b.bundle_id;
    opt.textContent = `${b.bundle_id} (${b.n_time} bins)`;
    els.bundleSelect.appendChild(opt);
  });
  els.bundleSelect.addEventListener('change', () => selectBundle(els.bundleSelect.value));
  els.binSlider.addEventListener('input', () => loadBin(Number(els.binSlider.value)));
  els.prevBin.addEventListener('click', () => {
    const next = Math.max(0, state.binIndex - 1);
    loadBin(next);
  });
  els.nextBin.addEventListener('click', () => {
    const next = Math.min(state.summary.n_time - 1, state.binIndex + 1);
    loadBin(next);
  });
  els.clearPaint.addEventListener('click', () => {
    if (state.revealed) return;
    state.weights = zeros2d(state.nX, state.nY);
    redrawPaint();
  });
  els.revealBtn.addEventListener('click', () => reveal().catch((e) => alert(e.message)));

  els.paintCanvas.addEventListener('mousedown', (evt) => {
    state.painting = true;
    const { ix, iy } = eventToBin(evt);
    stampBrush(ix, iy);
  });
  els.paintCanvas.addEventListener('mousemove', (evt) => {
    if (!state.painting) return;
    const { ix, iy } = eventToBin(evt);
    stampBrush(ix, iy);
  });
  window.addEventListener('mouseup', () => { state.painting = false; });
  els.paintCanvas.addEventListener('mouseleave', () => { state.painting = false; });

  window.addEventListener('keydown', (evt) => {
    if (evt.key === 'Enter') {
      evt.preventDefault();
      reveal().catch((e) => alert(e.message));
    } else if (evt.key === 'ArrowLeft') {
      els.prevBin.click();
    } else if (evt.key === 'ArrowRight') {
      els.nextBin.click();
    }
  });

  await selectBundle(state.bundles[0].bundle_id);
}

init().catch((err) => {
  console.error(err);
  els.binLabel.textContent = `Failed to start: ${err.message}`;
});
