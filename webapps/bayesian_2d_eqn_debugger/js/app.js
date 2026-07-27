/**
 * Interactive Bayesian / DST 2D equation debugger (classic script — no ES modules).
 * Loads JSON payloads from ./data/ (written by eqn_debugger_export.py).
 */
(function () {
  "use strict";

  const MathApi = window.BayesEqnMath;
  if (!MathApi) {
    document.getElementById("status").textContent =
      "Failed to load math.js (BayesEqnMath missing).";
    return;
  }

  const {
    poissonFactorMaps,
    dstEiMaps,
    computeConflictMap,
    peakRatesAndExpectedN,
    prepareHeatmapZ,
  } = MathApi;

  const DEFAULT_DATA_DIR = "./data";
  const CELL_COLORS = [
    "#2ca02c",
    "#d62728",
    "#1f77b4",
    "#9467bd",
    "#ff7f0e",
    "#e377c2",
    "#bcbd22",
    "#17becf",
  ];

  const state = {
    payload: null,
    n: [],
    peakRates: [],
    E_n: [],
  };

  function $(id) {
    return document.getElementById(id);
  }

  function setStatus(msg, isError) {
    const el = $("status");
    el.textContent = msg;
    el.classList.toggle("error", !!isError);
  }

  function dataDir() {
    return ($("data-dir").value || DEFAULT_DATA_DIR).replace(/\/+$/, "");
  }

  async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${url} (${resp.status})`);
    }
    return resp.json();
  }

  async function listGroupKeys() {
    const catalog = await fetchJson(`${dataDir()}/groups.json`);
    return Array.isArray(catalog.keys) ? catalog.keys.map(String) : [];
  }

  async function loadGroupPayload(groupKey) {
    const raw = await fetchJson(`${dataDir()}/${groupKey}.json`);
    return {
      groupKey: raw.group_key || groupKey,
      availableKeys: [],
      attrs: raw,
      tau: Number(raw.tau),
      is_dst: Boolean(raw.is_dst),
      n_cells: Number(raw.n_cells),
      nx: Number(raw.nx),
      ny: Number(raw.ny),
      neuron_ids: raw.neuron_ids.map((x) => Math.trunc(x)),
      xbin: raw.xbin,
      ybin: raw.ybin,
      seed_n: raw.seed_n.map((x) => Math.trunc(x)),
      tuning_curves: raw.tuning_curves,
      max_spikes_per_cell: Number(raw.max_spikes_per_cell ?? 15),
      show_log_likelihood: Boolean(raw.show_log_likelihood ?? true),
      drop_negative_contributing_terms_mode: Boolean(
        raw.drop_negative_contributing_terms_mode ?? true,
      ),
      reliability_active: raw.reliability_active || null,
      reliability_silent: raw.reliability_silent || null,
    };
  }

  function heatmapTrace(z, x0, x1, y0, y1, title, colorscale) {
    const nrows = z.length;
    const ncols = z[0].length;
    const xs = Array.from(
      { length: ncols },
      (_, i) => x0 + ((x1 - x0) * (i + 0.5)) / ncols,
    );
    const ys = Array.from(
      { length: nrows },
      (_, i) => y0 + ((y1 - y0) * (i + 0.5)) / nrows,
    );
    return {
      data: [
        {
          type: "heatmap",
          z: z,
          x: xs,
          y: ys,
          colorscale: colorscale || "Viridis",
          showscale: false,
          hoverongaps: false,
        },
      ],
      layout: {
        title: { text: title, font: { size: 12, color: "#e7ecf1" } },
        font: { color: "#e7ecf1" },
        margin: { t: 36, b: 20, l: 20, r: 10 },
        xaxis: { visible: false, scaleanchor: "y", scaleratio: 1 },
        yaxis: { visible: false },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      },
    };
  }

  function ensurePlotHost(id) {
    let el = document.getElementById(id);
    if (!el) {
      el = document.createElement("div");
      el.id = id;
      el.className = "plot";
    }
    return el;
  }

  function renderHeatmap(container, id, M, xbin, ybin, title, opts) {
    opts = opts || {};
    if (typeof Plotly === "undefined") {
      setStatus("Plotly failed to load from CDN.", true);
      return;
    }
    const host = ensurePlotHost(id);
    if (!host.parentElement) container.appendChild(host);
    const z = prepareHeatmapZ(M, !!opts.logScale);
    const packed = heatmapTrace(
      z,
      xbin[0],
      xbin[xbin.length - 1],
      ybin[0],
      ybin[ybin.length - 1],
      title,
      opts.colorscale || "Viridis",
    );
    Plotly.react(host, packed.data, packed.layout, {
      displayModeBar: false,
      responsive: true,
    });
  }

  function buildSliderUI(payload) {
    const wrap = $("sliders");
    wrap.innerHTML = "";
    payload.neuron_ids.forEach(function (aclu, i) {
      const row = document.createElement("div");
      row.className = "slider-row";
      const label = document.createElement("label");
      label.textContent = "n[" + aclu + "]";
      label.style.color = CELL_COLORS[i % CELL_COLORS.length];
      const input = document.createElement("input");
      input.type = "range";
      input.min = "0";
      input.max = String(payload.max_spikes_per_cell);
      input.step = "1";
      input.value = String(state.n[i] != null ? state.n[i] : 1);
      input.dataset.idx = String(i);
      const val = document.createElement("span");
      val.className = "slider-val";
      val.textContent = input.value;
      const expected = document.createElement("span");
      expected.className = "expected";
      expected.textContent =
        "E≈" + (state.E_n[i] != null ? state.E_n[i] : 0).toFixed(2);
      input.addEventListener("input", function () {
        const idx = Number(input.dataset.idx);
        state.n[idx] = Number(input.value);
        val.textContent = input.value;
        redraw();
      });
      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(val);
      row.appendChild(expected);
      wrap.appendChild(row);
    });
  }

  function readSlidersIntoState() {
    $("sliders")
      .querySelectorAll('input[type="range"]')
      .forEach(function (input) {
        state.n[Number(input.dataset.idx)] = Number(input.value);
      });
  }

  function setAllN(values) {
    const maxN = state.payload.max_spikes_per_cell;
    state.n = values.map(function (v) {
      return Math.max(0, Math.min(maxN, Math.trunc(v)));
    });
    $("sliders")
      .querySelectorAll('input[type="range"]')
      .forEach(function (input) {
        const idx = Number(input.dataset.idx);
        input.value = String(state.n[idx]);
        input.parentElement.querySelector(".slider-val").textContent =
          input.value;
      });
    redraw();
  }

  function redraw() {
    const p = state.payload;
    if (!p) return;
    readSlidersIntoState();

    const parts = poissonFactorMaps(
      p.tuning_curves,
      state.n,
      p.tau,
      p.drop_negative_contributing_terms_mode,
    );

    let ei = null;
    if (p.is_dst && p.reliability_active && p.reliability_silent) {
      ei = dstEiMaps(
        p.tuning_curves,
        state.n,
        p.tau,
        p.reliability_active,
        p.reliability_silent,
      );
    }

    const cellCols = $("cell-columns");
    const factorRow = $("row-factors");
    cellCols.innerHTML = "";
    factorRow.innerHTML = "";

    p.neuron_ids.forEach(function (aclu, i) {
      const col = document.createElement("div");
      col.className = "cell-column";
      col.id = "cell-col-" + i;

      const header = document.createElement("div");
      header.className = "cell-column-header";
      header.style.color = CELL_COLORS[i % CELL_COLORS.length];
      header.textContent = "aclu " + aclu + " · n=" + state.n[i];
      col.appendChild(header);
      cellCols.appendChild(col);

      const cmap = ["Greens", "Reds", "Blues", "Purples", "Oranges"][i % 5];
      renderHeatmap(col, "pf-" + i, p.tuning_curves[i], p.xbin, p.ybin, "PF", {
        colorscale: cmap,
      });
      renderHeatmap(col, "Li-" + i, parts.per_cell_L[i], p.xbin, p.ybin, "Lᵢ", {
        colorscale: "Viridis",
      });
      if (ei) {
        const a = ei.alphas[i];
        renderHeatmap(col, "Ei-" + i, ei.per_cell_E[i], p.xbin, p.ybin, "Eᵢ α=" + a.toFixed(3), {
          colorscale: "Cividis",
        });
      }
    });

    if (ei) {
      const conflict = computeConflictMap(ei.per_cell_E);
      renderHeatmap(
        factorRow,
        "conflict",
        conflict.conflict_map,
        p.xbin,
        p.ybin,
        "conflict K̄=" + conflict.K.toFixed(4),
        { colorscale: "Hot" },
      );
    }

    renderHeatmap(factorRow, "post", parts.posterior, p.xbin, p.ybin, "P(x|n)", {
      colorscale: "Viridis",
    });
    renderHeatmap(factorRow, "pow", parts.power_term, p.xbin, p.ybin, "Π (τf)ⁿ", {
      colorscale: "Viridis",
    });
    renderHeatmap(factorRow, "exp", parts.exp_term, p.xbin, p.ybin, "Π e^{-τf}", {
      colorscale: "Viridis",
    });
    renderHeatmap(
      factorRow,
      "Ljoint",
      parts.L,
      p.xbin,
      p.ybin,
      p.show_log_likelihood ? "log₁₀ L" : "L",
      { colorscale: "Viridis", logScale: p.show_log_likelihood },
    );

    $("meta").textContent =
      "τ=" +
      p.tau +
      " · cells=[" +
      p.neuron_ids.join(", ") +
      "] · n=[" +
      state.n.join(", ") +
      "] · DST=" +
      p.is_dst;
  }

  async function applyPayload(payload) {
    state.payload = payload;
    const peaks = peakRatesAndExpectedN(payload.tuning_curves, payload.tau);
    state.peakRates = peaks.peakRates;
    state.E_n = peaks.E_n;
    state.n = payload.seed_n.slice();
    $("chk-log").checked = payload.show_log_likelihood;
    $("chk-drop").checked = payload.drop_negative_contributing_terms_mode;
    buildSliderUI(payload);
    redraw();
    setStatus(
      'Loaded group "' +
        payload.groupKey +
        '" (' +
        payload.n_cells +
        " cells, " +
        payload.nx +
        "×" +
        payload.ny +
        ")",
    );
  }

  async function loadSelectedGroup() {
    const groupKey = $("group-key").value;
    if (!groupKey) {
      setStatus("No group selected. Export a decoder first.", true);
      return;
    }
    setStatus("Loading " + groupKey + " …");
    try {
      const payload = await loadGroupPayload(groupKey);
      await applyPayload(payload);
    } catch (e) {
      console.error(e);
      setStatus("Load failed: " + (e.message || e), true);
    }
  }

  async function refreshGroupList() {
    setStatus("Listing groups in " + dataDir() + " …");
    try {
      const keys = await listGroupKeys();
      const sel = $("group-key");
      sel.innerHTML = "";
      if (!keys.length) {
        setStatus(
          "No groups in data/groups.json. Re-run export_bayesian_2d_eqn_debugger(...).",
          true,
        );
        return;
      }
      keys.forEach(function (k) {
        const opt = document.createElement("option");
        opt.value = k;
        opt.textContent = k;
        sel.appendChild(opt);
      });
      setStatus("Found " + keys.length + " group(s).");
      await loadSelectedGroup();
    } catch (e) {
      console.error(e);
      setStatus("Could not list groups: " + (e.message || e), true);
    }
  }

  function wireControls() {
    $("btn-refresh").addEventListener("click", function () {
      refreshGroupList();
    });
    $("btn-load").addEventListener("click", function () {
      loadSelectedGroup();
    });
    $("group-key").addEventListener("change", function () {
      loadSelectedGroup();
    });
    $("btn-n0").addEventListener("click", function () {
      setAllN(state.n.map(function () {
        return 0;
      }));
    });
    $("btn-n1").addEventListener("click", function () {
      setAllN(state.n.map(function () {
        return 1;
      }));
    });
    $("btn-nE").addEventListener("click", function () {
      setAllN(
        state.E_n.map(function (v) {
          return Math.round(v);
        }),
      );
    });
    $("chk-log").addEventListener("change", function () {
      if (!state.payload) return;
      state.payload.show_log_likelihood = $("chk-log").checked;
      redraw();
    });
    $("chk-drop").addEventListener("change", function () {
      if (!state.payload) return;
      state.payload.drop_negative_contributing_terms_mode = $("chk-drop").checked;
      redraw();
    });
  }

  wireControls();
  refreshGroupList();
})();
