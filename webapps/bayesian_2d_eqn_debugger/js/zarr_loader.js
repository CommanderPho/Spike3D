/**
 * Load bayesian_2d_eqn_debugger Zarr v2 stores.
 * Group listing uses plain fetch (no zarrita) so the UI works even if the CDN fails.
 * Array loads use a dynamic zarrita import.
 */
import { reshapeTuningCurves } from "./math.js";

const DEFAULT_STORE_URL = "./data/bayesian_2d_eqn_debugger.zarr";
const ZARRITA_CDN = "https://cdn.jsdelivr.net/npm/zarrita/+esm";

let _zarrMod = null;

async function getZarr() {
  if (_zarrMod) return _zarrMod;
  try {
    _zarrMod = await import(ZARRITA_CDN);
    return _zarrMod;
  } catch (e) {
    throw new Error(
      `Failed to load zarrita from CDN (${ZARRITA_CDN}). Check network/adblock. Underlying: ${e.message || e}`,
    );
  }
}

function joinUrl(base, ...parts) {
  let url = String(base).replace(/\/+$/, "");
  for (const p of parts) {
    const seg = String(p).replace(/^\/+|\/+$/g, "");
    if (seg) url = `${url}/${seg}`;
  }
  return url;
}

/**
 * @param {string} storeUrl
 * @param {string} [groupKey]
 */
export async function fetchZattrs(storeUrl, groupKey = "") {
  const url = groupKey
    ? joinUrl(storeUrl, groupKey, ".zattrs")
    : joinUrl(storeUrl, ".zattrs");
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${url} (${resp.status})`);
  }
  return await resp.json();
}

async function fetchJsonOptional(url) {
  const resp = await fetch(url);
  if (!resp.ok) return null;
  try {
    return await resp.json();
  } catch {
    return null;
  }
}

async function groupExists(storeUrl, groupKey) {
  const resp = await fetch(joinUrl(storeUrl, groupKey, ".zgroup"), { method: "GET" });
  return resp.ok;
}

/**
 * List export groups without zarrita (plain HTTP).
 * Prefers ``groups.json`` sidecar, then root ``.zattrs.keys``, filtering to existing subgroups.
 * @param {string} storeUrl
 * @returns {Promise<string[]>}
 */
export async function listGroupKeys(storeUrl) {
  const candidates = [];

  const catalog = await fetchJsonOptional(joinUrl(storeUrl, "groups.json"));
  if (catalog && Array.isArray(catalog.keys)) {
    candidates.push(...catalog.keys.map(String));
  }

  try {
    const attrs = await fetchZattrs(storeUrl);
    if (Array.isArray(attrs.keys)) {
      for (const k of attrs.keys.map(String)) {
        if (!candidates.includes(k)) candidates.push(k);
      }
    }
  } catch (e) {
    console.warn("Could not read root .zattrs", e);
  }

  if (!candidates.length) {
    return [];
  }

  const existing = [];
  for (const k of candidates) {
    if (await groupExists(storeUrl, k)) {
      existing.push(k);
    } else {
      console.warn(`Listed key "${k}" has no .zgroup under store; skipping`);
    }
  }
  return existing;
}

/**
 * @param {string} storeUrl
 */
export async function openRootGroup(storeUrl = DEFAULT_STORE_URL) {
  const zarr = await getZarr();
  const store = new zarr.FetchStore(storeUrl);
  const rootLoc = zarr.root(store);
  const group = await zarr.open.v2(rootLoc, { kind: "group" });
  return { zarr, store, rootLoc, group };
}

/**
 * @param {*} parentGroup zarrita Group
 * @param {string} name
 * @param {*} zarr zarrita module
 */
async function getArray(zarr, parentGroup, name) {
  const arr = await zarr.open.v2(parentGroup.resolve(name), { kind: "array" });
  const view = await zarr.get(arr);
  return { data: view.data, shape: [...arr.shape], dtype: arr.dtype };
}

function typedToNumberArray(data) {
  return Array.from(data, (v) => Number(v));
}

/**
 * @param {string} storeUrl
 * @param {string} groupKey
 */
export async function loadEqnDebuggerPayload(storeUrl, groupKey) {
  const zarr = await getZarr();
  const { group: root } = await openRootGroup(storeUrl);
  const keys = await listGroupKeys(storeUrl);
  const child = await zarr.open.v2(root.resolve(groupKey), { kind: "group" });
  const attrs = await fetchZattrs(storeUrl, groupKey);

  const tcArr = await getArray(zarr, child, "tuning_curves");
  const neuronArr = await getArray(zarr, child, "neuron_ids");
  const xbinArr = await getArray(zarr, child, "xbin");
  const ybinArr = await getArray(zarr, child, "ybin");
  const seedArr = await getArray(zarr, child, "seed_n");

  const [nCells, nx, ny] = tcArr.shape.map((d) => Number(d));
  const tuning_curves = reshapeTuningCurves(tcArr.data, nCells, nx, ny);

  const payload = {
    groupKey,
    availableKeys: keys.length ? keys : [groupKey],
    attrs,
    tau: Number(attrs.tau),
    is_dst: Boolean(attrs.is_dst),
    n_cells: nCells,
    nx,
    ny,
    neuron_ids: typedToNumberArray(neuronArr.data).map((x) => Math.trunc(x)),
    xbin: typedToNumberArray(xbinArr.data),
    ybin: typedToNumberArray(ybinArr.data),
    seed_n: typedToNumberArray(seedArr.data).map((x) => Math.trunc(x)),
    tuning_curves,
    max_spikes_per_cell: Number(attrs.max_spikes_per_cell ?? 15),
    show_log_likelihood: Boolean(attrs.show_log_likelihood ?? true),
    drop_negative_contributing_terms_mode: Boolean(
      attrs.drop_negative_contributing_terms_mode ?? true,
    ),
    reliability_active: null,
    reliability_silent: null,
  };

  if (payload.is_dst) {
    try {
      const ra = await getArray(zarr, child, "reliability_active");
      const rs = await getArray(zarr, child, "reliability_silent");
      payload.reliability_active = typedToNumberArray(ra.data);
      payload.reliability_silent = typedToNumberArray(rs.data);
    } catch (e) {
      console.warn("DST group missing reliability arrays", e);
      payload.is_dst = false;
    }
  }

  return payload;
}

export { DEFAULT_STORE_URL };
