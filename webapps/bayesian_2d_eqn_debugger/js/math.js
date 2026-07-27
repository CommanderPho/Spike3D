/**
 * Math port of InteractiveBayesian2DEquationDebugger / DST helpers.
 * Parity with PendingNotebookCode._poisson_factor_maps, _dst_Ei_maps,
 * _compute_conflict_map, _orient_2d_for_imshow and
 * BayesianPlacemapPositionDecoderDST.iterative_intersection.
 */

/** @param {number} n */
function factorial(n) {
  const k = Math.max(0, Math.trunc(n));
  let out = 1;
  for (let i = 2; i <= k; i++) {
    out *= i;
  }
  return out;
}

/**
 * Rotate 2D array 90° clockwise (numpy rot90(k=-1)), then fliplr.
 * Input M is row-major flat or nested [nx][ny] — we use nested [nx][ny].
 * @param {number[][]} M
 * @returns {number[][]}
 */
function orient2dForImshow(M) {
  const nx = M.length;
  const ny = M[0].length;
  // rot90(k=-1): (i, j) -> (j, nx-1-i)  => shape (ny, nx)
  const rotated = Array.from({ length: ny }, () => new Array(nx));
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      rotated[j][nx - 1 - i] = M[i][j];
    }
  }
  // fliplr on (ny, nx)
  const out = Array.from({ length: ny }, () => new Array(nx));
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      out[r][nx - 1 - c] = rotated[r][c];
    }
  }
  return out;
}

/**
 * @param {Float32Array|number[]} flat flat length nCells*nx*ny in C order
 * @param {number} nCells
 * @param {number} nx
 * @param {number} ny
 * @returns {number[][][]} nested [nCells][nx][ny]
 */
function reshapeTuningCurves(flat, nCells, nx, ny) {
  const out = [];
  let idx = 0;
  for (let c = 0; c < nCells; c++) {
    const cell = [];
    for (let i = 0; i < nx; i++) {
      const row = new Array(ny);
      for (let j = 0; j < ny; j++) {
        row[j] = flat[idx++];
      }
      cell.push(row);
    }
    out.push(cell);
  }
  return out;
}

function zeros2d(nx, ny, fill = 0) {
  return Array.from({ length: nx }, () => Array.from({ length: ny }, () => fill));
}

function ones2d(nx, ny) {
  return zeros2d(nx, ny, 1);
}

function clipFloor(v) {
  if (!Number.isFinite(v) || Number.isNaN(v)) return 1e-12;
  return Math.max(v, 1e-12);
}

function nanSum2d(M) {
  let s = 0;
  for (let i = 0; i < M.length; i++) {
    for (let j = 0; j < M[i].length; j++) {
      const v = M[i][j];
      if (Number.isFinite(v)) s += v;
    }
  }
  return s;
}

function nanMean2d(M) {
  let s = 0;
  let n = 0;
  for (let i = 0; i < M.length; i++) {
    for (let j = 0; j < M[i].length; j++) {
      const v = M[i][j];
      if (Number.isFinite(v)) {
        s += v;
        n += 1;
      }
    }
  }
  return n > 0 ? s / n : NaN;
}

/**
 * @param {number[][][]} tuningCurvesXY [nCells][nx][ny]
 * @param {number[]} spikeCounts
 * @param {number} tau
 * @param {boolean} dropNegativeContributingTermsMode
 */
function poissonFactorMaps(tuningCurvesXY, spikeCounts, tau, dropNegativeContributingTermsMode = true) {
  const nCells = tuningCurvesXY.length;
  const nx = tuningCurvesXY[0].length;
  const ny = tuningCurvesXY[0][0].length;
  if (spikeCounts.length !== nCells) {
    throw new Error(`spikeCounts length ${spikeCounts.length} != nCells ${nCells}`);
  }

  const F = tuningCurvesXY.map((cell) =>
    cell.map((row) => row.map((v) => clipFloor(v))),
  );

  let powerTerm = ones2d(nx, ny);
  let expTerm = ones2d(nx, ny);
  let factorialTerm = 1.0;
  const perCellL = [];

  for (let i = 0; i < nCells; i++) {
    const n_i = Math.trunc(spikeCounts[i]);
    const cellPower = zeros2d(nx, ny);
    const cellExp = zeros2d(nx, ny);
    const cellL = zeros2d(nx, ny);
    const cellFac = 1.0 / factorial(n_i);

    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        const tau_f = tau * F[i][x][y];
        const p = Math.pow(tau_f, n_i);
        const e = Math.exp(-tau_f);
        cellPower[x][y] = p;
        cellExp[x][y] = e;
        cellL[x][y] = p * e * cellFac;
      }
    }

    perCellL.push(cellL);

    if (!(dropNegativeContributingTermsMode && n_i === 0)) {
      for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
          powerTerm[x][y] *= cellPower[x][y];
          expTerm[x][y] *= cellExp[x][y];
        }
      }
      factorialTerm *= cellFac;
    }
  }

  const L = zeros2d(nx, ny);
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      L[x][y] = powerTerm[x][y] * expTerm[x][y] * factorialTerm;
    }
  }
  const Z = nanSum2d(L);
  const posterior = zeros2d(nx, ny, NaN);
  if (Z > 0) {
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        posterior[x][y] = L[x][y] / Z;
      }
    }
  }

  return {
    per_cell_L: perCellL,
    power_term: powerTerm,
    exp_term: expTerm,
    factorial_term: factorialTerm,
    L,
    posterior,
    F,
  };
}

/**
 * @param {number[][][]} tuningCurvesXY
 * @param {number[]} spikeCounts
 * @param {number} tau
 * @param {number[]} reliabilityActive
 * @param {number[]} reliabilitySilent
 */
function dstEiMaps(tuningCurvesXY, spikeCounts, tau, reliabilityActive, reliabilitySilent) {
  const nCells = tuningCurvesXY.length;
  const nx = tuningCurvesXY[0].length;
  const ny = tuningCurvesXY[0][0].length;
  const nBins = nx * ny;
  const F = tuningCurvesXY.map((cell) =>
    cell.map((row) => row.map((v) => clipFloor(v))),
  );
  const perCellE = [];
  const alphas = new Array(nCells);

  for (let i = 0; i < nCells; i++) {
    const n_i = Math.trunc(spikeCounts[i]);
    const L_i = zeros2d(nx, ny);
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        const tau_f = tau * F[i][x][y];
        L_i[x][y] = Math.pow(tau_f, n_i) * Math.exp(-tau_f);
      }
    }
    const Z_i = nanSum2d(L_i);
    const p_i = zeros2d(nx, ny);
    if (Z_i > 0) {
      for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
          p_i[x][y] = L_i[x][y] / Z_i;
        }
      }
    } else {
      const u = 1.0 / nBins;
      for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
          p_i[x][y] = u;
        }
      }
    }
    const alpha_i = n_i > 0 ? Number(reliabilityActive[i]) : Number(reliabilitySilent[i]);
    alphas[i] = alpha_i;
    const E_i = zeros2d(nx, ny);
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        E_i[x][y] = alpha_i * p_i[x][y] + (1.0 - alpha_i);
      }
    }
    perCellE.push(E_i);
  }

  return { per_cell_E: perCellE, alphas };
}

/**
 * Port of BayesianPlacemapPositionDecoderDST.iterative_intersection
 * @param {...number[][]} args
 */
function iterativeIntersection(...args) {
  if (args.length < 2) {
    return args.length === 1 ? args[0] : [];
  }
  const nx = args[0].length;
  const ny = args[0][0].length;
  const out = zeros2d(nx, ny, 0);
  for (let a = 0; a < args.length; a++) {
    const anArg = args[a];
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        const isConflicting = !(out[x][y] > 0) && anArg[x][y] > 0;
        if (isConflicting) {
          out[x][y] = anArg[x][y];
        }
      }
    }
  }
  return out;
}

/**
 * @param {number[][][]} perCellE
 */
function computeConflictMap(perCellE) {
  const conflictMap = iterativeIntersection(...perCellE);
  const K = nanMean2d(conflictMap);
  return { conflict_map: conflictMap, K };
}

/**
 * Peak rates and expected spike counts at peak: E_n = tau * peak_rate
 * @param {number[][][]} tuningCurves
 * @param {number} tau
 */
function peakRatesAndExpectedN(tuningCurves, tau) {
  const peakRates = tuningCurves.map((cell) => {
    let m = -Infinity;
    for (let i = 0; i < cell.length; i++) {
      for (let j = 0; j < cell[i].length; j++) {
        const v = cell[i][j];
        if (Number.isFinite(v) && v > m) m = v;
      }
    }
    return m;
  });
  const E_n = peakRates.map((r) => tau * r);
  return { peakRates, E_n };
}

/**
 * Prepare a map for Plotly heatmap (oriented + optional log10).
 * Returns z as [rows][cols] with origin lower implied by Plotly yaxis.
 * @param {number[][]} M
 * @param {boolean} logScale
 */
function prepareHeatmapZ(M, logScale = false) {
  let A = orient2dForImshow(M);
  if (logScale) {
    A = A.map((row) =>
      row.map((v) => {
        const clipped = Math.max(v, 1e-30);
        return Math.log10(clipped);
      }),
    );
  }
  return A;
}

window.BayesEqnMath = { factorial, orient2dForImshow, reshapeTuningCurves, poissonFactorMaps, dstEiMaps, iterativeIntersection, computeConflictMap, peakRatesAndExpectedN, prepareHeatmapZ };

