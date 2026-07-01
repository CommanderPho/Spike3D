---
name: Fix CuPy Env
overview: Normalize the workspace to a single CuPy CUDA wheel and verify NVRTC/kernel compilation, targeting CUDA 13.1 via `cupy-cuda13x` because the current lockfile and `NeuroPy` dependency already point that way.
todos:
  - id: normalize-deps
    content: Normalize dependency declarations to a single CuPy CUDA target (`cupy-cuda13x` without invalid extras).
    status: completed
  - id: sync-env
    content: Run uv lock/sync from a clean shell targeting project `.venv` and reinstall CuPy if needed.
    status: completed
  - id: verify-cupy
    content: Verify one CuPy package, CUDA runtime access, NVRTC loading, and kernel compilation.
    status: completed
  - id: fallback-if-needed
    content: If CUDA 13 verification fails, switch cleanly to the CUDA 12.9 / `cupy-cuda12x` path and re-verify.
    status: cancelled
isProject: false
---

# Fix CuPy CUDA Environment

I’ll target **CUDA 13.1 / `cupy-cuda13x`** unless you tell me otherwise, because the current `NeuroPy` dependency and `uv.lock` already resolve `cupy-cuda13x==13.6.0`, and your terminal showed `cupy-cuda13x` can import when not competing with `cupy-cuda12x`.

Key cleanup:

- In `h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\pyproject.toml`, change `cupy-cuda13x[ctk]>=13.6.0,<14` to `cupy-cuda13x>=13.6.0,<14` because uv reports that `cupy-cuda13x==13.6.0` has no `ctk` extra.
- In `h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\pyproject.toml`, remove the CUDA-12-specific `cutensor-cu12>=2.7.0` dependency unless another package requires it through resolution. This avoids mixing CUDA 12 runtime packages into the CUDA 13 environment.
- Keep `Spike3D` itself from depending on source `cupy`; use only the binary wheel package provided by `NeuroPy`.

Environment sync and validation:

- Start from a clean shell state so uv stops seeing `VIRTUAL_ENV=.venv_310` while operating on project `.venv`.
- Use uv to refresh the lock and sync `.venv`, reinstalling `cupy-cuda13x` if needed.
- Verify exactly one CuPy distribution is installed: `cupy-cuda13x` only, with no `cupy` source package and no `cupy-cuda12x`.
- Verify more than `cp.arange().sum()`: specifically test NVRTC loading and an elementwise/fused kernel compile, since the original failure happened at `nvrtc64_120_0.dll` during kernel compilation.

Validation commands after sync will check:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.metadata as md; print([d.metadata['Name'] for d in md.distributions() if d.metadata['Name'].lower().startswith('cupy')])"
.\.venv\Scripts\python.exe -c "import cupy as cp; from cupy_backends.cuda.libs import nvrtc; print(cp.__version__); print(nvrtc.getVersion()); x=cp.arange(10, dtype=cp.float32); print(float((x*x).sum()))"
```

If CUDA 13 still fails after dependency cleanup, I’ll fall back to the CUDA 12 path by switching both dependency declarations to `cupy-cuda12x>=13.6.0,<14`, removing CUDA 13 remnants, and verifying against the CUDA 12.9 toolkit instead.