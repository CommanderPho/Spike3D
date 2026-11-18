---
name: fix-replayswitchinghmm-packaging
overview: Configure ReplaySwitchingHMM packaging so uv can build it from a flat-layout repo that contains multiple top-level directories (`ssm`, `pbe_io`, `configs`, `submission`).
todos:
  - id: edit-pyproject
    content: Add explicit setuptools build-system and package discovery config in ReplaySwitchingHMM pyproject.toml
    status: completed
  - id: verify-uv-add
    content: Validate package builds with uv add from Spike3D repo
    status: completed
  - id: adjust-package-data-if-needed
    content: Add minimal package-data rules only if runtime config/data files are missing after install
    status: completed
isProject: false
---

# Fix ReplaySwitchingHMM Flat-Layout Packaging

## Goal
Make `uv add ../ReplaySwitchingHMM --group dev` succeed by replacing fragile setuptools auto-discovery with explicit package discovery for all top-level modules you chose to ship.

## Planned Changes
- Update [`H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/ReplaySwitchingHMM/pyproject.toml`](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/ReplaySwitchingHMM/pyproject.toml) to add:
  - a `[build-system]` section using `setuptools.build_meta`
  - `[tool.setuptools]` settings to enable controlled discovery
  - `[tool.setuptools.packages.find]` with `where=["."]`, `namespaces=true`, and `include=["pbe_io*", "ssm*", "configs*", "submission*"]`
- Preserve existing project metadata (`name`, `version`, `requires-python`, deps, `tool.uv.sources`) with minimal edits.

## Validation
- Re-run from Spike3D repo:
  - `uv add ..\ReplaySwitchingHMM --group dev`
- Confirm build no longer fails with `Multiple top-level packages discovered in a flat-layout`.
- If packaging still excludes expected non-code files (for example YAML under `configs`), add minimal `tool.setuptools.package-data` rules in the same file.

## Notes
- This keeps your current flat layout intact (no `src/` migration).
- Explicit discovery avoids accidental breakage as new top-level folders are added.