# Clusterless RTC decoder patch for pyPhoPlaceCellAnalysis

Copy these files into the sibling `pyPhoPlaceCellAnalysis` repo (paths mirror `src/` and `tests/`).

Also add to `pyPhoPlaceCellAnalysis/pyproject.toml` dependencies:

```toml
"replay_trajectory_classification",
```

Spike3D already declares `replay_trajectory_classification` in its root `pyproject.toml`.

Pipeline usage:

```python
computation_functions_name_includelist = [..., 'position_decoding_clusterless']
# or '_perform_clusterless_position_decoding_computation'
```

Outputs: `pf1D_ClusterlessDecoder`, `pf2D_ClusterlessDecoder`.
