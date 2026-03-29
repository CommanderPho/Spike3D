---
name: BinningContainer slideby attribute
overview: "Add an optional `slideby: Optional[float] = None` field to `BinningContainer` so overlapping/sliding temporal windows are explicitly tagged (hop/step between window starts). Wire it through `__init__`, pickle backward compatibility, custom `__repr__`, and set it automatically in `from_sliding_windows`."
todos:
  - id: add-field-init
    content: Add `slideby` serialized_field, `__init__` param/assignment, `__setstate__` default
    status: completed
  - id: factory-repr
    content: Set `slideby=float(hop)` in `from_sliding_windows`; extend `__repr__` and class docstring
    status: completed
isProject: false
---

# Add `slideby` to `BinningContainer`

## Goal

Make non-standard **sliding-window** binning obvious at the container level: `slideby is not None` means windows advance by that step (seconds in the decoder path), matching the existing `hop` argument in `[from_sliding_windows](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\binning_helpers.py)`. Typical tiled bins from edges keep `slideby=None`.

## File to change

- `[neuropy/utils/mixins/binning_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\binning_helpers.py)` — `BinningContainer` only (minimal scope).

## Implementation

1. **Declared field** (after `window_stop_edges`, same style as other optional metadata):
  - `slideby: Optional[float] = serialized_field(default=None, repr=True, is_computable=False)`
  - Short class/docstring note: `None` = standard edge-defined bins; non-`None` = sliding windows, value is hop between consecutive window starts (same as `center_info.step` when built via `from_sliding_windows`).
2. `**__init__`**: Add keyword `slideby: Optional[float] = None`, assign `self.slideby = None if slideby is None else float(slideby)` (normalize `numpy` scalars). Keep signature on one line if feasible per workspace rules; split only if line length forces it.
3. `**from_sliding_windows`**: Pass `slideby=float(hop)` into `cls(...)` so every sliding path is self-describing without touching call sites such as `[epochs_spkcount](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\decoders.py)`.
4. `**__setstate__**`: If `'slideby' not in self.__dict__`, set `self.slideby = None` for old pickles/HDF round-trips.
5. **Custom `__repr__`**: Append a line `slideby=...` so logs and debug output show the flag immediately.
6. `**init_from_edges**`: No API change required; default `slideby` stays `None` for contiguous-bin construction.

## Out of scope (unless you want them later)

- Inferring `slideby` when only `window_start_edges` are set but `slideby` was omitted (could be a follow-up).
- Duplicating this type in other repos; grep shows `BinningContainer` is only defined in NeuroPy here.

## Verification

- Quick sanity: instantiate via `from_sliding_windows`, assert `container.slideby == hop`; via `init_from_edges`, assert `slideby is None`.
- Run NeuroPy tests touching decoders if any are active (many assertions in `test_decoders.py` are commented).

