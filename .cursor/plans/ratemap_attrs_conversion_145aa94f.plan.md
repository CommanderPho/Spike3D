---
name: Ratemap attrs conversion
overview: Convert `Ratemap` to `@custom_define(slots=False, eq=False)` with field names and pickle/HDF behavior preserved so existing PfND pipelines, constructors, and nested HDF writes keep working.
todos:
  - id: define-fields
    content: Add @custom_define + AttrsBasedClassHelperMixin; declare attrs fields with converters, aliases, custom-HDF tags, shape-aware repr
    status: pending
  - id: post-init-props
    content: Replace __init__ with __attrs_post_init__ validation; re-implement metadata/filename properties on Ratemap over _metadata/_filename
    status: pending
  - id: pickle-migration
    content: Add __setstate__ migrating metadata/_metadata, defaults for missing historical attrs
    status: pending
  - id: preserve-api
    content: Keep neuron_ids properties, computed properties, __getitem__, custom to_hdf, classmethods unchanged in behavior
    status: pending
  - id: smoke-verify
    content: Run placefield/HDF-related NeuroPy tests and mentally verify constructor call sites still match
    status: pending
isProject: false
---

# Convert Ratemap to attrs carefully

## Decisions (locked in)

- Keep `DataWriter` inheritance; model `_filename` / `_metadata` as attrs fields and preserve the public `metadata` / `filename` property behavior.
- Keep stored attribute names `_neuron_ids` / `_neuron_extended_ids` (with `alias=` so init still accepts `neuron_ids=` / `neuron_extended_ids=`).
- Keep the hand-written `to_hdf` override unchanged in layout/semantics.
- Use `@custom_define(slots=False, eq=False)` + add `AttrsBasedClassHelperMixin` to the MRO (same pattern as [`BasePositionDecoder`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction.py)).
- Add `__setstate__` migration for older pickles that may store `metadata` vs `_metadata`, or omit `_filename`.

## Scope

Single file primary change: [`NeuroPy/neuropy/core/ratemap.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\ratemap.py).

Call-site audit completed: no edits needed in placefields / pyPhoPlaceCellAnalysis / Spike3D / pyPhoCoreHelpers if the public constructor, aliases, and writable fields stay identical (see audit section below).

## Implementation

### 1. Class decorator and bases

Replace the plain class definition with:

```python
@custom_define(slots=False, eq=False)
class Ratemap(HDFMixin, AttrsBasedClassHelperMixin, NeuronIdentitiesDisplayerMixin, RatemapPlottingMixin,
              ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin,
              NeuronUnitSlicableObjectProtocol, BinnedPositionsMixin, DataWriter):
```

Imports for `custom_define` / `serialized_field` / `shape_only_repr` / `array_values_preview_repr` are already present (or adjacent) in this file / `AttrsClassHelpers`.

### 2. Attrs fields (match current instance state)

Declare fields in init order matching today’s constructor:

| Field | Notes |
|-------|--------|
| `tuning_curves` | required; converter `np.asarray`; `repr=shape_only_repr`; shape metadata `('n_neurons', '*spatial')` |
| `unsmoothed_tuning_maps` | default `None`; convert only when not None |
| `spikes_maps` | default `None`; convert when not None (today always `np.asarray(spikes_maps)`, including `None` → weird object array — **preserve current behavior** with the same `np.asarray` converter so call sites that pass `None` do not change) |
| `xbin`, `ybin`, `zbin` | default `None`; `repr=array_values_preview_repr` |
| `occupancy` | default `None` |
| `_neuron_ids` | `alias='neuron_ids'`, default `None` |
| `_neuron_extended_ids` | `alias='neuron_extended_ids'`, default `None` |
| `_metadata` | `alias='metadata'`, default `None` — **or** keep as private field and retain DataWriter’s merging `metadata` property |
| `_filename` | default `None`, `init=False` or default None, non-serialized |

**Concrete metadata approach:** keep DataWriter’s `metadata` / `filename` property methods (do not regenerate conflicting attrs properties). Store state on `_metadata` / `_filename` as attrs fields with `init=False` for `_filename`, and accept `metadata=` via a temporary pattern:

- Prefer: attrs field named `_metadata` with `alias='metadata'` **and** remove/override DataWriter’s property if it would clash; replicate DataWriter’s merge-on-set semantics in a custom setter via `field(on_setattr=...)` or a thin `@metadata.setter` after attrs define.
- Simplest compatible path that avoids double-definition: declare `_metadata: Optional[dict] = non_serialized_field(default=None)` and `_filename: Optional[Any] = non_serialized_field(default=None, init=False)`, keep DataWriter’s property methods for `metadata`/`filename`, and accept constructor `metadata=` by marking `_metadata` with `alias='metadata'` while **not** defining a conflicting property on Ratemap — if DataWriter properties remain on the MRO they will shadow attrs accessors. To avoid that, **stop relying on DataWriter property methods for storage** by re-implementing equivalent `metadata`/`filename` properties on `Ratemap` after the fields (same merge logic as [`DataWriter`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\datawriter.py)), so public behavior matches and attrs owns the backing fields.

### 3. Replace `__init__` with `__attrs_post_init__`

Move length assertions only:

```python
def __attrs_post_init__(self):
    if self._neuron_ids is not None:
        assert len(self._neuron_ids) == self.tuning_curves.shape[0]
    if self._neuron_extended_ids is not None:
        assert len(self._neuron_extended_ids) == self.tuning_curves.shape[0]
        if self._neuron_ids is not None:
            assert len(self._neuron_extended_ids) == len(self._neuron_ids)
```

Do **not** call `DataWriter.__init__` (attrs constructs the instance); `_filename` defaults to `None` via the field.

### 4. Keep public properties that are not attrs fields

Leave all computed properties and methods as-is (`n_neurons`, `ndim`, occupancy helpers, normalization, `__getitem__`, `get_by_id`, static helpers, `to_hdf`, `build_merged_ratemap`, `get_debug_binning_info`).

For `neuron_ids` / `neuron_extended_ids`: with attrs `alias=`, public init kwargs work; keep explicit `@property` getters/setters targeting `_neuron_ids` / `_neuron_extended_ids` so `NeuronIdentitiesDisplayerMixin` and code that assigns `ratemap.neuron_ids = ...` (e.g. `__getitem__`) continue to work. If attrs also generates accessors, prefer keeping the explicit properties and naming fields with leading underscore without letting attrs create public name collisions (standard private-field + alias pattern).

### 5. Pickle / `__setstate__` compatibility

Add:

```python
def __setstate__(self, state):
    # migrate pre-attrs / DataWriter variants
    if 'metadata' in state and '_metadata' not in state:
        state['_metadata'] = state.pop('metadata')
    state.setdefault('_filename', None)
    state.setdefault('unsmoothed_tuning_maps', None)
    state.setdefault('zbin', None)
    # neuron ids may be missing on incomplete historical objects
    state.setdefault('_neuron_ids', None)
    state.setdefault('_neuron_extended_ids', None)
    self.__dict__.update(state)
```

Optionally mirror a minimal `__getstate__` that dumps `__dict__` (default is fine with `slots=False`).

### 6. HDF serialization fields metadata

Mark map / bin / occupancy / neuron-id fields with `serialized_field(..., is_hdf_handled_custom=True)` so future automatic HDF paths know `to_hdf` owns them. **Do not** replace the existing `to_hdf` body.

### 7. `__getitem__` compatibility

Leave deepcopy+mutate logic. It already writes `_neuron_extended_ids` and `neuron_ids` — ensure setters still map to `_neuron_ids`. No `evolve` rewrite in this pass (minimize risk).

## Call-site compatibility audit (completed)

**Verdict: no pyPhoPlaceCellAnalysis / Spike3D / pyPhoCoreHelpers call-site edits required.** All live `Ratemap(...)` constructions are in NeuroPy and already use planned field/alias names. Downstream code only reads public attrs or mutates mutable fields, which remains valid with `slots=False` (not frozen) and `eq=False`.

### Live constructions (must keep working)

| Site | Pattern | Compatible? |
|------|---------|-------------|
| [`placefields.py:1127`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\placefields.py) | positional `tuning_curves` + kwargs (`unsmoothed_tuning_maps`, `spikes_maps`, `xbin`/`ybin`/`zbin`, `neuron_ids`, `occupancy`, `neuron_extended_ids`) | Yes |
| [`placefields.py:1985–1988`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\placefields.py) | keyword maps + `**bin_kwargs` (`xbin`/`ybin`/`zbin` only) + `occupancy`/`neuron_ids`/`neuron_extended_ids` | Yes — no unexpected kwargs |
| [`time_dependent_placefields.py:258–261`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\time_dependent_placefields.py) | positional maps + bin/neuron/occupancy kwargs (no `metadata`, no `unsmoothed_tuning_maps`) | Yes |
| [`ratemap.py:412`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\ratemap.py) / [`:540`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\ratemap.py) | includes `metadata=` | Yes — requires Ratemap-owned `metadata` property over `_metadata` |

No constructions in Spike3D / pyPhoPlaceCellAnalysis / pyPhoCoreHelpers. Notebooks only `import Ratemap`. No rogue kwargs (`neuron_IDs=`, `xbins=`, etc.).

### Mutations that must remain writable (attrs fields, not frozen)

| Site | Assignment |
|------|------------|
| [`placefields.py:1241`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\placefields.py) | `self.ratemap.occupancy = value` |
| [`placefields.py:1263`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\placefields.py) | `self.ratemap.neuron_extended_ids = value` |
| [`PendingNotebookCode.py:10884–10885`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) | `.ratemap.ybin = ...` |
| SequenceBasedComputations (~398) | `pf.ratemap = pf.ratemap.get_by_id(...)` (whole-object replace) |

### Confirmed absent (no extra risk)

- External access to `_neuron_ids` / `_metadata` / `_filename`
- `vars(ratemap)` / `asdict(ratemap)` / `ratemap ==` / hashing of Ratemap instances
- Code assuming Ratemap lacks `__attrs_attrs__`

### Soft behavior note (implementation-only)

Today, omitting `neuron_ids` leaves `_neuron_ids` unset (`AttributeError` on access). Attrs `default=None` always defines the field — slightly safer, no call-site impact.

## Downstream verification (smoke after conversion)

- Construct via the four live sites above.
- Mutate `occupancy` / `neuron_extended_ids` / `ybin` as in the mutation table.
- `deepcopy` + index slice `ratemap[i]` / `get_by_id`.
- `PfND.to_hdf` nested `ratemap` write ([`placefields.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\placefields.py) ~L1666).
- Unpickle a known `PfND` / pipeline pickle and access `.ratemap.tuning_curves` / `.neuron_ids` / `.metadata`.
- Existing NeuroPy tests: [`tests/test_placefields.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\tests\test_placefields.py), [`tests/test_dandi_001754_three_dim_spatial.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\tests\test_dandi_001754_three_dim_spatial.py).

## Out of scope

- Rewriting `__getitem__` to `attrs.evolve`
- Switching HDF to fully automatic `HDFMixin` serialization
- Dropping `DataWriter`
- Downstream call-site edits (audit confirmed none needed)