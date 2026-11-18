---
name: Consolidate Masking Block
overview: Refactor the filtering/recompute block in build_masked_container to remove duplication and verbosity without changing behavior or outputs.
todos:
  - id: consolidate-epoch-span-map
    content: Build epoch span mapping once and reuse inline
    status: completed
  - id: reduce-duplication
    content: Replace repeated list checks/loops with shared code
    status: completed
    dependencies:
      - consolidate-epoch-span-map
  - id: keep-prints
    content: Preserve current logging messages/behavior
    status: completed
    dependencies:
      - reduce-duplication
---

# Consolidate Masking Block

## Scope

- Refactor the verbose block in `build_masked_container` while preserving behavior and outputs, including all prints and edge-case handling.

## Planned Changes

- Introduce small, inline local variables and compact loops to avoid repeated traversal and repeated mapping of epoch times.
- Collapse duplicate logic for building epoch time spans into a single inline block reused by both the filtering and recompute sections.
- Keep the behavior identical: same matching rules, tolerance, overlap detection, and recomputation of `is_future_present_past` for both lists.

## Files

- [`pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py)

## Implementation Todos

- consolidate-epoch-span-map: Build epoch span mapping once and reuse inline.
- reduce-duplication: Replace repeated list checks and loops with compact shared code paths.
- keep-prints: Preserve current logging messages and behavior.