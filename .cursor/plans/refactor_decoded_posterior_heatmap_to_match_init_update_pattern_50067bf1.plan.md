---
name: Refactor decoded posterior heatmap to match init/update pattern
overview: Restructure the decoded posterior heatmap code to only create a new figure on first initialization, and update the existing canvas figure on subsequent calls, matching the pattern used by the trajectory plotters.
todos: []
---

# Refactor Decoded Posterior Heatmap to Mat

ch Init/Update Pattern

## Current Issue

The decoded posterior heatmap code (lines 2447-2524) always creates a new figure and plots on it, even when updating an existing canvas. This is inefficient and doesn't match the pattern used by the trajectory plotters.

## Solution

Restructure the code to:

1. Check `needed_init` first (whether canvas exists)
2. **If init**: Create new figure, plot heatmap, embed in dock widget
3. **If update**: Get existing canvas, clear figure, replot on existing figure

## Changes Required

In [`pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py):

1. **Move the `needed_init` check earlier** (before figure creation)
2. **Restructure the init branch** (lines 2453-2508):

- Only create figure and plot when `needed_init` is True
- Keep the embedding logic as-is

3. **Restructure the update branch** (lines 2509-2524):

- Get existing canvas first
- Clear and replot on existing figure
- The existing else block logic is correct, just needs to be the primary path