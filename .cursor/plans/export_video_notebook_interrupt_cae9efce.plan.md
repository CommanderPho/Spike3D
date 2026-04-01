---
name: export_video notebook interrupt
overview: Make `TimeSynchronizedPositionDecoderPlotter.export_video` stop promptly on notebook/kernel interrupt (KeyboardInterrupt), restore sane SIGINT behavior while Qt is active when possible, optionally support a threading.Event for cooperative cancel, and finalize partial outputs (especially GIF) before returning.
todos:
  - id: sigint-guard
    content: Add SIGINT save/restore in export_video try/finally (main thread, ignore failures)
    status: completed
  - id: stop-event-api
    content: Add stop_event + reraise_keyboard_interrupt params and _export_video_should_abort helper
    status: completed
  - id: loops-six
    content: Abort checks + KeyboardInterrupt handling in all six frame loops
    status: completed
  - id: partial-gif
    content: On interrupt, imageio.mimsave partial frames_list for GIF branches
    status: completed
  - id: messages-return
    content: Conditional success vs interrupted messaging; re-raise KBINT when configured
    status: completed
isProject: false
---

# Reliable interruption for `export_video` from notebooks

## Context

All logic lives in `[TimeSynchronizedPositionDecoderPlotter.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\PyQtPlots\TimeSynchronizedPlotters\TimeSynchronizedPositionDecoderPlotter.py)`: method `export_video` (~642–1016). It already uses `try` / `finally` to `out.release()` for OpenCV writers and restore `debug_print`.

**Why interrupts feel unreliable today**

1. **No interrupt checkpoints** — frame loops run to completion; there is no `KeyboardInterrupt` handling or cooperative cancel.
2. **Qt + SIGINT** — On many Unix setups, Qt installs a SIGINT handler so Ctrl+C does not raise `KeyboardInterrupt` in Python until the handler is restored ([common PyQt pattern](https://stackoverflow.com/questions/42814093/signal-handling-with-pyqt)): briefly set `signal.signal(signal.SIGINT, signal.default_int_handler)` around the export (in a nested `try/finally`), **only on the main thread**, catching `ValueError`/`OSError` if unsupported (Windows / non-main thread).
3. **GIF path** — Frames are accumulated in `frames_list` and `imageio.mimsave` runs only at the end; interrupt means **no file** unless we save partial GIF on stop.
4. **Video path** — Partial file is already plausible once `out.release()` runs; success message after `try` should **not** claim success when export was interrupted.

## Implementation strategy (minimal, localized)

**1. New optional parameters** on `export_video` (single-line signature per project rules):

- `stop_event: Optional[threading.Event] = None` — if set, abort remaining frames when `stop_event.is_set()` (works everywhere, including Windows Jupyter where another cell can `event.set()`).
- Optionally `reraise_keyboard_interrupt: bool = True` — after finalizing writers / partial GIF, re-raise `KeyboardInterrupt` so the notebook still shows a normal interrupt traceback (if `False`, only print and return path; pick default based on your preference; **recommend `True`** for familiar notebook behavior).

**2. Small helper inside `export_video`** (after locals are set up, before the big `try`):

- `_export_video_should_abort() -> bool`: `stop_event.is_set()` if `stop_event` is not `None`.

**3. SIGINT restoration** (start of `try`, paired with existing `finally`):

- Save `prev = signal.signal(signal.SIGINT, signal.default_int_handler)` when `signal` is available and assignment succeeds; in `finally`, restore `signal.signal(signal.SIGINT, prev)` if `prev` was stored.

**4. Frame loops** — there are **six** variants (`use_grab` × gif/video, `use_composite` × gif/video, legacy × gif/video). For each loop that appends frames or calls `out.write`:

- Before processing frame `i > 0`, if `_export_video_should_abort()`: set a module-level or closure flag `interrupted = True` and `break`.
- Wrap the per-iteration body (or entire iteration after the `i == 0` continue) in `try` / `except KeyboardInterrupt`: set `interrupted = True`, `break`.

Initialize `interrupted = False` once at the start of the outer `try`.

**5. Partial GIF on interrupt**

- For each GIF branch, if `interrupted` and `frames_list` is non-empty, call `imageio.mimsave(...)` with collected frames before falling through (same `duration_sec` as full export).

**6. Post-`try` messaging and return**

- Track `interrupted` in a way visible after `try`/`finally` (closure variable set before `finally` runs).
- If `interrupted`: print a short message (e.g. partial export path, frame count if cheap to track); **skip** the current unconditional success print.
- If `reraise_keyboard_interrupt` and interrupted via `KeyboardInterrupt` path, re-raise after `finally` (store `exc_info` or a boolean `interrupted_by_keyboard`).

**7. Limits (document in docstring)**

- Interrupt is only checked **between** frames; a single slow `grab()` / `ImageExporter.export()` cannot be aborted mid-call.

## Files to change

- Single file: `[TimeSynchronizedPositionDecoderPlotter.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\PyQtPlots\TimeSynchronizedPlotters\TimeSynchronizedPositionDecoderPlotter.py)` — extend imports (`signal`, `threading` if not present), extend `export_video` signature, SIGINT guard, loop changes, partial GIF, conditional success print, optional re-raise.

No notebook edits required unless you want an example cell showing `stop_event`.

## Verification

- Run a short `export_video` to a `.avi` and press interrupt mid-loop: should stop soon after current frame, file should exist and be playable up to written frames; kernel shows interrupt if `reraise_keyboard_interrupt=True`.
- Same for `.gif`: partial GIF should exist after interrupt.
- Optional: set `stop_event` from another context and confirm clean stop.

