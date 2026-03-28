"""
Run a vispy example script with the window title set to the example name.

Argv: ``python _run_vispy_example.py <example_name> <script_path> [extra_args...]`` — extra tokens
are forwarded after ``sys.argv`` is rewritten to the script path (see below).

``vispy_EXAMPLES.py`` starts this wrapper with ``QProcess`` so examples run in a subprocess with
correct interpreter and env. From ``PhoCodeConsoleWidget`` inside that browser, prefer
``browser.run_example()`` / ``run_vispy_example("name")`` (async) over ``%run`` or ``!`` on this file,
which block the Qt UI until the child / script finishes.
"""
import os
import sys
import runpy
from pathlib import Path

if __name__ != "__main__":
    sys.exit(1)

if len(sys.argv) < 3:
    print("Usage: python _run_vispy_example.py <example_name> <script_path>")
    sys.exit(1)

example_name = sys.argv[1]
script_path = str(Path(sys.argv[2]).resolve())
if (not Path(script_path).exists()) or (not Path(script_path).is_file()):
    print(f"Error: example script path is invalid: {script_path}")
    sys.exit(1)

os.environ["VISPY_EXAMPLE_TITLE"] = example_name
print(f"[vispy-example-launch] sys.executable: {sys.executable}")
print(f"[vispy-example-launch] VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
print(f"[vispy-example-launch] CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")
print(f"[vispy-example-launch] PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"[vispy-example-launch] script_path: {script_path}")

# Patch vispy Canvas so the first created canvas uses the example name as window title
import vispy.app
_orig_canvas_init = vispy.app.Canvas.__init__

def _patched_canvas_init(self, *args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["title"] = example_name
    if args:
        args = args[1:]
    return _orig_canvas_init(self, *args, **kwargs)

vispy.app.Canvas.__init__ = _patched_canvas_init  # type: ignore[assignment]

try:
    import vispy.scene.canvas
    _orig_scene_canvas_init = vispy.scene.canvas.SceneCanvas.__init__

    def _patched_scene_canvas_init(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["title"] = example_name
        if args:
            args = args[1:]
        return _orig_scene_canvas_init(self, *args, **kwargs)

    vispy.scene.canvas.SceneCanvas.__init__ = _patched_scene_canvas_init  # type: ignore[assignment]
except Exception:
    pass

sys.argv = [script_path] + sys.argv[3:]
runpy.run_path(script_path, run_name="__main__")
