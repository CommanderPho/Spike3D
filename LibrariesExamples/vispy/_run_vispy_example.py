"""
Run a vispy example script with the window title set to the example name.
Used by vispy_EXAMPLES.py. Usage: python _run_vispy_example.py <example_name> <script_path>
"""
import os
import sys
import runpy

if __name__ != "__main__":
    sys.exit(1)

if len(sys.argv) < 3:
    sys.exit(1)

example_name = sys.argv[1]
script_path = sys.argv[2]
os.environ["VISPY_EXAMPLE_TITLE"] = example_name

# Patch vispy Canvas so the first created canvas uses the example name as window title
import vispy.app
_orig_canvas_init = vispy.app.Canvas.__init__

def _patched_canvas_init(self, *args, **kwargs):
    kwargs["title"] = example_name
    return _orig_canvas_init(self, *args, **kwargs)

vispy.app.Canvas.__init__ = _patched_canvas_init

try:
    import vispy.scene.canvas
    _orig_scene_canvas_init = vispy.scene.canvas.SceneCanvas.__init__

    def _patched_scene_canvas_init(self, *args, **kwargs):
        kwargs["title"] = example_name
        return _orig_scene_canvas_init(self, *args, **kwargs)

    vispy.scene.canvas.SceneCanvas.__init__ = _patched_scene_canvas_init
except Exception:
    pass

sys.argv = [script_path] + sys.argv[3:]
runpy.run_path(script_path, run_name="__main__")
