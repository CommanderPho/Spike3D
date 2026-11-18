# Launches the jupyer notebook
# cd H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D
# uv sync --all-extras
# .venv/Scripts/activate.ps1
uv run jupyter-lab --no-browser --port='8889' --ServerApp.ip='0.0.0.0' --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True

