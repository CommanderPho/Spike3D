
cd /home/halechr/repos/Spike3D
./scripts/unix/repos_pull_changes.sh
poetry shell
jupyter-lab --no-browser --port=8889 --NotebookApp.ip='0.0.0.0' --NotebookApp.allow_origin='*'
