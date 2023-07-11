
cd /home/halechr/repos/Spike3D
./scripts/unix/repos_pull_changes.sh
poetry shell
jupyter-lab --no-browser --port=8889 --ip=0.0.0.0