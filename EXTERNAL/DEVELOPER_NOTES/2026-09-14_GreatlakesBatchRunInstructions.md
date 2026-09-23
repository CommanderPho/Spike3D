# Greatlakes (GL) run instructions:

## Terminal 1: Generate the batch run scripts from `ProcessBatchOutputs_qclus1246789_Only.ipy`:
```bash
export SPIKE3D_REPO_ROOT='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv'
cd "${SPIKE3D_REPO_ROOT}/Spike3D"
deactivate
source "${SPIKE3D_REPO_ROOT}/Spike3D/.venv_modern/bin/activate"
ipython ProcessBatchOutputs_qclus1246789_Only.ipy
```

## Terminal 2: Paste the run scripts to start batch execution:
```bash
export SPIKE3D_REPO_ROOT='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv'
cd "${SPIKE3D_REPO_ROOT}/Spike3D"
deactivate
source "${SPIKE3D_REPO_ROOT}/Spike3D/.venv_modern/bin/activate"

```


## Copy to Swap SSD `/tmpssd/` for speed:
```bash

TARGET_PARENT='/tmpssd/halechr'
SPIKE3D_REPO_ROOT="${TARGET_PARENT}/Spike3D_ExploreEnv"

if [ ! -d "${SPIKE3D_REPO_ROOT}" ]; then
    mkdir -p "${TARGET_PARENT}"
    # cp -R '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv' "${TARGET_PARENT}/"
	rsync -ah --info=progress2 '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/' "${SPIKE3D_REPO_ROOT}"
else
    echo "Using existing ${SPIKE3D_REPO_ROOT}"
fi

### Run:
export SPIKE3D_REPO_ROOT="${TARGET_PARENT}/Spike3D_ExploreEnv"
cd "${SPIKE3D_REPO_ROOT}/Spike3D"
deactivate
source "${SPIKE3D_REPO_ROOT}/Spike3D/.venv_modern/bin/activate"
ipython ProcessBatchOutputs_qclus1246789_Only.ipy
```

#### Issue: 
```

```


### or to `/dev/shm`:
```bash

TARGET_PARENT='/dev/shm/halechr'
SPIKE3D_REPO_ROOT="${TARGET_PARENT}/Spike3D_ExploreEnv"

if [ ! -d "${SPIKE3D_REPO_ROOT}" ]; then
    mkdir -p "${TARGET_PARENT}"
    # cp -R '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv' "${TARGET_PARENT}/"
	rsync -ah --info=progress2 '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/' "${SPIKE3D_REPO_ROOT}"
else
    echo "Using existing ${SPIKE3D_REPO_ROOT}"
fi

### Run:
export SPIKE3D_REPO_ROOT="${TARGET_PARENT}/Spike3D_ExploreEnv"
cd "${SPIKE3D_REPO_ROOT}/Spike3D"
deactivate
source "${SPIKE3D_REPO_ROOT}/Spike3D/.venv_modern/bin/activate"
# ipython ProcessBatchOutputs_qclus12_Only.ipy
ipython ProcessBatchOutputs_qclus1246789_Only.ipy

```


# Temporary Pickle Copying Workarounds

```bash
export SPIKE3D_REPO_ROOT='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv'
cd "${SPIKE3D_REPO_ROOT}/Spike3D"
deactivate
source "${SPIKE3D_REPO_ROOT}/Spike3D/.venv_modern/bin/activate"

## `qclus12`
python scripts/archive_qclus_75ms_pickles.py --qclus qclus12 --dry-run
python scripts/archive_qclus_75ms_pickles.py --qclus qclus12 --execute

## `qclus1246789`:
python scripts/archive_qclus_75ms_pickles.py --qclus qclus1246789 --dry-run
python scripts/archive_qclus_75ms_pickles.py --qclus qclus1246789 --execute

# or: --data-root /nfs/turbo/umms-kdiba/Data

```