---
title: Find and clean old .pkl files recursively by size
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/820b802d-5fde-452a-a5e7-df3634c93fad
url: https://phohale.sourcegraph.app/deepsearch/380
created: 2026-07-15T03:48:08Z
updated: 2026-07-15T03:50:17Z
---

# find and clean up all .pkl files recursively sorted by size older than a certain date

Here are a few approaches depending on your needs:

**Find and list `.pkl` files older than a date, sorted by size:**

```bash
find . -name "*.pkl" -older-than ... 
# "older than" uses a reference file or mtime days:
find . -name "*.pkl" -mtime +30 -printf "%s\t%p\n" | sort -rn
```

Replace `+30` with the number of days (e.g. `+90` for files older than 90 days).

**Human-readable sizes with dates:**

```bash
find . -name "*.pkl" -mtime +30 -exec ls -lh {} + | sort -k5 -rh
```

**Dry run — preview what would be deleted:**

```bash
find . -name "*.pkl" -mtime +30 -printf "%s\t%TY-%Tm-%Td\t%p\n" | sort -rn
```

**Delete after confirming the list:**

```bash
find . -name "*.pkl" -mtime +30 -delete
```

**Or interactively confirm each deletion:**

```bash
find . -name "*.pkl" -mtime +30 -ok rm {} \;
```

**Specific date cutoff (instead of relative days):**

```bash
# Files older than 2025-01-01
find . -name "*.pkl" ! -newer /tmp/ref_date -printf "%s\t%p\n" | sort -rn
# Create reference file first:
touch -d "2025-01-01" /tmp/ref_date
```

**Summary of key flags:**

| Flag | Meaning |
|---|---|
| `-mtime +N` | Modified more than N days ago |
| `-name "*.pkl"` | Match `.pkl` files |
| `-printf "%s\t%p\n"` | Print size in bytes + path |
| `sort -rn` | Sort numerically, largest first |
| `-delete` | Delete matched files (no confirmation) |
| `-ok rm {} \;` | Delete with per-file confirmation |

Always do a dry run before using `-delete`.


---

# github.com/CommanderPho/Spike3D github.com/CommanderPho/Spike3D I thought there was a function or notebook section that did this already

The implementation lives in `pyphoplacecellanalysis`. The primary notebook using it is [PostBatchCopyResults.ipynb](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/PostBatchCopyResults.ipynb?L1422-L1489). Here's a summary of how it works:

---

**The function:** `OldFileArchiver.remove_backup_files_from_session_data_folders`
**Imported from:** `pyphoplacecellanalysis.SpecificResults.AcrossSessionResults`

**The pattern used in [PostBatchCopyResults.ipynb](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/PostBatchCopyResults.ipynb?L1422-L1489) (the most up-to-date version):**

```python
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import OldFileArchiver
from datetime import datetime

# Always delete these (no date condition):
always_delete_patterns = {
    '.pkltmp': '*.pkltmp',
    '.pkl.bak': '*.pkl.bak',
}

# Conditionally delete only if older than cutoff:
conditional_delete_patterns = {
    # '.pkl': '*.pkl',                          # all pkl (commented out - too broad)
    'pipeline.pkl': '*loadedSessPickle*.pkl',
    'global.pkl': '*global_computation_results*.pkl',
    ## Numpy files:
    'position.npy': '*.position.npy',
}

# Set your cutoff date:
conditional_delete_files_cutoff_date = datetime(2025, 8, 1)

# Dry run first:
deleted_file_list = OldFileArchiver.remove_backup_files_from_session_data_folders(
    good_session_concrete_folders=all_known_session_concrete_folders,
    always_delete_patterns=always_delete_patterns,
    conditional_delete_patterns=conditional_delete_patterns,
    conditional_delete_files_cutoff_date=conditional_delete_files_cutoff_date,
    is_dryrun=True   # <-- flip to False to actually delete
)
deleted_file_list
```

**Key notes:**
- The older notebooks (in [`EXTERNAL/_DEP_OldNotebooks/`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/tree/EXTERNAL/_DEP_OldNotebooks)) used a `cutoff_date=` kwarg; the current `PostBatchCopyResults.ipynb` uses `conditional_delete_files_cutoff_date=` — make sure you use the newer signature.
- Always run with `is_dryrun=True` first to preview what will be deleted.
- `always_delete_patterns` are unconditional (no date check); `conditional_delete_patterns` only apply to files older than the cutoff.

## Sources

- [repo:^github.com/CommanderPho/Spike3D$ .pkl find cleanup delete](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20.pkl%20find%20cleanup%20delete)
- [repo:^github.com/CommanderPho/Spike3D$ pkl files size old delete remove](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20pkl%20files%20size%20old%20delete%20remove)
- [repo:^github.com/CommanderPho/Spike3D$ OldFileArchiver remove_backup_files_from_session_data_folders file:.*.py](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OldFileArchiver%20remove_backup_files_from_session_data_folders%20file:.%2A.py)

