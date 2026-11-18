---
title: Find interpolated positions in matlab-to-neuropy-exporter repo
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/d4839808-c83a-4284-af6f-2bf25689add5
url: https://phohale.sourcegraph.app/deepsearch/227
created: 2025-10-29T15:54:38Z
updated: 2026-01-15T10:15:29Z
---

# search this repo: github.com/diba-lab/matlab-to-neuropy-exporter and find any cases were position is interpolated

Based on my search through the [github.com/diba-lab/matlab-to-neuropy-exporter](https://github.com/diba-lab/matlab-to-neuropy-exporter) repository, I found **one main case where position is interpolated**:

## Position Interpolation in `binpos.m`

The primary position interpolation occurs in [`Unused/DibaCode/binpos.m`](https://phohale.sourcegraph.app/github.com/diba-lab/matlab-to-neuropy-exporter/-/blob/Unused/DibaCode/binpos.m), which:

1. **Interpolates x and y coordinates** using MATLAB's `interp1` function
2. **Resamples position data** to a regular time grid with specified bin size
3. **Takes input** in the format `[x, y, timestamp]` and outputs binned position data

```matlab
function xytbin=binpos(xyt,gridsize)
% Change the time increment of xyt to gridsize
% interpolate x and y.
% txy = [t;x;y] 

tbin = linspace(xyt(1,3),xyt(end,3),gridsize);
xbin = interp1(xyt(:,3),xyt(:,1),tbin);
ybin = interp1(xyt(:,3),xyt(:,2),tbin);

xytbin = [xbin;ybin;tbin]';
```

## Usage of Position Interpolation

This interpolation function is used in several places:

1. **[`xyt2whl.m`](https://phohale.sourcegraph.app/github.com/diba-lab/matlab-to-neuropy-exporter/-/blob/Unused/DibaCode/xyt2whl.m)** - Creates `.whl` files by binning position data at 60 Hz (or specified frequency)

2. **[`clu1dsaveIII.m`](https://phohale.sourcegraph.app/github.com/diba-lab/matlab-to-neuropy-exporter/-/blob/Unused/DibaCode/clu1dsaveIII.m)** - Bins position data into regular 10ms time bins for spike analysis

The interpolation serves to:
- Convert irregularly sampled position data to regular time intervals
- Ensure consistent temporal resolution for downstream analysis
- Fill gaps in position tracking data using linear interpolation

All these files are located in the `Unused/DibaCode/` directory, suggesting they may be legacy code from an earlier version of the analysis pipeline.

## Sources

- [repo:^github.com/diba-lab/matlab-to-neuropy-exporter$ position interpolat](/search?q=patterntype:nls%20repo:%5Egithub.com%2Fdiba-lab%2Fmatlab-to-neuropy-exporter$%20position%20interpolat)
- [repo:^github.com/diba-lab/matlab-to-neuropy-exporter$ interpolat](/search?q=patterntype:keyword%20repo:%5Egithub.com%2Fdiba-lab%2Fmatlab-to-neuropy-exporter$%20interpolat)
- [github.com/diba-lab/matlab-to-neuropy-exporter/Unused/DibaCode/binpos.m@fa22c32](/github.com/diba-lab/matlab-to-neuropy-exporter@fa22c3290f93fef2ac7403f26c9e5d2a2a93ffc5/-/blob/Unused/DibaCode/binpos.m)
- [repo:^github.com/diba-lab/matlab-to-neuropy-exporter$ interp](/search?q=patterntype:keyword%20repo:%5Egithub.com%2Fdiba-lab%2Fmatlab-to-neuropy-exporter$%20interp)
- [repo:^github.com/diba-lab/matlab-to-neuropy-exporter$ position AND (resample OR sample OR grid)](/search?q=patterntype:keyword%20repo:%5Egithub.com%2Fdiba-lab%2Fmatlab-to-neuropy-exporter$%20position%20AND%20%28resample%20OR%20sample%20OR%20grid%29)
- [repo:^github.com/diba-lab/matlab-to-neuropy-exporter$ bin](/search?q=patterntype:keyword%20repo:%5Egithub.com%2Fdiba-lab%2Fmatlab-to-neuropy-exporter$%20bin)
- [github.com/diba-lab/matlab-to-neuropy-exporter/Unused/DibaCode/xyt2whl.m@fa22c32](/github.com/diba-lab/matlab-to-neuropy-exporter@fa22c3290f93fef2ac7403f26c9e5d2a2a93ffc5/-/blob/Unused/DibaCode/xyt2whl.m)
- [github.com/diba-lab/matlab-to-neuropy-exporter/Unused/DibaCode/clu1dsaveIII.m@fa22c32?L35-50](/github.com/diba-lab/matlab-to-neuropy-exporter@fa22c3290f93fef2ac7403f26c9e5d2a2a93ffc5/-/blob/Unused/DibaCode/clu1dsaveIII.m?L35-50)

## Suggested Follow-ups

- Are there alternative interpolation methods used in the repository besides linear interpolation?
- What are the potential limitations or drawbacks of using linear interpolation for position data in this context?

