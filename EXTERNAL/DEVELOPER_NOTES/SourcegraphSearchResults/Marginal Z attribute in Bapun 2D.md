---
title: Marginal Z attribute in Bapun 2D code
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/ad983681-a139-4849-8a08-52d9475581b5
url: https://phohale.sourcegraph.app/deepsearch/351
created: 2026-02-19T21:51:22Z
updated: 2026-02-19T21:53:13Z
---

# github.com/CommanderPho/Spike3D github.com/CommanderPho/pyPhoPlaceCellAnalysis where is the attribute 'marginal_z' added to a 'DecodedFilterEpochsResult' in my Bapun 2D computation code? Find all the places sorted from most recent to oldest

Based on my search across both repositories, I've found where the attribute `marginal_z` is added to `DecodedFilterEpochsResult` in your Bapun 2D computation code. Here are all the places sorted from **most recent to oldest**:

## Most Recent to Oldest

### 1. **Most Recent: 2025-09-21** - [github.com/CommanderPho/Spike3D](InteractivePipelineLoadFromPickle_Bapun.ipynb)
   - **Commit**: [cbbf8cf](https://phohale.sourcegraph.app/github.com/CommanderPho/Spike3D/-/commit/cbbf8cfd6b69896c467402abbd5a41b45a039eb4) "bapun"
   - **Location**: [InteractivePipelineLoadFromPickle_Bapun.ipynb lines 1253-1254](https://phohale.sourcegraph.app/github.com/CommanderPho/Spike3D/-/blob/InteractivePipelineLoadFromPickle_Bapun.ipynb?L1253-1254)
   ```python
   ## ASSIGN IT to .marginal_z
   all_context_filter_epochs_decoder_result.marginal_z = DynamicContainer(p_x_given_n=marginal_z, most_likely_positions_2D=None)
   ```

### 2. **2025-09-23** - [github.com/CommanderPho/pyPhoPlaceCellAnalysis](src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py)
   - **Commit**: [1b746f1](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/commit/1b746f10314261a0f35dc3ab9d60c9ac70d77710) "factored out _add_context_marginal_to_timeline, _add_context_decoded_epoch_marginals_to_timeline"
   - **Location**: [src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py lines 651-652](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L651-652)
   ```python
   ## ASSIGN IT to .marginal_z
   all_context_filter_epochs_decoder_result.marginal_z = DynamicContainer(p_x_given_n=marginal_z, most_likely_positions_2D=None)
   ```

### 3. **Earlier iterations** - Multiple notebook versions
   - Various [Day4OpenField](https://phohale.sourcegraph.app/github.com/CommanderPho/Spike3D/-/blob/InteractivePipelineLoadFromPickle_Bapun_Day4OpenField.ipynb) and [Day5TwoNovel](https://phohale.sourcegraph.app/github.com/CommanderPho/Spike3D/-/blob/InteractivePipelineLoadFromPickle_Bapun_Day5TwoNovel.ipynb) notebooks with similar assignments

### Key Pattern

The attribute is consistently added after computing the marginal over x and y bins:
```python
marginal_z = np.nansum(p_x_given_n, axis=(0, 1))  # sum over spatial dimensions
marginal_z = marginal_z / np.sum(marginal_z, axis=0, keepdims=True)  # normalize
all_context_filter_epochs_decoder_result.marginal_z = DynamicContainer(p_x_given_n=marginal_z, most_likely_positions_2D=None)
```

This computes a context-specific marginal posterior (marginalizing over the x and y spatial dimensions) from the 4D posterior `p_x_given_n` with shape `(n_x_bins, n_y_bins, n_contexts, n_t_bins)`, resulting in a 2D array of shape `(n_contexts, n_t_bins)`.

## Sources

- [repo:github.com/CommanderPho marginal_z DecodedFilterEpochsResult](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%20marginal_z%20DecodedFilterEpochsResult)
- [repo:github.com/CommanderPho marginal_z Bapun](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%20marginal_z%20Bapun)
- [repo:github.com/CommanderPho DecodedFilterEpochsResult marginal_z assign attribute](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%20DecodedFilterEpochsResult%20marginal_z%20assign%20attribute)
- [type:commit (content:marginal_z OR content:DecodedFilterEpochsResult) (repo:github.com/CommanderPho/Spike3D OR repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis) count:100](/search?q=type:commit%20%28content:marginal_z%20OR%20content:DecodedFilterEpochsResult%29%20%28repo:github.com%2FCommanderPho%2FSpike3D%20OR%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%29%20count:100)
- [type:commit (message:marginal_z OR message:Bapun) (repo:github.com/CommanderPho/Spike3D OR repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis) count:100](/search?q=type:commit%20%28message:marginal_z%20OR%20message:Bapun%29%20%28repo:github.com%2FCommanderPho%2FSpike3D%20OR%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%29%20count:100)
- [type:diff content:\.marginal_z\s*= (repo:github.com/CommanderPho/Spike3D OR repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis) select:commit.diff.added count:20 patternType:regexp](/search?q=type:diff%20content:%5C.marginal_z%5Cs%2A=%20%28repo:github.com%2FCommanderPho%2FSpike3D%20OR%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%29%20select:commit.diff.added%20count:20%20patternType:regexp)
- [type:diff content:marginal_z_list (repo:github.com/CommanderPho/Spike3D OR repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis) select:commit.diff.added count:20](/search?q=type:diff%20content:marginal_z_list%20%28repo:github.com%2FCommanderPho%2FSpike3D%20OR%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%29%20select:commit.diff.added%20count:20)
- [github.com/CommanderPho/Spike3D/InteractivePipelineLoadFromPickle_Bapun.ipynb@a8816ef?L1200-1260](/github.com/CommanderPho/Spike3D@a8816ef63527a29e84f83856af9a73863a1d8766/-/blob/InteractivePipelineLoadFromPickle_Bapun.ipynb?L1200-1260)
- [github.com/CommanderPho/Spike3D (diff cbbf8cfd6b69896c467402abbd5a41b45a039eb4~1...cbbf8cfd6b69896c467402abbd5a41b45a039eb4)](/github.com%2FCommanderPho%2FSpike3D/-/compare/cbbf8cfd6b69896c467402abbd5a41b45a039eb4~1..cbbf8cfd6b69896c467402abbd5a41b45a039eb4)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis (diff 1b746f10314261a0f35dc3ab9d60c9ac70d77710~1...1b746f10314261a0f35dc3ab9d60c9ac70d77710)](/github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis/-/compare/1b746f10314261a0f35dc3ab9d60c9ac70d77710~1..1b746f10314261a0f35dc3ab9d60c9ac70d77710)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L645-655](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L645-655)

## Suggested Follow-ups

- Can you explain the purpose of normalizing the marginal_z after summing over the spatial dimensions?
- How does the shape of p_x_given_n affect the computation and interpretation of marginal_z?

