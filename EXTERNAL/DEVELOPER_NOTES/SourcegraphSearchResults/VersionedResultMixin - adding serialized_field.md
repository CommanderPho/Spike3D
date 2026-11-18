---
title: VersionedResultMixin - adding serialized_field
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/60d27336-e9f9-46bf-b6f4-f1aeaa189d9f
url: https://phohale.sourcegraph.app/deepsearch/300
created: 2025-12-12T11:30:23Z
updated: 2026-07-22T18:15:13Z
---

# github.com/CommanderPho/pyPhoCoreHelpers github.com/diba-lab/NeuroPy github.com/CommanderPho/pyPhoPlaceCellAnalysis I forget how to add a new serialized_field to a `VersionedResultMixin` subclass properly with the date modified info

Based on the code I've found in your repositories, here's how to properly add a new `serialized_field` to a `VersionedResultMixin` subclass with the date modified info:

```python
# When adding a new field to a VersionedResultMixin subclass:

# 1. Add the field with metadata={'field_added': 'YYYY.MM.DD_V'}
AnyC_aclus: NDArray = serialized_field(
    init=False, 
    hdf_metadata={'track_eXclusive_cells': 'AnyC'}, 
    metadata={'field_added': "2025.07.23_0"}
)

# 2. Update the class version string
_VersionedResultMixin_version: str = "2025.07.23_0"
```

**Key points:**

1. **Add `metadata={'field_added': 'YYYY.MM.DD_V'}`** to the field using the date format matching the version string (e.g., `"2025.07.23_0"`)

2. **Update the class `_VersionedResultMixin_version`** to the same or later version

3. **Handle deserialization** in `__setstate__` to set defaults for old objects:
   ```python
   def __setstate__(self, state):
       result_version: str = state.get('result_version', None)
       if result_version is None:
           result_version = "2024.01.10_0"
           state['result_version'] = result_version
       
       # Add defaults for new fields based on version
       if self.is_result_version_earlier_than("2025.07.23_0"):
           state['AnyC_aclus'] = np.array([])  # or appropriate default
   ```

**Examples from your codebase:**
- [LongShortTrackComputations.py?L3116](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py?L3116)
- [EpochComputationFunctions.py?L1293-L1294](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py?L1293-L1294)
- [context_dependent.py?L183](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py?L183)

The version format is documented in [ComputationResults.py?L167-L177](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Model/ComputationResults.py?L167-L177).

## Sources

- [repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis VersionedResultMixin](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20VersionedResultMixin)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/ComputationResults.py@26f673d?L161-250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Model/ComputationResults.py?L161-250)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/diba-lab/NeuroPy serialized_field date modified VersionedResultMixin](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20serialized_field%20date%20modified%20VersionedResultMixin)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers serialized_field metadata field_added](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20serialized_field%20metadata%20field_added)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers serialized_attribute_field](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20serialized_attribute_field)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis serialized_field metadata field_added example](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20serialized_field%20metadata%20field_added%20example)
- [repo:github.com/CommanderPho/pyPhoCoreHelpers serialized_field metadata](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20serialized_field%20metadata)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis serialized_field metadata field_added 2025](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20serialized_field%20metadata%20field_added%202025)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py@26f673d?L3113-3120](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py?L3113-3120)

## Suggested Follow-ups

- Can you provide a complete example of adding a new serialized_field, including the field definition, `__setstate__` implementation, and version update?
- How does the `is_result_version_earlier_than` function work, and what are some best practices for its implementation?

