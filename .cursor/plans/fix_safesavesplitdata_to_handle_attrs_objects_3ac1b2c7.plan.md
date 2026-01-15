---
name: Fix safeSaveSplitData to handle attrs objects
overview: Update `safeSaveSplitData` to handle both dictionary inputs and attrs-based objects by detecting the type and converting attrs objects to dictionaries when needed.
todos:
  - id: "1"
    content: Add attrs import check and conversion logic at the start of safeSaveSplitData to detect and convert attrs objects to dictionaries
    status: completed
  - id: "2"
    content: Update the function to handle both dict and attrs object inputs gracefully
    status: completed
---

# Fix safeSaveSplitData to handle attrs objects

## Problem

The function `safeSaveSplitData` expects `computed_data` to be a dictionary with a `.keys()` method, but it's receiving an attrs-based object (`PredictiveDecodingComputationsContainer`) which doesn't have `.keys()`.

## Solution

Update the function to:

1. Check if `computed_data` is a dictionary - if so, use it directly
2. If it's an attrs object, convert it to a dictionary using `attrs.asdict()` or access its `__dict__`
3. Handle the conversion at the beginning of the function

## Implementation Details

### Detection Strategy

- Use `isinstance(computed_data, dict)` to check if it's already a dict
- Use `attrs.has(computed_data)` to check if it's an attrs object
- If it's an attrs object, convert using `attrs.asdict(computed_data)` or use `computed_data.__dict__`

### Code Changes

1. **Import attrs**: Add `from attrs import asdict, has` to the imports at the top of the function (or use the existing import)
2. **Add conversion logic**: At the start of the function, after resolving `pkl_path`, add logic to convert attrs objects to dicts
3. **Update type hint**: Consider updating the type hint to `Union[Dict[str, Any], Any]` to reflect that it can accept attrs objects

### Files to Modify

- [`pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py`](pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py) - Update `safeSaveSplitData` function to handle attrs objects