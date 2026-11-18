"""Intrinsic size of VectorCraft-placeable files, in document points."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


_SVG_LENGTH_TO_PT: Dict[str, float] = {'': 1.0, 'px': 1.0, 'pt': 1.0, 'in': 72.0, 'cm': 72.0 / 2.54, 'mm': 72.0 / 25.4, 'pc': 12.0}


def intrinsic_size_pts(path: Union[str, Path]) -> Optional[Tuple[float, float]]:
    """Intrinsic (width, height) in document points of a replacement file, or None if unknown (e.g. PDF)."""
    def _subfn_svg_length_to_pt(value: str) -> Optional[float]:
        m = re.fullmatch(r'\s*([0-9.]+(?:e[-+]?\d+)?)\s*([a-z%]*)\s*', value, flags=re.IGNORECASE)
        if m is None or m.group(2).lower() == '%':
            return None
        return float(m.group(1)) * _SVG_LENGTH_TO_PT[m.group(2).lower()]

    file_path: Path = Path(path)
    suffix: str = file_path.suffix.lower()
    if suffix == '.svg':
        head: str = file_path.read_text(encoding='utf-8', errors='replace')[:4096]
        svg_tag = re.search(r'<svg\b[^>]*>', head, flags=re.IGNORECASE | re.DOTALL)
        assert svg_tag is not None, f"no <svg> tag found in '{file_path.as_posix()}'"
        tag: str = svg_tag.group(0)
        w_attr = re.search(r'\bwidth="([^"]+)"', tag)
        h_attr = re.search(r'\bheight="([^"]+)"', tag)
        if (w_attr is not None) and (h_attr is not None):
            w_pt, h_pt = _subfn_svg_length_to_pt(w_attr.group(1)), _subfn_svg_length_to_pt(h_attr.group(1))
            if (w_pt is not None) and (h_pt is not None):
                return (w_pt, h_pt)
        vb = re.search(r'\bviewBox="([^"]+)"', tag)
        if vb is not None:
            _vx, _vy, vw, vh = [float(v) for v in re.split(r'[\s,]+', vb.group(1).strip())]
            return (vw, vh)
        return None
    if suffix == '.vectorcraft':
        doc = json.loads(file_path.read_text(encoding='utf-8'))
        rect = doc['document']['artboards'][0]['rect']
        return (float(rect['x1']) - float(rect['x0']), float(rect['y1']) - float(rect['y0']))
    return None
