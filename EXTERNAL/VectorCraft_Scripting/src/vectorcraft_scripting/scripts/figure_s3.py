import sys
from pathlib import Path

# Same as VectorCraft_Scripting.ipynb: allow `python figure_3.py` from spike3d venv without editable install.
_package_src: Path = Path(__file__).resolve().parents[2]
if str(_package_src) not in sys.path:
    sys.path.insert(0, str(_package_src))

from vectorcraft_scripting.scripts import PhoDibaPaperVectorCraftHelper

active_figure_name: str = 'figS3_TimeBinDurationOnDecodingPerform.pdf'

replacement_df = PhoDibaPaperVectorCraftHelper.replace_with_updated_images(active_figure_name)
replacement_df
