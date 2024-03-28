"""Generate the code reference pages and navigation.

See https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages
Requires

poetry add mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index mkdocs-matplotlib mkdocs-jupyter

mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index mkdocs-matplotlib mkdocs-jupyter
"""
import os
import sys
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

docs_dir = Path(os.path.dirname(os.path.abspath(__file__))) # /home/halechr/repos/Spike3D/docs
print(f'docs_dir: {docs_dir}')
root_dir = docs_dir.parent # Spike3D root repo dir



for path in sorted(Path("src").rglob("*.py")):
    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
