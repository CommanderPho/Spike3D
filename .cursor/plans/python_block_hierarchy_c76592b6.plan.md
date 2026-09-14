---
name: Python block hierarchy
overview: Create a personal Cursor skill with a Python helper that parses source into a stable block tree (module → file → imports / classes / functions), checksums each node from a location-stripped AST, and writes JSON suitable for later minimal diffs.
todos:
  - id: write-skill-md
    content: Write SKILL.md (frontmatter, CLI, JSON schema, future-diff identity rules)
    status: completed
  - id: write-parser
    content: "Implement scripts/block_hierarchy.py: AST walk, outline tree, own/tree SHA-256, JSON CLI"
    status: completed
isProject: false
---

# Python block-hierarchy skill

## What you get

A personal skill at [`C:/Users/pho/.cursor/skills/python-block-hierarchy/`](C:/Users/pho/.cursor/skills/python-block-hierarchy/) (not `skills-cursor`) with:

- [`SKILL.md`](C:/Users/pho/.cursor/skills/python-block-hierarchy/SKILL.md) — when to run the tool, CLI, JSON shape, and how later diffs should use identity vs checksums
- [`scripts/block_hierarchy.py`](C:/Users/pho/.cursor/skills/python-block-hierarchy/scripts/block_hierarchy.py) — the actual parser (`ast` only; stdlib)

Version 1 **extracts and checksums only**. No tree-diff yet, but the schema is built so a later comparer can emit add/remove/modify (and later, move) with no format break.

Follow existing personal-skill pattern from [`semantic-git-diff`](C:/Users/pho/.cursor/skills/semantic-git-diff/SKILL.md): agent **runs** the script; docs use forward slashes plus a Windows full-path note.

## Tree shape (v1)

```
Module          # file stem, or directory name if scanning a folder
  File          # one .py file
    Imports     # single group of top-level Import / ImportFrom
      import elements (each statement is a leaf with its own checksum)
    Class       # each top-level class
      field     # class-body Assign / AnnAssign (not methods)
      classmethod / staticmethod
      method    # instance methods (first param is `self`, or `@property` and similar that are not class/static)
    Function    # each top-level FunctionDef / AsyncFunctionDef
```

**In parent `own_checksum`, not listed as child blocks:** nested classes/functions; module-level assignments, `if __name__ == "__main__"`, and other leftover statements. That way outline stays as specified, but those edits still change the parent hash.

**Not separate blocks:** comments and formatting (stripped by AST). **Included in hashes:** docstrings (`ast.Expr` with a string in the usual docstring slot).

## Identity vs checksum (for future diffs)

Each node stores both:

| Field | Role |
| --- | --- |
| `identity` | Stable key: `kind` + relative path + qualname + occurrence index (duplicate names) |
| `own_checksum` | SHA-256 of `ast.dump` on a **copy** of this node’s exclusive AST (child block subtrees removed; lineno/col offsets cleared) |
| `tree_checksum` | SHA-256 of `own_checksum` plus children’s `tree_checksum`s in **identity-sorted** order |

Later comparer (not in v1): match on `identity`; if missing → added/removed; if present and `own_checksum` differs → modified; if `own_checksum` matches but parent identity changed → move (optional later). `tree_checksum` is a cheap “did this subtree change?” filter.

Class `own_checksum` = `ClassDef` with method/nested-class bodies removed (bases, decorators, docstring, and remaining non-field stmts stay). Each field/method has its own node. File `own_checksum` = leftover module body after stripping import/class/function children.

## Implementation sketch

- Parse with `ast.parse` (Python 3.9-safe; no 3.10+ `match` features required).
- `ast.walk` copy: null out `lineno`, `col_offset`, `end_lineno`, `end_col_offset`; dump with `include_attributes=False`.
- Classify methods: `@staticmethod` / `@classmethod` first; else instance method if first arg is `self` (also `@property`); other class-body functions go with instance methods rather than being dropped.
- CLI: path to a `.py` file or directory; `--out` JSON; stdout JSON if no `--out`. Recurse `**/*.py`, skip `__pycache__`.
- Script style: single-line `def` when ≤ 400 chars; two blank lines between methods; `## END for ...` on new loops.

JSON (illustrative):

```json
{
  "kind": "file",
  "name": "time_slicing.py",
  "identity": "file:neuropy/utils/mixins/time_slicing.py",
  "qualname": "time_slicing",
  "span": {"lineno": 1, "end_lineno": 748},
  "own_checksum": "...",
  "tree_checksum": "...",
  "children": []
}
```

## Skill behavior

- **name:** `python-block-hierarchy`
- **description:** third person, WHAT + WHEN (block tree, checksums, structural snapshot of Python, prepare for minimal semantic diffs).
- Omit `disable-model-invocation` so it can attach when the user asks for a Python block hierarchy / structural checksums.
- Instruct the agent to run the script, not re-implement AST walking in chat.
- Document v1 limits: no diff, outline-only children, syntax errors fail that file with a clear error object.

## Out of scope

- Comparing two JSON trees
- Git integration (can wrap this later next to `semantic-git-diff`)
- Non-Python files
- Installing a package into NeuroPy / Spike3D
