"""Helper that converts a list of reference docs into a Python source string
which defines `def load_reference(name)`."""

from __future__ import annotations

import pprint


def make_load_reference_script(reference_docs: list[dict]) -> str:
    """Emit Python source defining `load_reference(name: str) -> str`.

    Each reference_doc must be a dict with keys:
      - name: filename (e.g. "smiles_syntax.md")
      - content: full markdown body of the doc
      - summary: (optional) one-line human summary

    The returned string is valid Python. When `exec`'d by the trajectory
    runtime, it registers a single top-level function `load_reference(name)`
    that returns the matching doc body (with a header prefixing the summary),
    or an error message if the name is not known.
    """
    refs_payload = {
        d["name"]: d.get("content", "") for d in reference_docs if d.get("name")
    }
    summaries = {
        d["name"]: (d.get("summary") or "").strip()
        for d in reference_docs if d.get("name")
    }
    payload_src = pprint.pformat(refs_payload, width=120, sort_dicts=True)
    summaries_src = pprint.pformat(summaries, width=120, sort_dicts=True)
    available = sorted(refs_payload.keys())

    return f'''
def load_reference(name: str) -> str:
    """Load the full markdown body of ONE reference document shipped with this skill. Pass `name` as the exact filename shown in the reference manifest. Only call this when the document is directly relevant to the current instance - loading everything is wasteful."""
    _REFS = {payload_src}
    _SUMMARIES = {summaries_src}
    _AVAILABLE = {available!r}
    _name = (name or "").strip()
    if not _name:
        return "Error: `name` argument is required. Available references: " + ", ".join(_AVAILABLE)
    if _name not in _REFS:
        return f"Reference '{{_name}}' not found. Available: " + ", ".join(_AVAILABLE)
    _summary = _SUMMARIES.get(_name, "")
    _header = f"# Reference: {{_name}}\\n"
    if _summary:
        _header += f"\\n> {{_summary}}\\n"
    return _header + "\\n" + _REFS[_name]
'''.strip()
