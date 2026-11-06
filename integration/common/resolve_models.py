"""Resolve proxy model aliases to provider model names for run provenance."""
from __future__ import annotations

import urllib.error
import urllib.request
import json as _json


def resolve_model_aliases(proxy_base_url: str, aliases: list[str]) -> dict[str, str]:
    """Return `{alias: physical_model}` for each alias found live on the
    proxy at `proxy_base_url` (e.g. "http://localhost:4000/v1"). An alias
    not found on the proxy, or any failure to reach the proxy at all, maps
    to the literal string "unresolved" for that alias -- never silently
    omitted, and never left as an empty/missing key (a caller checking for
    `!= "unresolved"` must never mistake "didn't check" for "checked and
    it's fine").
    """
    out = {alias: "unresolved" for alias in aliases}
    base = proxy_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    url = f"{base}/v1/model/info"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - local proxy only
            payload = _json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return out

    for entry in payload.get("data", []):
        name = entry.get("model_name")
        if name in out:
            physical = (entry.get("litellm_params") or {}).get("model")
            if physical:
                out[name] = physical
    return out
