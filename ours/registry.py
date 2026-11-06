"""Register partitioners, operators, selection rules, and issue trackers."""
from __future__ import annotations

from typing import Callable

KINDS = ("partitioner", "operator", "tie_break", "deployment_rule", "tracker")

_REGISTRY: dict[str, dict[str, Callable]] = {kind: {} for kind in KINDS}


class RegistryError(KeyError):
    """An unknown component kind or name was requested."""


def register(kind: str, name: str):
    """Class decorator: make `cls` constructible by `name` for `kind`.

    Registering the same name twice for the same kind is an error rather than
    a silent override -- two experiments would otherwise disagree about what a
    name in their artifact means.
    """
    if kind not in _REGISTRY:
        raise RegistryError(f"unknown component kind {kind!r}; kinds: {KINDS}")

    def _decorate(factory):
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not factory:
            raise RegistryError(f"{kind} {name!r} is already registered to {existing!r}")
        _REGISTRY[kind][name] = factory
        return factory

    return _decorate


def build(kind: str, name: str, **options):
    """Construct the registered component `name` of `kind`."""
    try:
        factory = _REGISTRY[kind][name]
    except KeyError:
        raise RegistryError(
            f"no {kind} registered as {name!r}; available: {names(kind)}"
        ) from None
    return factory(**options)


def names(kind: str) -> tuple[str, ...]:
    if kind not in _REGISTRY:
        raise RegistryError(f"unknown component kind {kind!r}; kinds: {KINDS}")
    return tuple(sorted(_REGISTRY[kind]))


def factory(kind: str, name: str):
    return _REGISTRY[kind][name]


__all__ = ["KINDS", "RegistryError", "build", "factory", "names", "register"]
