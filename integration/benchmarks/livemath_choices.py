"""Shuffle LiveMath choices deterministically while preserving the correct answer."""
from __future__ import annotations

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.envs.livemathematicianbench.dataloader import (  # noqa: E402
    LiveMathematicianBenchDataLoader,
)

#: Default shuffle seed for the shared protocol. Matches SkillOpt's own
#: `split_seed`/`seed` default (42) so a run of this infrastructure and a
#: native SkillOpt run at default settings present items identically.
DEFAULT_CHOICE_SHUFFLE_SEED = 42

_SHUFFLER = LiveMathematicianBenchDataLoader(shuffle_choices=True)


def shuffle_item(item: dict, seed: int = DEFAULT_CHOICE_SHUFFLE_SEED) -> dict:
    """Return `item` with its choices deterministically shuffled.

    Delegates to SkillOpt's own implementation. `item` is the raw normalized
    dict shape `load_items` produces; the returned dict has `choices` reordered
    and relabeled A.. and `correct_choice` pointing at the new label.
    """
    return _SHUFFLER._shuffle_item_choices(item, int(seed))


def shuffle_seed_for(manifest) -> int:
    """The shuffle seed a manifest governs, defaulting to the protocol value."""
    value = (manifest.sampling or {}).get("choice_shuffle_seed")
    return int(DEFAULT_CHOICE_SHUFFLE_SEED if value is None else value)
