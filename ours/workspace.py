"""Allocate isolated directories for runs, rounds, and candidates."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


class CandidateIsolationError(RuntimeError):
    """A directory was about to be reused across two candidates."""


def assert_fresh_dir(path: str, what: str) -> str:
    """Create *path*, or fail loudly if something is already in it.

    A populated directory here means a resumable native cache from a
    *different* candidate is about to be read back as if it belonged to this
    one. That failure mode is silent and free, so it is converted into a loud
    one.
    """
    if os.path.isdir(path) and os.listdir(path):
        raise CandidateIsolationError(
            f"{what} directory is not empty: {path}\n"
            "Reusing it would let a previous candidate's cached rollout "
            "results or analyst patches be replayed for this candidate. "
            "Allocate a fresh directory."
        )
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class CandidateWorkspace:
    """Per-candidate directory allocator for one step/round.

    Layout under ``step_dir``::

        evidence/                            parent rollout (shared, read-only)
        candidates/<id>/patches/             per-candidate reflect cache
        candidates/<id>/artifacts/           per-candidate proposal artifacts
        candidates/<id>/selection_eval/      per-candidate evaluation rollout
    """

    step_dir: str
    _claimed: set = field(default_factory=set, repr=False)

    def evidence_dir(self, name: str = "evidence") -> str:
        path = os.path.join(self.step_dir, name)
        self._claim(path, "evidence")
        return assert_fresh_dir(path, "evidence")

    def candidate_root(self, candidate_id: str) -> str:
        """A fresh directory owned by exactly one candidate.

        This is the optimizer-agnostic contract: adapters carve their own
        subdirectories inside it with :func:`assert_fresh_dir`.
        """
        path = os.path.join(self.step_dir, "candidates", candidate_id)
        self._claim(path, f"candidate {candidate_id} root")
        return assert_fresh_dir(path, f"candidate {candidate_id} root")

    def candidate_dirs(self, candidate_id: str) -> dict[str, str]:
        base = os.path.join(self.step_dir, "candidates", candidate_id)
        out = {
            "base": base,
            "patches": os.path.join(base, "patches"),
            "artifacts": os.path.join(base, "artifacts"),
            "selection_eval": os.path.join(base, "selection_eval"),
        }
        for key in ("patches", "artifacts", "selection_eval"):
            self._claim(out[key], f"candidate {candidate_id} {key}")
            assert_fresh_dir(out[key], f"candidate {candidate_id} {key}")
        return out

    def _claim(self, path: str, what: str) -> None:
        real = os.path.abspath(path)
        if real in self._claimed:
            raise CandidateIsolationError(
                f"{what} directory already claimed in this workspace: {real}"
            )
        self._claimed.add(real)


@dataclass
class RunWorkspace:
    """Root of one OURS search run; hands out one workspace per round."""

    root: str
    _rounds: dict = field(default_factory=dict, repr=False)

    def round_workspace(self, round_id: int) -> CandidateWorkspace:
        if round_id in self._rounds:
            return self._rounds[round_id]
        step_dir = os.path.join(self.root, "rounds", f"round_{round_id:03d}")
        os.makedirs(step_dir, exist_ok=True)
        ws = CandidateWorkspace(step_dir=step_dir)
        self._rounds[round_id] = ws
        return ws
