"""Describe optimizer execution paths and benchmark support."""
from __future__ import annotations

#: How each optimizer executes benchmark tasks during OPTIMIZATION.
OPTIMIZATION_EXECUTION = {
    "skillopt": (
        "SkillOpt's own native rollout for the benchmark, run unmodified in a "
        "subprocess, pointed at the shared proxy through its generic "
        "`openai_compatible` backend (and, for SpreadsheetBench only, through "
        "`azure_openai`'s `openai_compatible` auth mode, which is the hook that "
        "module provides for exactly this)."
    ),
    "sso": (
        "the shared Direct harness, supplied to SSO through its own `Executor` "
        "extension point (`SharedDirectExecutor`)."
    ),
    "skillgen": (
        "the shared Direct harness, reached through a `benchmark == "
        "'shared_direct'` dispatch branch in `SkillGen/trajectory.py`, the same "
        "pattern SkillGen already uses for tau_bench and its other benchmarks."
    ),
}

#: Caveats that remain true even for a successful run.
NATIVE_CAVEATS: dict = {
    ("skillopt", "spreadsheetbench"): (
        "SkillOpt's SpreadsheetBench codegen path hard-imports `azure_openai` "
        "instead of the generic backend router. The runner sets that module's "
        "own `openai_compatible` auth-mode env vars, which is configuration, not "
        "a code change -- but without them the optimization phase silently makes "
        "zero model calls, so the setting is load-bearing."
    ),
    ("skillgen", "spreadsheetbench"): (
        "SkillGen also ships a native SpreadsheetBench path whose grader skips "
        "the official recalculation and whose loader defaults to the full_912 "
        "variant. Manifest-driven runs tag instances `shared_direct`, so that "
        "path is unreachable and the official cell-value evaluator is used."
    ),
}

#: The one structural asymmetry between optimizers, stated rather than hidden.
OPTIMIZATION_HARNESS_NOTE = (
    "SkillOpt's algorithm is built around its own rollout implementation and is "
    "run unmodified, so its optimization-phase rollouts are SkillOpt's code, not "
    "the shared harness. The shared harness is built on the same primitives "
    "(SkillOpt's own prompt files, tool runtime, code executor and ALFWorld "
    "wiring), the benchmark / solver contract (tool turns, repair turns, env "
    "steps, solver max_tokens / temperature) is handed to SkillOpt's env from the "
    "same main-experiment config the shared harness reads "
    "(runners/skillopt.py::harness_cfg_options), and deployment -- where the "
    "reported metric comes from -- is byte-identical code for all three optimizers."
)
