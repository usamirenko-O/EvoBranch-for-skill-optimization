"""Read shared hyperparameters and resolve explicit run overrides."""
from __future__ import annotations

import argparse
import json

from integration.common.paths import WORKSPACE_ROOT

HPARAMS_PATH = WORKSPACE_ROOT / "main_experiment" / "baseline_hparams.json"
HPARAMS_FILE = "main_experiment/baseline_hparams.json"

#: SSO knobs the SSO runner resolves itself (not passed as --extra by run_cell).
SSO_KNOB_KEYS = ("sso_history_per_step", "sso_num_probes", "sso_update_budget")
SSO_SCHEDULE_BLOCK = "training_schedule"

#: Native knobs that parameterise the optimizer's OWN outer loop. OURS replaces
#: that loop (its depth is ``--rounds``), so a +OURS run must not inherit them;
#: the OURS drivers refuse them outright.
NATIVE_LOOP_ONLY_KEYS = {
    "skillopt": ("skillopt_num_epochs", "skillopt_batch_size"),
    "sso": ("sso_steps",),
    "skillgen": (),
}


def source(path: str) -> str:
    return f"{HPARAMS_FILE}#{path}"


def load_hparams() -> dict:
    return json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))


def _public(block: dict) -> dict:
    return {k: v for k, v in (block or {}).items() if not k.startswith("_")}


# ───────────────────────────── per-cell settings ────────────────────────────


def solver_call(solver: str | None = None) -> dict:
    """``{max_tokens, optimizer_max_tokens, temperature}`` -> ``(value, source)``.

    ``solver`` is the proxy alias; one listed in ``solver_call.max_tokens_by_solver``
    gets that ``max_tokens`` instead of the shared one."""
    block = load_hparams()["solver_call"]
    call = {key: (block[key], source(f"solver_call.{key}"))
            for key in ("max_tokens", "optimizer_max_tokens", "temperature")}
    by_solver = _public(block.get("max_tokens_by_solver") or {})
    if solver in by_solver:
        call["max_tokens"] = (by_solver[solver],
                              source(f"solver_call.max_tokens_by_solver.{solver}"))
    return call


def harness_options(benchmark: str) -> dict:
    """``option -> (value, source)`` for ``benchmark`` (may be empty)."""
    block = _public(load_hparams()["harness"].get(benchmark) or {})
    return {key: (value, source(f"harness.{benchmark}.{key}")) for key, value in block.items()}


def deployment_workers(benchmark: str) -> tuple:
    """``(value, source)``; a benchmark the file does not list gets 1, as
    run_cell.sh always did."""
    block = load_hparams()["deployment_workers"]
    if benchmark in block:
        return int(block[benchmark]), source(f"deployment_workers.{benchmark}")
    return 1, f"{source('deployment_workers')} (benchmark not listed -> 1)"


def optimizer_extras(optimizer: str) -> dict:
    """Flat optimizer knobs (``--extra`` material) for ``optimizer``."""
    block = _public(load_hparams()["optimizer"].get(optimizer) or {})
    block.pop(SSO_SCHEDULE_BLOCK, None)
    return block


def ours_optimizer_extras(optimizer: str) -> dict:
    """``key -> (value, source)``: the native knobs a +OURS run of ``optimizer``
    inherits from its native cell.

    Everything in ``optimizer.<optimizer>`` except the native outer-loop
    parameters (:data:`NATIVE_LOOP_ONLY_KEYS`). SSO's |H_t| / K / U are
    excluded here only because the SSO runner resolves them itself, through
    :func:`sso_knobs`, for native and +OURS alike."""
    block = optimizer_extras(optimizer)
    skip = set(NATIVE_LOOP_ONLY_KEYS.get(optimizer, ()))
    if optimizer == "sso":
        skip |= set(SSO_KNOB_KEYS)
    return {key: (value, source(f"optimizer.{optimizer}.{key}"))
            for key, value in block.items() if key not in skip}


def ours_native_epoch_protocol(optimizer: str):
    """Return whether the optimizer's per-branch epoch protocol is enabled and its source."""
    block = _public((load_hparams().get("ours") or {}).get("native_epoch_protocol") or {})
    if optimizer not in block:
        return None
    return bool(block[optimizer]), source(f"ours.native_epoch_protocol.{optimizer}")


def ours_embedding_model() -> tuple:
    """``(alias, source)`` of the embedding endpoint OURS framework components
    (partitioner / operator) may call. A resource alias, not a design choice."""
    block = load_hparams().get("ours") or {}
    if "embedding_model" not in block:
        raise ValueError(f"{source('ours')} has no embedding_model")
    return str(block["embedding_model"]), source("ours.embedding_model")


OURS_DESIGN_KEYS = ("rounds", "beam_m", "k_subsets")


def ours_design_defaults() -> dict:
    """``{field -> (value, source)}``: the frozen main-experiment OURS design,
    shared by every optimizer. T / M / K (:data:`OURS_DESIGN_KEYS`) are
    required; any other `OursDesign` field present in the block (a decided
    component and its options, e.g. ``partitioner`` /
    ``partitioner_options``) is returned as well. A field absent here is a
    slot the author has not frozen and stays whatever `OursDesign` says
    (a placeholder). Missing T / M / K are an error, never a silent default."""
    block = _public((load_hparams().get("ours") or {}).get("design") or {})
    missing = [k for k in OURS_DESIGN_KEYS if k not in block]
    if missing:
        raise ValueError(f"{source('ours.design')} is missing {missing}")
    out = {key: (int(block[key]), source(f"ours.design.{key}")) for key in OURS_DESIGN_KEYS}
    for key, value in block.items():
        if key not in OURS_DESIGN_KEYS:
            out[key] = (dict(value) if isinstance(value, dict) else value, source(f"ours.design.{key}"))
    return out


# ─────────────────────────────────── SSO ────────────────────────────────────


def sso_training_schedule(benchmark: str) -> dict:
    """``{sso_steps, sso_train_per_step, sso_schedule_seed}`` for ``benchmark``.
    A benchmark the file does not list is an error, never a silent fallback."""
    block = load_hparams()["optimizer"]["sso"][SSO_SCHEDULE_BLOCK]
    per_benchmark = block["sso_train_per_step"]
    if benchmark not in per_benchmark:
        raise ValueError(
            f"no main-experiment sso_train_per_step for benchmark {benchmark!r} in "
            f"{source('optimizer.sso.training_schedule')}; listed: {sorted(per_benchmark)}"
        )
    return {
        "sso_steps": int(block["sso_steps"]),
        "sso_train_per_step": int(per_benchmark[benchmark]),
        "sso_schedule_seed": int(block["sso_schedule_seed"]),
    }


def sso_knobs() -> dict:
    """``{sso_history_per_step, sso_num_probes, sso_update_budget}`` -> ``(value, source)``."""
    block = load_hparams()["optimizer"]["sso"]
    missing = [k for k in SSO_KNOB_KEYS if k not in block]
    if missing:
        raise ValueError(f"{source('optimizer.sso')} is missing {missing}")
    return {key: (int(block[key]), source(f"optimizer.sso.{key}")) for key in SSO_KNOB_KEYS}


# ─────────────────────────────── native launch ──────────────────────────────


def launch_args(optimizer: str, benchmark: str, solver: str | None = None) -> list[str]:
    """The flags `run_cell.sh` hands `integration.run_baseline` for a native
    cell: optimizer ``--extra`` knobs, harness options, the solver call and the
    deployment workers.

    SSO's training schedule and its three knobs are NOT passed: the SSO runner
    resolves them from this same file itself, so native SSO and SSO+OURS read
    them through one path and record the file as their source.
    """
    args: list[str] = []
    skip = set(SSO_KNOB_KEYS) if optimizer == "sso" else set()
    for key, value in optimizer_extras(optimizer).items():
        if key in skip:
            continue
        args += ["--extra", f"{key}={json.dumps(value) if not isinstance(value, str) else value}"]
    for key, (value, _) in harness_options(benchmark).items():
        args += ["--harness-option", f"{key}={value}"]
    call = solver_call(solver)
    args += ["--max-tokens", str(call["max_tokens"][0])]
    args += ["--optimizer-max-tokens", str(call["optimizer_max_tokens"][0])]
    args += ["--temperature", str(call["temperature"][0])]
    args += ["--deployment-workers", str(deployment_workers(benchmark)[0])]
    return args


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description="print run_cell.sh launch flags, one per line")
    p.add_argument("command", choices=("launch-args",))
    p.add_argument("optimizer")
    p.add_argument("benchmark")
    p.add_argument("solver", nargs="?", default=None)
    args = p.parse_args(argv)
    for token in launch_args(args.optimizer, args.benchmark, args.solver):
        print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HPARAMS_FILE", "HPARAMS_PATH", "NATIVE_LOOP_ONLY_KEYS", "SSO_KNOB_KEYS",
    "OURS_DESIGN_KEYS", "deployment_workers", "harness_options", "launch_args", "load_hparams",
    "optimizer_extras", "ours_design_defaults", "ours_embedding_model", "ours_optimizer_extras",
    "solver_call", "source", "sso_knobs", "sso_training_schedule",
]
