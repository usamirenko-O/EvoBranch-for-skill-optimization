"""Configure native SkillOpt resources and fresh one-step proposal schedulers."""
from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path

ensure_on_sys_path(SKILLOPT_ROOT)


ONE_STEP_TOTAL_STEPS = 1


class SkillOptNativeError(RuntimeError):
    """Native SkillOpt cannot be set up for this OURS run."""


# ───────────────────────────── 1. launch inputs ─────────────────────────────


def native_launch_inputs(ctx, *, split_dir: Path, out_root: Path) -> tuple[Path, list, dict]:
    """Return the native training config path, options, and environment overrides."""
    from integration.experiment.runners import skillopt as native_runner
    from integration.experiment.runners.split_materialization import skillopt_env_overrides

    benchmark = ctx.adapter.name
    _, skill_rel = native_runner._BENCHMARK_CONFIG[benchmark]
    native_runner._write_config_overrides()
    skill_init = SKILLOPT_ROOT / skill_rel

    cfg = ctx.config
    env: dict = {}
    for role, alias, max_tokens in (
        ("TARGET", cfg.solver_model, cfg.max_tokens),
        ("OPTIMIZER", cfg.optimizer_model, cfg.optimizer_max_tokens),
    ):
        env[f"{role}_OPENAI_COMPATIBLE_BASE_URL"] = cfg.proxy_base_url
        env[f"{role}_OPENAI_COMPATIBLE_MODEL"] = alias
        env[f"{role}_OPENAI_COMPATIBLE_MAX_TOKENS"] = str(max_tokens)
        env[f"{role}_OPENAI_COMPATIBLE_API_KEY"] = cfg.api_key
    if benchmark == "alfworld":
        env["ALFWORLD_DATA"] = ctx.harness_options.get("alfworld_data") or os.environ.get(
            "ALFWORLD_DATA", ""
        )
    if benchmark == "spreadsheetbench":
        for role in ("TARGET", "OPTIMIZER"):
            env[f"{role}_AZURE_OPENAI_AUTH_MODE"] = "openai_compatible"
            env[f"{role}_AZURE_OPENAI_ENDPOINT"] = cfg.proxy_base_url
            env[f"{role}_AZURE_OPENAI_API_KEY"] = cfg.api_key
        env["AZURE_OPENAI_AUTH_MODE"] = "openai_compatible"
        env["AZURE_OPENAI_ENDPOINT"] = cfg.proxy_base_url
        env["AZURE_OPENAI_API_KEY"] = cfg.api_key

    batch_size = min(int(cfg.extra.get("skillopt_batch_size", len(ctx.train_ids))), len(ctx.train_ids))
    cfg_options = [
        f"env.split_dir={Path(split_dir).resolve()}",
        f"env.skill_init={skill_init.resolve()}",
        "env.split_mode=split_dir",
        f"model.target={cfg.solver_model}",
        f"model.optimizer={cfg.optimizer_model}",
        f"train.train_size={len(ctx.train_ids)}",
        f"train.batch_size={batch_size}",
        f"train.num_epochs={int(cfg.extra.get('skillopt_num_epochs', 1))}",
        "evaluation.eval_test=false",
    ]
    for key, value in skillopt_env_overrides(benchmark).items():
        cfg_options.append(f"{key}={value}")
    cfg_options.extend(native_runner.harness_cfg_options(benchmark, ctx.harness_options))
    cfg_options.extend(cfg.extra.get("skillopt_cfg_options", []))
    return native_runner.config_override_path(benchmark), cfg_options, env


def apply_native_environment(env_updates: dict) -> dict:
    """Apply native environment settings and refresh cached backend configuration."""
    os.environ.update({k: str(v) for k, v in env_updates.items()})
    from skillopt.model import openai_compatible_backend as compat

    for role, config in (("optimizer", compat.OPTIMIZER_CONFIG), ("target", compat.TARGET_CONFIG)):
        fresh = compat._initial_config(role)
        for name in fresh.__dataclass_fields__:
            setattr(config, name, getattr(fresh, name))
    compat._reset_clients()
    applied = {"openai_compatible": {
        "optimizer": {"base_url": compat.OPTIMIZER_CONFIG.base_url,
                      "deployment": compat.OPTIMIZER_CONFIG.deployment,
                      "max_tokens": compat.OPTIMIZER_CONFIG.max_tokens},
        "target": {"base_url": compat.TARGET_CONFIG.base_url,
                   "deployment": compat.TARGET_CONFIG.deployment,
                   "max_tokens": compat.TARGET_CONFIG.max_tokens},
    }}
    if any("AZURE_OPENAI" in k for k in env_updates):
        from skillopt.model import azure_openai

        azure_openai.configure_azure_openai(
            endpoint=env_updates.get("AZURE_OPENAI_ENDPOINT"),
            api_key=env_updates.get("AZURE_OPENAI_API_KEY"),
            auth_mode=env_updates.get("AZURE_OPENAI_AUTH_MODE"),
            optimizer_endpoint=env_updates.get("OPTIMIZER_AZURE_OPENAI_ENDPOINT"),
            optimizer_api_key=env_updates.get("OPTIMIZER_AZURE_OPENAI_API_KEY"),
            optimizer_auth_mode=env_updates.get("OPTIMIZER_AZURE_OPENAI_AUTH_MODE"),
            target_endpoint=env_updates.get("TARGET_AZURE_OPENAI_ENDPOINT"),
            target_api_key=env_updates.get("TARGET_AZURE_OPENAI_API_KEY"),
            target_auth_mode=env_updates.get("TARGET_AZURE_OPENAI_AUTH_MODE"),
        )
        applied["azure_openai"] = {"target_endpoint": azure_openai.TARGET_ENDPOINT,
                                   "target_auth_mode": azure_openai.TARGET_AUTH_MODE}
    return applied


# ─────────────────────────────── 2. config ──────────────────────────────────


def _train_script():
    """SkillOpt's own ``scripts/train.py``, imported as a module (not run)."""
    name = "_skillopt_scripts_train"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SKILLOPT_ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _argv(argv):
    saved = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = saved


def load_native_cfg(config_path: Path, cfg_options: list, out_root: Path) -> dict:
    """The flat trainer config SkillOpt's own CLI would produce for this argv."""
    train = _train_script()
    argv = ["train.py", "--config", str(config_path), "--cfg-options", *cfg_options,
            "--out_root", str(Path(out_root).resolve())]
    with _argv(argv):
        args = train.parse_args()
    return train.load_config(args)


# ─────────────────────────────── 3. runtime ─────────────────────────────────


@dataclass
class NativeSkillOptRuntime:
    """The live native objects one SkillOpt+OURS run uses."""

    cfg: dict
    env_adapter: object
    dataloader: object
    #: Builds a FRESH one-step scheduler; called once per OURS round.
    scheduler_factory: object
    skill_init_path: str
    batch_size: int
    train_size: int
    steps_per_epoch: int
    seed: int
    update_mode: str
    lr_control_mode: str
    merge_batch_size: int
    analyst_workers: int
    rewrite_reasoning_effort: str | None
    rewrite_max_completion_tokens: int
    _epoch_batches: dict = field(default_factory=dict)
    batch_log: list = field(default_factory=list)

    @property
    def out_root(self) -> str:
        return self.cfg["out_root"]

    def one_step_budget(self) -> dict:
        """The budget one native one-step proposal gets (the same every round)."""
        from skillopt.engine.propose import resolve_step_budget

        budget = resolve_step_budget(self.scheduler_factory(), update_mode=self.update_mode,
                                     lr_control_mode=self.lr_control_mode)
        return {"mode": budget.mode, "value": budget.value}

    def batch_for_round(self, round_id: int):
        """Round ``round_id``'s TRAIN evidence BatchSpec (data sampling only).

        Batch ``round_id`` of SkillOpt's own epoch plan (``plan_train_epoch``,
        shuffle seed ``seed + epoch * 1000``; ``accumulation`` is 1, enforced at
        set-up). It selects WHICH train items are rolled out; it advances no
        scheduler and carries no optimizer state.
        """
        epoch = round_id // self.steps_per_epoch + 1
        step_in_epoch = round_id % self.steps_per_epoch
        if epoch not in self._epoch_batches:
            self._epoch_batches[epoch] = self.dataloader.plan_train_epoch(
                epoch=epoch,
                steps_per_epoch=self.steps_per_epoch,
                accumulation=1,
                batch_size=self.batch_size,
                seed=self.seed,
                out_root=self.out_root,
            )
        batch = self._epoch_batches[epoch][step_in_epoch]
        if any(e["round_id"] == round_id for e in self.batch_log):
            return batch
        self.batch_log.append({
            "round_id": round_id, "epoch": epoch,
            "batch_index_in_epoch": step_in_epoch, "batch_seed": batch.seed,
            "batch_size": batch.batch_size,
            "task_ids": [str(i.get("id")) for i in (batch.payload or []) if isinstance(i, dict)],
        })
        return batch


def unsupported_native_settings(cfg: dict) -> list[str]:
    """Native settings the existing SkillOpt adapter cannot honour. Blocking."""
    errors = []
    if int(cfg.get("accumulation", 1) or 1) != 1:
        errors.append(
            f"train.accumulation={cfg.get('accumulation')}: one native step then spans "
            "several batches, but SkillOptAdapter collects one evidence batch per Parent"
        )
    if cfg.get("use_skill_aware_reflection", False):
        errors.append(
            "optimizer.use_skill_aware_reflection=true: native flushes a protected "
            "appendix around each step, which SkillOptAdapter does not do"
        )
    return errors


def cross_round_state(cfg: dict, protocol: dict | None = None) -> dict:
    """Describe per-branch slow-update and meta-skill state carried between rounds."""
    state = _stateless_cross_round_state(cfg)
    if protocol is None:
        return state
    disabled = state["disabled"]
    return {
        "decision": "Use stateless native proposals and per-branch epoch updates after Top-M selection; OURS round t corresponds to native epoch t+1.",
        "enabled_per_branch": {
            "slow_update": {"configured": bool(cfg.get("use_slow_update", False)),
                            "native": disabled["slow_update"]["native"],
                            "under_ours": "run for every NEW survivor of round t>=1 against its anchor "
                                          "parent on the native slow-update sample; writes the "
                                          "protected block; force-accepted (no re-validation)"},
            "meta_skill": {"configured": bool(cfg.get("use_meta_skill", False)),
                           "native": disabled["meta_skill"]["native"],
                           "under_ours": "distilled per NEW survivor from the same comparison; stored "
                                         "in metadata.native_state and fed to the next round's "
                                         "native proposal as meta_skill_context"},
        },
        "disabled": {k: disabled[k] for k in ("step_buffer", "lr_history", "scheduler_position")},
        "selection_gate": state["selection_gate"],
        "protocol": protocol,
    }


def _stateless_cross_round_state(cfg: dict) -> dict:
    return {
        "decision": "SkillOpt is a stateless native one-step proposer inside OURS: "
                    "each (Parent, evidence subset) independently calls the native "
                    "one-step proposal; no accumulated optimizer state crosses rounds; "
                    "a Beam member carries only its skill text",
        "disabled": {
            "step_buffer": {"native": "per-epoch buffer of failure patterns / rejected edits",
                            "under_ours": "Use branch history as step_buffer_context when enabled; otherwise leave the native step buffer empty."},
            "slow_update": {"configured": bool(cfg.get("use_slow_update", False)),
                            "native": "epoch-end slow update",
                            "under_ours": "never run"},
            "meta_skill": {"configured": bool(cfg.get("use_meta_skill", False)),
                           "native": "epoch-end meta-skill loaded into later epochs",
                           "under_ours": "meta_skill_context='' for every proposal"},
            "lr_history": {"native": "run-level lr_history.jsonl",
                           "under_ours": "lr_history_path=None for every proposal"},
            "scheduler_position": {"configured": cfg.get("lr_scheduler"),
                                   "native": "one scheduler ticked once per step",
                                   "under_ours": "a fresh one-step scheduler per round"},
        },
        "selection_gate": {
            "configured": cfg.get("use_gate", True) is not False,
            "under_ours": "never called; OURS selects by global Top-M on the shared validation split",
        },
    }


def configure_native_runtime(cfg: dict) -> NativeSkillOptRuntime:
    """Configure native SkillOpt resources and a factory for fresh one-step schedulers."""
    from skillopt.engine import trainer as T

    unsupported = unsupported_native_settings(cfg)
    if unsupported:
        raise SkillOptNativeError("; ".join(unsupported))

    out_root = cfg["out_root"]
    os.makedirs(out_root, exist_ok=True)
    T.reset_token_tracker()

    backend = cfg.get("model_backend", "azure_openai")
    optimizer_backend, target_backend = T._resolve_role_backends(
        backend, cfg.get("optimizer_backend"), cfg.get("target_backend")
    )
    cfg["optimizer_backend"] = optimizer_backend
    cfg["target_backend"] = target_backend
    T.set_optimizer_backend(optimizer_backend)
    T.set_target_backend(target_backend)
    T.configure_codex_exec_from_config(cfg)

    env_adapter = _train_script().get_adapter(cfg)
    env_adapter.setup(cfg)
    dataloader = env_adapter.get_dataloader()
    if dataloader is None:
        raise SkillOptNativeError(
            f"env {cfg.get('env')!r} has no dataloader; the native seed-pool path "
            "is not wired for OURS"
        )

    T.configure_azure_openai(
        endpoint=(cfg.get("azure_openai_endpoint") or cfg.get("azure_endpoint") or None),
        api_version=(cfg.get("azure_openai_api_version") or cfg.get("azure_api_version") or None),
        api_key=(cfg.get("azure_openai_api_key") or cfg.get("azure_api_key") or None),
        auth_mode=cfg.get("azure_openai_auth_mode") or None,
        ad_scope=cfg.get("azure_openai_ad_scope") or None,
        managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
        optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
        optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
        optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
        optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
        optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
        optimizer_managed_identity_client_id=(
            cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
        ),
        target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
        target_api_version=cfg.get("target_azure_openai_api_version") or None,
        target_api_key=cfg.get("target_azure_openai_api_key") or None,
        target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
        target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
        target_managed_identity_client_id=(
            cfg.get("target_azure_openai_managed_identity_client_id") or None
        ),
    )
    T.set_optimizer_deployment(cfg["optimizer_model"])
    T.set_target_deployment(cfg["target_model"])
    T.configure_claude_code_exec(
        path=cfg.get("claude_code_exec_path", "claude"),
        profile=cfg.get("claude_code_exec_profile", ""),
        use_sdk=cfg.get("claude_code_exec_use_sdk", None),
        effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
        max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
    )
    T.configure_cursor_exec(
        path=cfg.get("cursor_exec_path") or None,
        sandbox=cfg.get("cursor_exec_sandbox") or None,
    )
    T.configure_copilot_exec(
        path=cfg.get("copilot_exec_path") or None,
        home=cfg.get("copilot_exec_home") or None,
        allow_all_tools=cfg.get("copilot_exec_allow_all_tools"),
    )
    T.configure_copilot_chat(
        optimizer_model=cfg.get("copilot_chat_optimizer_model") or None,
        target_model=cfg.get("copilot_chat_target_model") or None,
        timeout=cfg.get("copilot_chat_timeout") or None,
    )
    T.configure_qwen_chat(
        base_url=cfg.get("qwen_chat_base_url") or None,
        api_key=cfg.get("qwen_chat_api_key") or None,
        temperature=cfg.get("qwen_chat_temperature"),
        timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
        max_tokens=cfg.get("qwen_chat_max_tokens"),
        enable_thinking=cfg.get("qwen_chat_enable_thinking"),
        thinking_mode=cfg.get("qwen_chat_thinking_mode"),
        optimizer_base_url=cfg.get("optimizer_qwen_chat_base_url") or None,
        optimizer_api_key=cfg.get("optimizer_qwen_chat_api_key") or None,
        optimizer_temperature=cfg.get("optimizer_qwen_chat_temperature"),
        optimizer_timeout_seconds=cfg.get("optimizer_qwen_chat_timeout_seconds"),
        optimizer_max_tokens=cfg.get("optimizer_qwen_chat_max_tokens"),
        optimizer_enable_thinking=cfg.get("optimizer_qwen_chat_enable_thinking"),
        optimizer_thinking_mode=cfg.get("optimizer_qwen_chat_thinking_mode"),
        target_base_url=cfg.get("target_qwen_chat_base_url") or None,
        target_api_key=cfg.get("target_qwen_chat_api_key") or None,
        target_temperature=cfg.get("target_qwen_chat_temperature"),
        target_timeout_seconds=cfg.get("target_qwen_chat_timeout_seconds"),
        target_max_tokens=cfg.get("target_qwen_chat_max_tokens"),
        target_enable_thinking=cfg.get("target_qwen_chat_enable_thinking"),
        target_thinking_mode=cfg.get("target_qwen_chat_thinking_mode"),
    )
    T.configure_minimax_chat(
        region=cfg.get("minimax_region") or None,
        base_url=cfg.get("minimax_base_url") or None,
        api_key=cfg.get("minimax_api_key") or None,
        temperature=cfg.get("minimax_temperature"),
        max_tokens=cfg.get("minimax_max_tokens"),
        enable_thinking=cfg.get("minimax_enable_thinking"),
    )
    minimax_model_cfg = cfg.get("minimax_model")
    if minimax_model_cfg and cfg.get("target_backend") == "minimax_chat":
        T.set_target_deployment(str(minimax_model_cfg))
    T._configure_trace_to_optimizer_gates(target_backend, cfg)
    T.set_reasoning_effort(cfg.get("reasoning_effort", "") or None)

    if env_adapter.requires_ray():
        import ray

        if not ray.is_initialized():
            ray.init(num_gpus=0)

    batch_size = cfg["batch_size"]
    seed = cfg["seed"]
    merge_bs = cfg["merge_batch_size"]
    update_mode = T.normalize_update_mode(cfg.get("skill_update_mode", "patch"))
    lr_control_mode = T._normalise_lr_control_mode(cfg.get("lr_control_mode", "fixed"))
    if T.is_full_rewrite_minibatch_mode(update_mode):
        lr_control_mode = "none"
    rewrite_reasoning_effort = cfg.get("rewrite_reasoning_effort", "high")
    if rewrite_reasoning_effort == "":
        rewrite_reasoning_effort = None
    rewrite_max_completion_tokens = int(cfg.get("rewrite_max_completion_tokens", 64000))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if merge_bs < 2:
        raise ValueError(f"merge_batch_size must be >= 2, got {merge_bs}")

    train_size = T._resolve_train_size(cfg, dataloader)
    steps_per_epoch = math.ceil(train_size / batch_size)
    cfg["train_size"] = train_size
    cfg["steps_per_epoch"] = steps_per_epoch
    cfg["skill_update_mode"] = update_mode
    cfg["lr_control_mode"] = lr_control_mode

    def scheduler_factory():
        return T.build_scheduler(
            mode=cfg.get("lr_scheduler", "constant"),
            max_lr=cfg["edit_budget"],
            min_lr=cfg.get("min_edit_budget", 2),
            total_steps=ONE_STEP_TOTAL_STEPS,
        )

    scheduler_factory()  # fail at set-up, not in round 0, on a bad scheduler config
    T.configure_skill_aware_reflection(
        cfg.get("use_skill_aware_reflection", False),
        cfg.get("skill_aware_appendix_source", "both"),
    )
    return NativeSkillOptRuntime(
        cfg=cfg,
        env_adapter=env_adapter,
        dataloader=dataloader,
        scheduler_factory=scheduler_factory,
        skill_init_path=os.path.abspath(cfg["skill_init"]),
        batch_size=int(batch_size),
        train_size=int(train_size),
        steps_per_epoch=int(steps_per_epoch),
        seed=int(seed),
        update_mode=update_mode,
        lr_control_mode=lr_control_mode,
        merge_batch_size=int(merge_bs),
        analyst_workers=int(cfg["analyst_workers"]),
        rewrite_reasoning_effort=rewrite_reasoning_effort,
        rewrite_max_completion_tokens=rewrite_max_completion_tokens,
    )


def redacted_cfg(cfg: dict) -> dict:
    """The config with credentials masked, by the trainer's own ``_redact_cfg``
    (what native writes to ``config.json``)."""
    from skillopt.engine.trainer import _redact_cfg

    return _redact_cfg(cfg)


def tracker_summary() -> dict:
    """SkillOpt's own cumulative token tracker (every backend, every stage)."""
    from skillopt.model import get_token_summary

    return get_token_summary()


def inflight_rollouts_at_boundary(benchmark_env: str) -> int | None:
    """Native count of rollout threads still running at a phase boundary
    (audit hazard 5), where the env's rollout module exposes it."""
    try:
        module = importlib.import_module(f"skillopt.envs.{benchmark_env}.rollout")
    except ImportError:
        return None
    fn = getattr(module, "get_and_reset_phase_boundary_inflight_count", None)
    return int(fn()) if fn else None


__all__ = [
    "NativeSkillOptRuntime", "apply_native_environment", "SkillOptNativeError", "configure_native_runtime",
    "inflight_rollouts_at_boundary", "load_native_cfg", "native_launch_inputs",
    "ONE_STEP_TOTAL_STEPS", "cross_round_state", "redacted_cfg", "tracker_summary", "unsupported_native_settings",
]
