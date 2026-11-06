"""Resolve the benchmark, optimizer, and search configuration for one OURS run."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from integration.experiment.config import (
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_OUT_ROOT,
    ExperimentConfig,
)
from ours import registry
from ours.design import HYPERPARAMETERS, SLOTS, OursDesign

#: Optimizers with an OURS run driver (see `ours/run_ours.py::_DRIVERS`).
OURS_OPTIMIZERS = ("skillgen", "skillopt", "sso")

METHODS = ("ours",)

#: Registered component names (kept for callers that list choices).
PARTITIONERS = registry.names("partitioner")
OPERATORS = registry.names("operator")
TIE_BREAKS = registry.names("tie_break")
DEPLOYMENT_RULES = registry.names("deployment_rule")

#: Design fields that `build(**kwargs)` accepts directly, for convenience.
DESIGN_KWARGS = HYPERPARAMETERS + tuple(SLOTS) + tuple(f"{s}_options" for s in SLOTS) + (
    "strict_ties",
)

DESIGN_DEFAULT_SOURCE = "OursDesign default"


class OursConfigError(ValueError):
    """The run configuration is not one OURS can execute."""


@dataclass
class OursRunConfig:
    """An OURS run for one optimizer."""

    base: ExperimentConfig
    method: str = "ours"
    #: THE OURS experimental design -- identical meaning for every optimizer.
    design: OursDesign = field(default_factory=OursDesign)
    # ── SkillGen-only native knobs (ignored, and recorded as n/a, for SSO) ──
    #: Which train tasks the partitioner may cluster over (see
    #: `ours.adapters.skillgen.EVIDENCE_SCOPES`).
    evidence_scope: str = "diagnostic_cases"
    #: Re-synthesise each subset's revision guidance from that subset's own
    #: case analyses, instead of inheriting the parent-level guidance.
    resynthesise_subset_guidance: bool = True
    # ── execution knobs ──
    seed: int = 42
    validation_workers: int = 1
    #: SSO only: where ``design.rounds`` (= SSO's T) came from -- an explicit
    #: rounds, or the main-experiment SSO training schedule. None otherwise.
    rounds_source: str | None = None
    #: design field -> where its value came from (default / main-experiment
    #: config / explicit override). Filled by `build`.
    design_sources: dict = field(default_factory=dict)
    #: The resolved formal main-experiment configuration of this cell (the
    #: benchmark / solver contract, the inherited native knobs and, for SSO,
    #: its schedule), each entry ``{"value", "source"}``. Filled by `build`.
    main_experiment_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise OursConfigError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.base.optimizer not in OURS_OPTIMIZERS:
            raise OursConfigError(
                f"no OURS run driver for optimizer {self.base.optimizer!r}; "
                f"available: {OURS_OPTIMIZERS}"
            )
        if not isinstance(self.design, OursDesign):
            raise OursConfigError("design must be an OursDesign")

    # ---- design passthrough (read-only conveniences) ----

    @property
    def rounds(self) -> int:
        return self.design.rounds

    @property
    def beam_m(self) -> int:
        return self.design.beam_m

    @property
    def k_subsets(self) -> int:
        return self.design.k_subsets

    @property
    def partitioner(self) -> str:
        return self.design.partitioner

    @property
    def operator(self) -> str:
        return self.design.operator

    @property
    def strict_ties(self) -> bool:
        return self.design.strict_ties

    # ---- passthrough to the frozen native config ----

    @property
    def optimizer(self) -> str:
        return self.base.optimizer

    @property
    def benchmark(self) -> str:
        return self.base.benchmark

    @property
    def run_id(self) -> str:
        return self.base.run_id

    def adapter(self):
        return self.base.adapter()

    def solver_config(self):
        return self.base.solver_config()

    def run_dir(self) -> Path:
        return self.base.run_dir()

    def artifact_path(self) -> Path:
        """Deliberately NOT `experiment_record.json`.

        A native-baseline record and an OURS record must never be collected by
        the same glob: an OURS run carries the native optimizer's name because
        that optimizer's mechanism produced its proposals, and a matrix script
        that could not tell the two apart would report an OURS number as a
        baseline number.
        """
        return self.run_dir() / "ours_experiment_record.json"

    # ---- scientific status of the design, for this optimizer ----

    def frozen_main_experiment_defaults(self) -> dict:
        """Return shared search defaults, checking that SSO steps match OURS rounds."""
        return {name: value for name, (value, _) in
                _frozen_design_defaults(self.optimizer, self.benchmark).items()}

    def hyperparameter_status(self) -> dict:
        """Per T / M / K: value used, scientific status, frozen default,
        whether it was explicitly overridden, and source."""
        frozen = self.frozen_main_experiment_defaults()
        out = {}
        for name in HYPERPARAMETERS:
            value = getattr(self.design, name)
            where = self.design_sources.get(name, DESIGN_DEFAULT_SOURCE)
            entry = {
                "value": value,
                "status": "frozen_main_experiment_default" if name in frozen else "placeholder",
                "is_placeholder": name not in frozen,
                "frozen_main_experiment_default": frozen.get(name),
                "overridden": where.startswith("override"),
                "source": where,
            }
            if name in frozen:
                entry["matches_frozen_default"] = value == frozen[name]
            out[name] = entry
        return out

    def design_report(self) -> dict:
        return self.design.component_report(hyperparameter_status=self.hyperparameter_status(),
                                            sources=self.design_sources)

    def placeholders_in_force(self) -> list[str]:
        return self.design.placeholders_in_force(hyperparameter_status=self.hyperparameter_status())

    def search_settings(self) -> dict:
        return {
            "method": self.method,
            **self.design.to_dict(),
            "design_components": self.design_report(),
            "placeholders_in_force": self.placeholders_in_force(),
            "evidence_scope": self.evidence_scope,
            "resynthesise_subset_guidance": bool(self.resynthesise_subset_guidance),
            "seed": int(self.seed),
            **({"rounds_source": self.rounds_source} if self.rounds_source else {}),
            **({"main_experiment_config": self.main_experiment_config}
               if self.main_experiment_config else {}),
        }

    def to_dict(self) -> dict:
        return {"base": self.base.to_dict(), **self.search_settings(),
                "validation_workers": int(self.validation_workers)}

    # ---- construction ----

    @classmethod
    def build(
        cls,
        *,
        optimizer: str,
        benchmark: str,
        solver_model: str,
        optimizer_model: str = DEFAULT_OPTIMIZER_MODEL,
        harness: str = "direct",
        split_manifest_path: str | None = None,
        out_root: Path | str = DEFAULT_OUT_ROOT,
        run_id: str = "",
        harness_options: dict | None = None,
        extra: dict | None = None,
        deployment_workers: int | None = None,
        method: str = "ours",
        design: OursDesign | None = None,
        **ours_kwargs,
    ) -> "OursRunConfig":
        """Build from the same flags the native launcher takes, plus the OURS ones. Design fields (``rounds``, ``beam_m``, ``partitioner``, ...) may be passed directly; they override ``design``."""
        base_kwargs = dict(
            optimizer=optimizer,
            benchmark=benchmark,
            solver_model=solver_model,
            optimizer_model=optimizer_model,
            harness=harness,
            split_manifest_path=split_manifest_path,
            out_root=out_root,
            run_id=run_id or _default_run_id(optimizer, benchmark, method, harness, solver_model),
            harness_options=dict(harness_options or {}),
            extra=dict(extra or {}),
            deployment_workers=1 if deployment_workers is None else deployment_workers,
        )
        explicit_call = set()
        for key in ("proxy_base_url", "api_key", "temperature", "max_tokens",
                    "optimizer_max_tokens", "timeout_seconds", "max_retries"):
            if key in ours_kwargs:
                base_kwargs[key] = ours_kwargs.pop(key)
                explicit_call.add(key)
        try:
            resolved_main = _resolve_formal_cell(
                optimizer, base_kwargs, explicit_call=explicit_call,
                explicit_harness=dict(harness_options or {}),
                explicit_workers=deployment_workers,
                explicit_extra=dict(extra or {}),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise OursConfigError(str(exc)) from exc
        given_sources = dict(ours_kwargs.pop("design_sources", None) or {})
        design_overrides = {k: ours_kwargs.pop(k) for k in list(ours_kwargs) if k in DESIGN_KWARGS}
        # ── the frozen main-experiment T / M / K, unless explicitly overridden ──
        #    (an explicit kwarg, a CLI flag / design file recorded in
        #    ``given_sources``, or a design object whose value differs from
        #    OursDesign's placeholder default).
        try:
            benchmark_name = ExperimentConfig(**base_kwargs).benchmark
            frozen = _frozen_design_defaults(optimizer, benchmark_name)
        except (TypeError, ValueError, KeyError) as exc:
            raise OursConfigError(str(exc)) from exc
        placeholder_design = OursDesign()
        frozen_sources = {}
        explicit_names = set(design_overrides) | set(given_sources)   # before any frozen fill-in

        def _explicit(name):
            return (name in explicit_names
                    or (design is not None
                        and getattr(design, name) != getattr(placeholder_design, name)))

        for name, (value, where) in frozen.items():
            if _explicit(name):
                continue
            # A frozen component's options belong to THAT component: when the
            # slot itself is overridden, its frozen options are not applied.
            if name.endswith("_options") and _explicit(name[: -len("_options")]):
                continue
            design_overrides[name] = value
            frozen_sources[name] = where
        if optimizer == "sso":
            ours_kwargs.setdefault(
                "rounds_source",
                frozen_sources.get("rounds") or given_sources.get(
                    "rounds", "override: explicit rounds (--rounds / design object)"),
            )
        try:
            resolved = (design or OursDesign()).with_overrides(**design_overrides)
        except (TypeError, ValueError) as exc:
            raise OursConfigError(str(exc)) from exc
        sources = _design_sources(resolved, design=design, overrides=design_overrides,
                                  given=given_sources)
        sources.update(frozen_sources)
        if optimizer == "sso":
            sources["rounds"] = ours_kwargs["rounds_source"]
        config = cls(base=ExperimentConfig(**base_kwargs), method=method, design=resolved,
                     design_sources=sources, **ours_kwargs)
        try:
            config.main_experiment_config = _record_cell(config, resolved_main)
        except (TypeError, ValueError, KeyError) as exc:
            raise OursConfigError(str(exc)) from exc
        return config


def _design_sources(resolved: OursDesign, *, design, overrides: dict, given: dict) -> dict:
    """design field -> source, for the T / M / K and component slots."""
    default = OursDesign()
    sources = {}
    for name in HYPERPARAMETERS + tuple(SLOTS):
        if name in given:
            sources[name] = given[name]
        elif name in overrides:
            sources[name] = f"override: explicit {name}"
        elif design is not None and getattr(resolved, name) != getattr(default, name):
            sources[name] = "override: design object"
        else:
            sources[name] = DESIGN_DEFAULT_SOURCE
    return sources


#: ``extra`` key under which the framework's embedding alias travels.
OURS_EMBEDDING_MODEL_KEY = "ours_embedding_model"


def _resolve_formal_cell(optimizer: str, base_kwargs: dict, *, explicit_call: set,
                         explicit_harness: dict, explicit_workers, explicit_extra: dict) -> dict:
    """Fill the +OURS cell's benchmark / solver contract (solver call, harness
    options, deployment workers), the native knobs it inherits and the
    framework's embedding alias from the authoritative main-experiment config
    unless explicit; return ``{field: {"value", "source"}}`` for them.

    Same for every optimizer, by construction: this is what makes a +OURS cell
    and its native cell run under one contract."""
    from integration.experiment import main_experiment_config as mec

    benchmark = ExperimentConfig(optimizer=optimizer, benchmark=base_kwargs["benchmark"],
                                 solver_model=base_kwargs["solver_model"]).benchmark
    record = {}
    for key, (value, where) in mec.solver_call(base_kwargs["solver_model"]).items():
        if key in explicit_call:
            record[key] = {"value": base_kwargs[key], "source": f"override: explicit {key}"}
        else:
            base_kwargs[key] = value
            record[key] = {"value": value, "source": where}
    options = {}
    for key, (value, where) in mec.harness_options(benchmark).items():
        options[key] = value
        record[f"harness_options.{key}"] = {"value": value, "source": where}
    for key, value in explicit_harness.items():
        options[key] = value
        record[f"harness_options.{key}"] = {"value": value, "source": f"override: --harness-option {key}"}
    base_kwargs["harness_options"] = options
    if explicit_workers is None:
        workers, where = mec.deployment_workers(benchmark)
        base_kwargs["deployment_workers"] = workers
        record["deployment_workers"] = {"value": workers, "source": where}
    else:
        record["deployment_workers"] = {"value": explicit_workers,
                                        "source": "override: explicit deployment_workers"}
    extra = dict(base_kwargs["extra"])
    inherited = mec.ours_optimizer_extras(optimizer)
    inherited[OURS_EMBEDDING_MODEL_KEY] = mec.ours_embedding_model()


    protocol = mec.ours_native_epoch_protocol(optimizer)
    if protocol is not None:
        inherited[f"{optimizer}_epoch_protocol"] = protocol
    for key, (value, where) in inherited.items():
        if key in explicit_extra:
            record[f"extra.{key}"] = {"value": explicit_extra[key], "source": f"override: --extra {key}"}
        else:
            extra[key] = value
            record[f"extra.{key}"] = {"value": value, "source": where}
    base_kwargs["extra"] = extra
    return record


def _record_cell(config: "OursRunConfig", resolved: dict) -> dict:
    """The full resolved formal configuration of a +OURS cell."""
    from integration.experiment.config import DEFAULT_OPTIMIZER_MODEL

    base = config.base
    record = {
        "benchmark": {"value": base.benchmark, "source": "CLI --benchmark"},
        "solver_model": {"value": base.solver_model, "source": "CLI --solver-model"},
        "optimizer_model": {
            "value": base.optimizer_model,
            "source": ("integration.experiment.config.DEFAULT_OPTIMIZER_MODEL (native default too)"
                       if base.optimizer_model == DEFAULT_OPTIMIZER_MODEL
                       else "override: --optimizer-model"),
        },
        **resolved,
    }
    for name in ("rounds", "beam_m", "k_subsets"):
        record[name] = {"value": getattr(config.design, name),
                        "source": config.design_sources.get(name, DESIGN_DEFAULT_SOURCE)}
    if config.optimizer == "sso":
        record.update(_sso_cell_settings(config))
    return record


def _frozen_design_defaults(optimizer: str, benchmark: str) -> dict:
    """``{rounds, beam_m, k_subsets} -> (value, source)`` frozen for this cell.

    T / M / K come from ``#ours.design`` for every optimizer. For SSO, T is
    ``sso_steps`` (the matched protocol shares X_t per step/round), and the
    two must agree -- a disagreement is a protocol error, not a preference."""
    from integration.experiment import main_experiment_config as mec

    frozen = dict(mec.ours_design_defaults())
    if optimizer == "sso":
        from integration.experiment.runners.sso import HPARAMS_SOURCE, main_experiment_training_schedule

        sso_steps = main_experiment_training_schedule(benchmark)["sso_steps"]
        if sso_steps != frozen["rounds"][0]:
            raise ValueError(
                f"matched protocol violated: {mec.source('ours.design.rounds')} = {frozen['rounds'][0]} "
                f"but {HPARAMS_SOURCE} sso_steps = {sso_steps}; SSO+OURS round t shares native "
                "step t's X_t, so the two must be equal"
            )
        frozen["rounds"] = (sso_steps, f"{HPARAMS_SOURCE} (sso_steps)")
    return frozen


def _sso_cell_settings(config: "OursRunConfig") -> dict:
    """SSO's own knobs and schedule, through the SAME resolvers native SSO uses."""
    from integration.experiment.runners.sso import resolve_sso_knobs, resolve_training_schedule

    base = config.base
    record = {}
    knobs = resolve_sso_knobs(base.extra)
    record["sso_history_per_step"] = {"value": knobs.history_per_step,
                                      "source": knobs.sources["sso_history_per_step"]}
    record["sso_num_probes"] = {"value": knobs.num_probes, "source": knobs.sources["sso_num_probes"]}
    record["sso_update_budget"] = {"value": knobs.update_budget,
                                   "source": knobs.sources["sso_update_budget"]}
    training = resolve_training_schedule(base.benchmark, base.extra,
                                         steps=(config.rounds, config.rounds_source))
    record["rounds"] = {"value": training.steps, "source": training.sources["sso_steps"]}
    record["sso_train_per_step"] = {"value": training.train_per_step,
                                    "source": training.sources["sso_train_per_step"]}
    record["sso_schedule_seed"] = {"value": training.seed, "source": training.sources["sso_schedule_seed"]}
    return record


def _default_run_id(optimizer, benchmark, method, harness, solver_model) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{optimizer}+{method}__{benchmark}__{harness}__{solver_model}__{stamp}"


__all__ = [
    "DEPLOYMENT_RULES", "METHODS", "OPERATORS", "OURS_EMBEDDING_MODEL_KEY", "OURS_OPTIMIZERS",
    "PARTITIONERS", "TIE_BREAKS", "OursConfigError", "OursRunConfig",
]
