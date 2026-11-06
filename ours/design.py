"""Configure and validate the components of an OURS search."""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ours import registry

#: Modules whose import registers components. Add yours here.
COMPONENT_MODULES = (
    "ours.partition",
    "ours.operator",
    "ours.operator_recombination",
    "ours.operator_history_routed",
    "ours.operator_passthrough",
    "ours.beam",
    "ours.deployment_rule",
    "ours.issue_tracker",
)
for _module in COMPONENT_MODULES:
    importlib.import_module(_module)

#: slot field -> registry kind
SLOTS = {
    "partitioner": "partitioner",
    "operator": "operator",
    "tie_break": "tie_break",
    "deployment_rule": "deployment_rule",
}
HYPERPARAMETERS = ("rounds", "beam_m", "k_subsets")


class DesignError(ValueError):
    """The design names a component that does not exist or an invalid value."""


@dataclass(frozen=True)
class OursDesign:
    # ── hyperparameters (placeholder defaults; a frozen main-experiment
    #    default, where one exists, is applied by the run configuration) ──
    rounds: int = 2
    beam_m: int = 2
    k_subsets: int = 3

    # ── scientific components, by registered name ────────────────────────
    partitioner: str = "round_robin_placeholder"
    operator: str = "pass_through_placeholder"
    tie_break: str = "provisional_candidate_id_ascending"
    deployment_rule: str = "top_validation_score_placeholder"


    tracker: str | None = None


    attempt_aware_proposal: bool = True

    # ── constructor keyword arguments for each component ────────────────
    partitioner_options: dict = field(default_factory=dict)
    operator_options: dict = field(default_factory=dict)
    tie_break_options: dict = field(default_factory=dict)
    deployment_rule_options: dict = field(default_factory=dict)
    tracker_options: dict = field(default_factory=dict)

    #: Fail on a Top-M cutoff tie that a PLACEHOLDER tie-break would decide.
    strict_ties: bool = False

    def __post_init__(self) -> None:
        for name in HYPERPARAMETERS:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise DesignError(f"{name} must be an integer >= 1, got {value!r}")
        for slot, kind in SLOTS.items():
            if getattr(self, slot) not in registry.names(kind):
                raise DesignError(
                    f"{slot}={getattr(self, slot)!r} is not a registered {kind}; "
                    f"available: {registry.names(kind)}"
                )
            if not isinstance(getattr(self, f"{slot}_options"), dict):
                raise DesignError(f"{slot}_options must be a dict")
        if self.tracker is not None and self.tracker not in registry.names("tracker"):
            raise DesignError(
                f"tracker={self.tracker!r} is not a registered tracker; "
                f"available: {registry.names('tracker')}"
            )
        if not isinstance(self.tracker_options, dict):
            raise DesignError("tracker_options must be a dict")
        if type(self.attempt_aware_proposal) is not bool:
            raise DesignError("attempt_aware_proposal must be a bool")

    # ── construction ─────────────────────────────────────────────────────

    def _build(self, slot: str):
        return registry.build(SLOTS[slot], getattr(self, slot),
                              **getattr(self, f"{slot}_options"))

    def build_partitioner(self):
        return self._build("partitioner")

    def build_operator(self):
        return self._build("operator")

    def build_tie_break(self):
        return self._build("tie_break")

    def build_deployment_rule(self):
        return self._build("deployment_rule")

    def build_tracker(self):
        """The per-branch issue tracker, or None when the history layer is off."""
        if self.tracker is None:
            return None
        return registry.build("tracker", self.tracker, **self.tracker_options)

    def build_beam_manager(self, *, resources=None):
        from ours.beam import BeamManager

        tie_break = self.build_tie_break()
        tie_break.bind(resources)
        return BeamManager(m=self.beam_m, strict_ties=self.strict_ties, tie_break=tie_break)

    def build_controller(self, *, adapter, evaluator, workspace, seed: int | None = None,
                         resources=None):
        """The OURS round for ANY optimizer adapter, configured by this design.

        ``resources`` (:class:`~ours.resources.RunResources`) is bound to the
        partitioner, the operator and the tie-break, so a component that calls
        a model does it on the run's optimizer-side alias and its spend is
        metered into the run's optimization cost. Every run driver passes it;
        ``None`` is for offline use and makes a model-calling component fail
        loudly rather than fall back.
        """
        from ours.controller import SearchController

        partitioner = self.build_partitioner()
        partitioner.bind(resources)
        operator = self.build_operator()
        operator.bind(resources)
        tracker = self.build_tracker()
        if tracker is not None:
            tracker.bind(resources)
        return SearchController(
            adapter=adapter,
            partitioner=partitioner,
            operator=operator,
            evaluator=evaluator,
            beam_manager=self.build_beam_manager(resources=resources),
            workspace=workspace,
            k_subsets=self.k_subsets,
            seed=seed,
            resources=resources,
            tracker=tracker,
            attempt_aware_proposal=self.attempt_aware_proposal,
        )

    def preflight(self) -> list[str]:
        """Design-level blockers, checked before any spend."""
        errors = list(self.build_deployment_rule().check(self.beam_m))
        if self.deployment_rule == "final_merge_top1" and not hasattr(self.build_operator(), "final_merge"):
            errors.append(
                f"deployment_rule=final_merge_top1 needs an operator with final_merge(); "
                f"operator={self.operator!r} has none (an ablation without global evolution "
                "uses deployment_rule=beam_top1_no_merge)"
            )
        if self.beam_m > 1 and self.operator == "noop_placeholder":
            errors.append(
                "operator=noop_placeholder never produces a challenger, so the beam "
                "can never grow past Beam_0; use another operator or beam_m=1"
            )
        return errors

    # ── reporting ────────────────────────────────────────────────────────

    def component_report(self, *, hyperparameter_status: dict | None = None,
                         sources: dict | None = None) -> dict:
        """Per slot: the name in force, its options, whether it is a placeholder
        and its source. Under ``"hyperparameters"``, per T / M / K: the value,
        its scientific status, the frozen main-experiment default (if any),
        whether it was overridden, and its source.

        ``hyperparameter_status`` (from the run configuration) supplies the
        per-hyperparameter entries; without it each is a placeholder of unknown
        source.
        """
        sources = sources or {}
        report = {}
        for slot in SLOTS:
            component = self._build(slot)
            report[slot] = {
                "name": getattr(self, slot),
                "options": dict(getattr(self, f"{slot}_options")),
                "is_placeholder": bool(getattr(component, "is_placeholder", False)),
                "source": sources.get(slot),
            }
        status = hyperparameter_status or {}
        report["hyperparameters"] = {
            name: status.get(name) or {
                "value": getattr(self, name),
                "status": "placeholder",
                "is_placeholder": True,
                "frozen_main_experiment_default": None,
                "overridden": None,
                "source": sources.get(name),
            }
            for name in HYPERPARAMETERS
        }
        return report

    def placeholders_in_force(self, *, hyperparameter_status: dict | None = None) -> list[str]:
        """Slots and hyperparameters whose scientific choice is still a placeholder."""
        report = self.component_report(hyperparameter_status=hyperparameter_status)
        hyper = report.pop("hyperparameters")
        return ([slot for slot, entry in report.items() if entry["is_placeholder"]]
                + [name for name, entry in hyper.items() if entry["is_placeholder"]])

    def to_dict(self) -> dict:
        return asdict(self)

    # ── loading ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "OursDesign":
        if "hyperparameters_are_placeholders" in data:
            raise DesignError(
                "hyperparameters_are_placeholders was removed: each hyperparameter's "
                "status (placeholder / frozen main-experiment default) is now recorded "
                "per optimizer by the run configuration; drop the key"
            )
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise DesignError(f"unknown design fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: Path | str) -> "OursDesign":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, **overrides: Any) -> "OursDesign":
        """Apply non-None overrides; ``<slot>.<key>`` keys set component options."""
        plain = {k: v for k, v in overrides.items() if v is not None and "." not in k}
        design = replace(self, **plain) if plain else self
        dotted = {k: v for k, v in overrides.items() if v is not None and "." in k}
        for key, value in dotted.items():
            slot, _, option = key.partition(".")
            if slot not in SLOTS:
                raise DesignError(f"--design-option {key!r}: unknown slot {slot!r}")
            options = dict(getattr(design, f"{slot}_options"))
            options[option] = value
            design = replace(design, **{f"{slot}_options": options})
        return design


__all__ = ["COMPONENT_MODULES", "DesignError", "HYPERPARAMETERS", "OursDesign", "SLOTS"]
