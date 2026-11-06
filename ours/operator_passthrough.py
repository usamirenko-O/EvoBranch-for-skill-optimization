"""Select raw proposals evenly across branches for the no-evolution ablation."""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

from ours.operator import OperatorProposal, SkillOperator
from ours.registry import register
from ours.types import CandidateState, Evidence, RawProposal

#: The attempt kind recorded on the offspring's inherited branch history.
KIND_PASSTHROUGH = "raw_proposal_passthrough"


def select_branch_balanced(parents: Sequence[CandidateState], proposals: Sequence[RawProposal],
                           n_op: int) -> tuple[list[RawProposal], list[dict]]:
    """The deterministic selection rule, as data: ``(selected, plan)``.

    ``plan`` has one entry per selected proposal: ``{slot, parent_id,
    proposal_id}``; skipped proposals are reported in the caller's report."""
    parent_ids = [p.candidate_id for p in parents]
    parent_hashes = {p.skill_hash for p in parents}
    queues: dict[str, list[RawProposal]] = {pid: [] for pid in parent_ids}
    for q in sorted(proposals, key=lambda q: q.proposal_id):
        queues.setdefault(q.parent_id, []).append(q)
    order = parent_ids + [pid for pid in queues if pid not in parent_ids]
    seen_hashes: set = set()
    selected: list[RawProposal] = []
    plan: list[dict] = []
    if n_op <= 0 or not order:
        return selected, plan
    slot = 0
    remaining = sum(len(v) for v in queues.values())
    while len(selected) < n_op and remaining > 0:
        parent_id = order[slot % len(order)]
        slot += 1
        queue = queues.get(parent_id) or []
        picked = None
        while queue:
            q = queue.pop(0)
            remaining -= 1
            if q.skill_hash in parent_hashes or q.skill_hash in seen_hashes:
                continue          # identical to a parent / already selected: dropped
            picked = q
            break
        if picked is None:
            continue              # this parent had nothing usable left; next parent
        seen_hashes.add(picked.skill_hash)
        selected.append(picked)
        plan.append({"slot": len(selected) - 1, "parent_id": picked.parent_id,
                     "proposal_id": picked.proposal_id, "subset_id": picked.evidence_subset_id})
    return selected, plan


@register("operator", "branch_balanced_passthrough")
class BranchBalancedPassthroughOperator(SkillOperator):
    """See the module docstring. ``max_candidates`` must equal the full
    operator's (main experiment: 3) so the per-round VAL budget is matched."""

    name = "branch_balanced_passthrough"
    is_placeholder = False

    def __init__(self, *, max_candidates: int = 3) -> None:
        if int(max_candidates) < 1:
            raise ValueError("max_candidates must be >= 1")
        self.max_candidates = int(max_candidates)
        self.last_report: dict | None = None

    def expand(self, *, parents, proposals, evidence: Sequence[Evidence], round_id, allocate_dir):
        unique_hashes = {q.skill_hash for q in proposals}
        n_op = min(self.max_candidates, len(unique_hashes))
        selected, plan = select_branch_balanced(parents, proposals, n_op)
        report: dict[str, Any] = {
            "operator": self.name, "mode": "round", "round_id": round_id,
            "n_parents": len(parents), "n_proposals": len(proposals),
            "n_unique_proposals": len(unique_hashes), "n_op_target": n_op,
            "budget_rule": "N_op = min(max_candidates, #unique raw proposals) -- same as the full operator",
            "selection_rule": "branch-balanced round-robin over parents in beam order, "
                              "proposals in id order; no VAL, no model",
            "passthrough_plan": plan,
            "selected_proposal_ids": [q.proposal_id for q in selected],
            "skipped_proposal_ids": [q.proposal_id for q in proposals
                                     if q.proposal_id not in {s.proposal_id for s in selected}],
            "llm_calls": 0, "fallback": False, "n_emitted": len(selected),
        }
        self.last_report = report
        work_dir = allocate_dir("passthrough")
        with open(os.path.join(work_dir, "operator_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        out: list[OperatorProposal] = []
        for entry, q in zip(plan, selected):
            out.append(OperatorProposal(
                skill=q.skill,
                provenance={
                    "role": KIND_PASSTHROUGH,
                    "operator_type": KIND_PASSTHROUGH,
                    "target_issue_ids": [],
                    "rationale": f"raw proposal {q.proposal_id} of parent {q.parent_id} "
                                 f"passed through unchanged (slot {entry['slot']})",
                    "preserved_raw_proposal": q.proposal_id,
                    "declared_source_proposal_ids": [q.proposal_id],
                    "fallback": False,
                    "operator_report": report,
                },
                source_proposal_ids=(q.proposal_id,),
                source_parent_ids=(q.parent_id,),
            ))
        return out


__all__ = ["KIND_PASSTHROUGH", "BranchBalancedPassthroughOperator", "select_branch_balanced"]
