"""Route mutation, crossover, and regeneration using branch histories."""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

from ours import history
from ours.operator import OperatorProposal
from ours.operator_recombination import (
    GlobalDiversityRecombinationOperator,
    OperatorOutputError,
    _REPAIR_NOTE,
    _approx_tokens,
    _extract_json,
    _head_tail,
    _protected_note,
    _unique_in_order,
)
from ours.registry import register
from ours.resources import ROLE_OPERATOR
from ours.types import CandidateState, RawProposal, skill_fingerprint


ROLES = (history.KIND_MUTATION, history.KIND_CROSSOVER, history.KIND_REGENERATION)

SYSTEM_PROMPT = """You are the evolution operator family of a skill-optimization search.

A "skill" is a natural-language document prepended to a frozen model's instructions to make it solve a benchmark's tasks better. Several optimizers each proposed edited skills ("raw proposals"), each from ONE subset of the evidence gathered when a parent skill was run on training tasks. You are also shown, per branch, its HISTORY: the issues it has run into, which of them recur, and which fixes were already tried.

Your job is to produce new offspring skills by applying, per offspring, exactly ONE of three evolution operators. A separate held-out validation decides which candidates survive; you NEVER see those scores.

The three operators:
1. mutation: make a LOCAL change to the parent skill aimed at a persistent, still-unresolved issue. Do not rewrite the whole skill.
2. crossover: COMBINE the complementary, evidence-supported strategies of two or more different branches / raw proposals into one skill; resolve their conflicts.
3. regeneration: when a branch has repeatedly failed to fix an issue, STOP patching its old skill and write a NEW skill from scratch grounded in the accumulated evidence for that issue.

You are given a ROUTING PLAN that assigns each offspring an operator, a source branch and the target issue(s). Follow it: produce exactly one offspring per plan entry, in order, using the assigned operator.

Constraints:
- Every change you keep must be supported by the evidence shown; do not invent task facts.
- Do NOT return a parent skill unchanged; parents are already in the pool.
- The offspring must differ from each other in substance.
- Each offspring is a COMPLETE skill document, not an edit or a diff.

Return ONLY a JSON object of this exact shape, nothing else:
{"candidates": [{"operator_type": "mutation|crossover|regeneration", "target_issue_ids": ["<issue id>", ...], "source_proposal_ids": ["<raw proposal id>", ...], "rationale": "<one or two sentences>", "skill": "<the complete skill document>"}]}
"""

_USER_TEMPLATE = """Round {round_id}. Produce exactly N = {n_op} offspring, one per ROUTING PLAN entry, in order.

=== PARENTS (already in the pool; do not return unchanged) ===
{parents}

=== RAW PROPOSALS (operator material; each came from ONE evidence subset of its parent) ===
{proposals}

=== BRANCH HISTORIES (issues each branch ran into and fixes already tried) ===
{histories}

=== ROUTING PLAN (which operator produces each offspring, its source branch, its target issues) ===
{routing_plan}

=== EVIDENCE (the native evidence each subset was conditioned on{truncation_note}) ===
{evidence}
"""


@register("operator", "history_routed_operator_family")
class HistoryRoutedOperatorFamily(GlobalDiversityRecombinationOperator):
    """See the module docstring. Reuses the base operator's machinery; overrides
    the prompt, the deterministic routing and the per-candidate post-processing."""

    name = "history_routed_operator_family"
    is_placeholder = False

    def __init__(self, *, max_candidates: int = 3, prompt_char_budget: int = 400_000,
                 max_tokens: int | None = None, retries: int = 1,
                 regen_min_rounds: int = 2, regen_min_attempts: int = 2) -> None:
        super().__init__(max_candidates=max_candidates, prompt_char_budget=prompt_char_budget,
                         max_tokens=max_tokens, retries=retries)
        self.regen_min_rounds = int(regen_min_rounds)
        self.regen_min_attempts = int(regen_min_attempts)

    # ── deterministic routing ────────────────────────────────────────────────

    def compute_routing_plan(self, parents: Sequence[CandidateState],
                             unique_proposals: Sequence[RawProposal], n_op: int,
                             *, mode: str = "round") -> list[dict]:
        """Assign an operator, source branches, and target issues to each candidate slot."""
        if n_op <= 0:
            return []
        if mode == "final_merge":
            # Beam_T -> deployment merge: crossover of the survivors, N = 1.
            return [{"operator_type": history.KIND_CROSSOVER,
                     "source_branch": [q.parent_id for q in unique_proposals],
                     "target_issue_ids": []} for _ in range(n_op)]

        parent_by_id = {p.candidate_id: p for p in parents}
        plan: list[dict] = []

        # 1) regeneration slots: a branch stuck on a recurring issue it has
        #    repeatedly failed to fix.
        for parent in parents:
            if len(plan) >= n_op:
                break
            h = parent.metadata.get("history")
            targets = []
            for issue in history.issues(h):
                if (issue.get("status") == history.STATUS_RECURRING
                        and int(issue.get("rounds_seen", 0) or 0) >= self.regen_min_rounds
                        and len(history.attempts_for_issue(h, issue.get("issue_id")))
                        >= self.regen_min_attempts):
                    targets.append(str(issue.get("issue_id")))
            if targets:
                plan.append({"operator_type": history.KIND_REGENERATION,
                             "source_branch": [parent.candidate_id],
                             "target_issue_ids": sorted(targets)})

        # 2) one crossover slot: >= 2 distinct sources with differing target sets.
        if len(plan) < n_op:
            by_targets: dict[frozenset, list[tuple[str, str]]] = {}
            order: list[frozenset] = []
            for q in unique_proposals:
                parent = parent_by_id.get(q.parent_id)
                h = parent.metadata.get("history") if parent is not None else None
                subset_tasks = {str(t) for t in (q.provenance.get("subset_task_ids") or [])}
                targeted = frozenset(
                    str(issue.get("issue_id")) for issue in history.issues(h)
                    if subset_tasks & {str(t) for t in (issue.get("task_ids") or [])})
                if targeted not in by_targets:
                    by_targets[targeted] = []
                    order.append(targeted)
                by_targets[targeted].append((q.proposal_id, q.parent_id))
            if len(order) >= 2:
                chosen = [by_targets[order[0]][0], by_targets[order[1]][0]]
                target = sorted(set(order[0]) | set(order[1]))
                plan.append({"operator_type": history.KIND_CROSSOVER,
                             "source_branch": sorted({par for _, par in chosen}),
                             "target_issue_ids": target})


        untargeted = 0
        while len(plan) < n_op:
            pick = self._least_attempted_unresolved(parents)
            if pick is None:
                source = [parents[untargeted % len(parents)].candidate_id] if parents else []
                untargeted += 1
                plan.append({"operator_type": history.KIND_MUTATION,
                             "source_branch": source,
                             "target_issue_ids": []})
            else:
                issue_id, parent_id = pick
                plan.append({"operator_type": history.KIND_MUTATION,
                             "source_branch": [parent_id], "target_issue_ids": [issue_id]})
        return plan[:n_op]

    @staticmethod
    def _least_attempted_unresolved(parents: Sequence[CandidateState]) -> tuple[str, str] | None:
        best: tuple[int, str, str] | None = None
        for parent in parents:
            h = parent.metadata.get("history")
            for issue in history.unresolved_issues(h):
                iid = str(issue.get("issue_id"))
                n_att = len(history.attempts_for_issue(h, iid))
                key = (n_att, iid, parent.candidate_id)
                if best is None or key < best:
                    best = key
        return (best[1], best[2]) if best is not None else None

    # ── prompt ───────────────────────────────────────────────────────────────

    def _render_histories(self, parents: Sequence[CandidateState]) -> str:
        parts = [history.render_for_operator(p.metadata.get("history"), branch_id=p.candidate_id)
                 for p in parents]
        return "\n\n".join(parts) if parts else "(none)"

    @staticmethod
    def _render_routing_plan(plan: Sequence[dict]) -> str:
        if not plan:
            return "(none)"
        lines = []
        for idx, entry in enumerate(plan, 1):
            src = ", ".join(entry.get("source_branch") or []) or "?"
            targets = ", ".join(entry.get("target_issue_ids") or []) or "(none)"
            lines.append(f"{idx}. operator={entry['operator_type']}; source branch: {src}; "
                         f"target issues: {targets}")
        return "\n".join(lines)

    def build_prompt(self, *, parents, proposals, evidence, round_id, n_op, routing_plan=None):
        routing_plan = routing_plan or []
        parents_text = _protected_note(parents, proposals) + self._render_parents(parents)
        proposals_text = self._render_proposals(proposals)
        histories_text = self._render_histories(parents)
        routing_text = self._render_routing_plan(routing_plan)
        records = self._evidence_records(proposals, evidence)
        fixed = _USER_TEMPLATE.format(round_id=round_id, n_op=n_op, parents=parents_text,
                                      proposals=proposals_text, histories=histories_text,
                                      routing_plan=routing_text, truncation_note="", evidence="")
        remaining = self.prompt_char_budget - len(SYSTEM_PROMPT) - len(fixed)
        original_chars = sum(len(r["text"]) for r in records)
        per_record = None
        truncated = False
        if records and original_chars > max(remaining, 0):
            truncated = True
            per_record = max(remaining, 0) // len(records)
        kept_chars = 0
        parts = []
        for r in records:
            text = r["text"] if per_record is None else _head_tail(r["text"], per_record)
            kept_chars += len(text)
            header = f"--- evidence for {r['proposal_id']} / subset {r['subset_id']} / task {r['task_id']} ---"
            parts.append(f"{header}\n{text}" if text else f"{header}\n(omitted: no budget left)")
        note = ("; long records truncated head+tail under an equal per-record budget"
                if truncated else "")
        user = _USER_TEMPLATE.format(round_id=round_id, n_op=n_op, parents=parents_text,
                                     proposals=proposals_text, histories=histories_text,
                                     routing_plan=routing_text, truncation_note=note,
                                     evidence="\n\n".join(parts) if parts else "(none)")
        budget = {
            "prompt_char_budget": self.prompt_char_budget,
            "prompt_chars": len(SYSTEM_PROMPT) + len(user),
            "prompt_tokens_approx": _approx_tokens(SYSTEM_PROMPT + user),
            "evidence_records": len(records),
            "evidence_chars_original": original_chars,
            "evidence_chars_kept": kept_chars,
            "evidence_truncated": truncated,
            "per_record_char_budget": per_record,
        }
        return user, budget

    # ── the round ──────────────────────────────────────────────────────────

    def _run(self, *, parents, proposals, evidence, round_label, round_id, allocate_dir,
             max_candidates: int, mode: str):
        unique = _unique_in_order(proposals)
        n_op = min(max_candidates, len(unique))
        plan = self.compute_routing_plan(parents, unique, n_op, mode=mode)
        report: dict[str, Any] = {
            "operator": self.name, "mode": mode, "round_id": round_id,
            "n_parents": len(parents), "n_proposals": len(proposals),
            "n_unique_proposals": len(unique), "n_op_target": n_op,
            "roles": [entry["operator_type"] for entry in plan], "routing_plan": plan,
            "regen_min_rounds": self.regen_min_rounds, "regen_min_attempts": self.regen_min_attempts,
            "llm_calls": 0, "fallback": False,
            "dropped_parent_identical": 0, "dropped_duplicate": 0, "dropped_invalid": 0,
        }
        if n_op == 0:
            self.last_report = report
            return []

        resources = self.require_resources("the history-routed operator")
        work_dir = allocate_dir("recombination" if mode == "round" else "final_merge")
        user, budget = self.build_prompt(parents=parents, proposals=unique, evidence=evidence,
                                         round_id=round_label, n_op=n_op, routing_plan=plan)
        report["budget"] = budget
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        with open(os.path.join(work_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(SYSTEM_PROMPT + "\n\n" + user)

        raw_candidates: list[dict] | None = None
        errors: list[str] = []
        for attempt in range(1 + self.retries):
            attempt_messages = messages if attempt == 0 else [
                messages[0], {"role": "user", "content": user + _REPAIR_NOTE}]
            kwargs = {} if self.max_tokens is None else {"max_tokens": self.max_tokens}
            message = resources.chat(attempt_messages, role=ROLE_OPERATOR, **kwargs)
            report["llm_calls"] += 1
            finish = getattr(resources, "last_finish_reason", None)
            report.setdefault("finish_reasons", []).append(finish)
            text = getattr(message, "content", None) or ""
            with open(os.path.join(work_dir, f"response_{attempt}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            try:
                if finish == "length":
                    raise OperatorOutputError("reply truncated at max_tokens (finish_reason=length)")
                payload = _extract_json(text)
                items = payload.get("candidates")
                if not isinstance(items, list) or not items:
                    raise OperatorOutputError("no candidates in reply")
                raw_candidates = items
                break
            except OperatorOutputError as exc:
                errors.append(f"attempt {attempt}: {exc}")
        report["parse_errors"] = errors

        if raw_candidates is None:
            out = self._fallback(unique, n_op, parents, report)
        else:
            out = self._post_process(raw_candidates, unique, parents, n_op, report)
        self._annotate_routing(out, plan)
        report["n_emitted"] = len(out)
        self.last_report = report
        with open(os.path.join(work_dir, "operator_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        for op in out:
            op.provenance["operator_report"] = {k: v for k, v in report.items() if k != "parse_errors"}
        return out

    @staticmethod
    def _annotate_routing(out: Sequence[OperatorProposal], plan: Sequence[dict]) -> None:
        """Ensure every emitted offspring carries an ``operator_type`` and
        ``target_issue_ids``; fill from the routing plan by slot when the model
        (or the fallback) did not set them."""
        for index, op in enumerate(out):
            entry = plan[index] if index < len(plan) else {}
            op.provenance.setdefault("operator_type", entry.get("operator_type") or history.KIND_MUTATION)
            op.provenance.setdefault("target_issue_ids", list(entry.get("target_issue_ids") or []))
            op.provenance.setdefault("routed_source_branch", list(entry.get("source_branch") or []))

    def _post_process(self, items, unique, parents, n_op, report):
        by_id = {q.proposal_id: q for q in unique}
        parent_hashes = {p.skill_hash: p.candidate_id for p in parents}
        raw_hashes = {q.skill_hash: q.proposal_id for q in unique}
        parent_by_id = {p.candidate_id: p for p in parents}
        plan = report.get("routing_plan") or []
        seen_hashes: set = set()
        out: list[OperatorProposal] = []
        for index, item in enumerate(items):
            if len(out) >= n_op:
                break
            if not isinstance(item, dict) or not str(item.get("skill") or "").strip():
                report["dropped_invalid"] += 1
                continue
            skill = str(item["skill"])
            slot = plan[len(out)] if len(out) < len(plan) else {}
            operator_type = str(item.get("operator_type") or "")
            if operator_type not in ROLES:
                operator_type = slot.get("operator_type") or ROLES[min(index, len(ROLES) - 1)]
            target_issue_ids = [str(t) for t in (item.get("target_issue_ids")
                                                 or slot.get("target_issue_ids") or [])]
            h = skill_fingerprint(skill)
            if h in parent_hashes:
                report["dropped_parent_identical"] += 1
                continue
            if h in seen_hashes:
                report["dropped_duplicate"] += 1
                continue
            seen_hashes.add(h)
            declared = [str(s) for s in (item.get("source_proposal_ids") or []) if str(s) in by_id]
            sources = [by_id[s] for s in declared]
            if not sources and h in raw_hashes:
                sources = [by_id[raw_hashes[h]]]
            source_parents = []
            for q in sources:
                if q.parent_id not in source_parents:
                    source_parents.append(q.parent_id)
            anchor = (parent_by_id.get(source_parents[0]) if source_parents
                      else (parents[0] if parents else None))
            provenance = {
                "role": operator_type,
                "operator_type": operator_type,
                "target_issue_ids": target_issue_ids,
                "rationale": str(item.get("rationale") or ""),
                "preserved_raw_proposal": raw_hashes.get(h),
                "declared_source_proposal_ids": [str(s) for s in (item.get("source_proposal_ids") or [])],
                "fallback": False,
                **self._lineage_from(sources, anchor),
            }
            out.append(OperatorProposal(
                skill=skill, provenance=provenance,
                source_proposal_ids=tuple(q.proposal_id for q in sources),
                source_parent_ids=tuple(source_parents),
            ))
        return out


__all__ = ["ROLES", "SYSTEM_PROMPT", "HistoryRoutedOperatorFamily"]
