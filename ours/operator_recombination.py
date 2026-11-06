"""Generate diverse skill candidates from proposals, parents, and training evidence."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Sequence

from ours.operator import OperatorProposal, SkillOperator
from ours.registry import register
from ours.resources import ROLE_OPERATOR
from ours.types import CandidateState, Evidence, RawProposal, skill_fingerprint

ROLES = ("exploit_merge", "alternative_recombination", "diversity_explore")

#: Chars per token, used ONLY to report approximate token counts.
_CHARS_PER_TOKEN = 4

#: Provenance keys an offspring inherits along its lineage.
_LINEAGE_KEYS = ("analysis_id", "contextual_abstract")

SYSTEM_PROMPT = """You are the recombination operator of a skill-optimization search.

A "skill" is a natural-language document that is prepended to a frozen model's instructions to make it solve a benchmark's tasks better. Several optimizers each proposed edited skills ("raw proposals"), each from ONE subset of the evidence gathered when a parent skill was run on training tasks.

Your job is to RECOMBINE and EDIT, not to judge. A separate held-out validation decides which candidates survive; you never see those scores. Produce at most N offspring skills, each a COMPLETE skill document, following these roles in order:

1. exploit_merge: absorb the complementary, evidence-supported changes of several raw proposals into one skill; resolve conflicts between them; the strongest consensus child.
2. alternative_recombination: a different, compatible combination of the raw proposals' changes that deliberately does not take the same route as exploit_merge.
3. diversity_explore: keep one clearly different but still evidence-supported direction alive, so that merging does not collapse the proposals into one flavour.

Constraints:
- Every change you keep must be supported by the evidence shown; do not invent task facts.
- You may return a raw proposal unchanged if it already is a strong, distinct direction.
- Do not return a parent skill unchanged; parents are already in the pool.
- The offspring must differ from each other in substance.
- If N is smaller than 3, produce only the first N roles.

Return ONLY a JSON object of this exact shape, nothing else:
{"candidates": [{"role": "<one of the roles above>", "source_proposal_ids": ["<raw proposal id>", ...], "rationale": "<one or two sentences>", "skill": "<the complete skill document>"}]}
"""

_USER_TEMPLATE = """Round {round_id}. Produce at most N = {n_op} offspring.

=== PARENTS (already in the pool; do not return unchanged) ===
{parents}

=== RAW PROPOSALS (operator material; each came from ONE evidence subset of its parent) ===
{proposals}

=== EVIDENCE (the native evidence each subset was conditioned on{truncation_note}) ===
{evidence}
"""

_REPAIR_NOTE = (
    "\n\nYour previous reply could not be parsed. Return ONLY the JSON object described "
    "in the instructions, with no prose before or after it."
)


class OperatorOutputError(ValueError):
    """The model's reply was empty or not the required JSON."""


_PROTECTED_START, _PROTECTED_END = "<!-- SLOW_UPDATE_START -->", "<!-- SLOW_UPDATE_END -->"
_PROTECTED_NOTE = (
    "PROTECTED BLOCK RULE: a skill may contain a block between "
    f"{_PROTECTED_START} and {_PROTECTED_END}. It is written by the native optimizer's "
    "epoch-level update and is PROTECTED: reproduce the source branch's block VERBATIM in every "
    "offspring, never edit, merge, shorten or drop its contents, and make all of your changes "
    "outside it.\n\n"
)


def _protected_note(parents, proposals) -> str:
    if any(_PROTECTED_START in (x.skill or "") for x in list(parents) + list(proposals)):
        return _PROTECTED_NOTE
    return ""


def _approx_tokens(text: str) -> int:
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _head_tail(text: str, keep: int) -> str:
    """Deterministic truncation: keep half from the head and half from the tail."""
    if keep <= 0:
        return ""
    if len(text) <= keep:
        return text
    marker = "\n[... truncated ...]\n"
    if keep <= len(marker) + 2:
        return text[:keep]
    half = (keep - len(marker)) // 2
    return text[:half] + marker + text[len(text) - (keep - len(marker) - half):]


def _unique_in_order(proposals: Sequence[RawProposal]) -> list[RawProposal]:
    seen: set = set()
    out = []
    for q in sorted(proposals, key=lambda q: q.proposal_id):
        if q.skill_hash in seen:
            continue
        seen.add(q.skill_hash)
        out.append(q)
    return out


def _extract_json(text: str) -> dict:
    """The first JSON object in the reply (a ```json fence is tolerated)."""
    if not text or not text.strip():
        raise OperatorOutputError("empty reply")
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.S)
    if fence:
        body = fence.group(1)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise OperatorOutputError("no JSON object in reply")
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError as exc:
        raise OperatorOutputError(f"invalid JSON: {exc}") from exc


@register("operator", "global_diversity_recombination")
class GlobalDiversityRecombinationOperator(SkillOperator):
    """See the module docstring. Options are engineering, recorded per round."""

    name = "global_diversity_recombination"
    is_placeholder = False

    def __init__(self, *, max_candidates: int = 3, prompt_char_budget: int = 400_000,
                 max_tokens: int | None = None, retries: int = 1) -> None:
        if int(max_candidates) < 1:
            raise ValueError("max_candidates must be >= 1")
        self.max_candidates = int(max_candidates)
        self.prompt_char_budget = int(prompt_char_budget)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.retries = max(0, int(retries))
        self.last_report: dict | None = None

    # ── prompt assembly ─────────────────────────────────────────────────

    def _render_parents(self, parents: Sequence[CandidateState]) -> str:
        parts = []
        for p in parents:
            lineage = " -> ".join(p.lineage) if p.lineage else "(seed)"
            parts.append(f"--- parent {p.candidate_id} (lineage: {lineage}) ---\n{p.skill}")
        return "\n\n".join(parts) if parts else "(none)"

    def _render_proposals(self, proposals: Sequence[RawProposal]) -> str:
        parts = []
        for q in proposals:
            subset_ids = q.provenance.get("subset_task_ids") or []
            parts.append(
                f"--- raw proposal {q.proposal_id} (parent: {q.parent_id}; evidence subset: "
                f"{q.evidence_subset_id}; tasks: {', '.join(map(str, subset_ids)) or '?'}) ---\n{q.skill}"
            )
        return "\n\n".join(parts) if parts else "(none)"

    def _evidence_records(self, proposals: Sequence[RawProposal],
                          evidence: Sequence[Evidence]) -> list[dict]:
        """One record per (proposal, task) the proposal was conditioned on."""
        by_parent = {e.parent_id: e for e in evidence}
        records = []
        for q in proposals:
            ev = by_parent.get(q.parent_id)
            texts = (ev.metadata.get("_native_text_by_task") if ev is not None else None) or {}
            for tid in q.provenance.get("subset_task_ids") or []:
                text = texts.get(str(tid))
                if text is None:
                    continue
                records.append({"proposal_id": q.proposal_id, "subset_id": q.evidence_subset_id,
                                "task_id": str(tid), "text": str(text)})
        return records

    def build_prompt(self, *, parents, proposals, evidence, round_id: int, n_op: int) -> tuple[str, dict]:
        parents_text = _protected_note(parents, proposals) + self._render_parents(parents)
        proposals_text = self._render_proposals(proposals)
        records = self._evidence_records(proposals, evidence)
        fixed = _USER_TEMPLATE.format(round_id=round_id, n_op=n_op, parents=parents_text,
                                      proposals=proposals_text, truncation_note="", evidence="")
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
                                     proposals=proposals_text, truncation_note=note,
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
            "chars_per_token_assumed": _CHARS_PER_TOKEN,
        }
        return user, budget

    # ── the round ───────────────────────────────────────────────────────

    def expand(self, *, parents, proposals, evidence, round_id, allocate_dir):
        return self._run(parents=parents, proposals=proposals, evidence=evidence,
                         round_label=str(round_id), round_id=round_id,
                         allocate_dir=allocate_dir, max_candidates=self.max_candidates,
                         mode="round")

    def final_merge(self, survivors: Sequence[CandidateState], *, round_id, allocate_dir):
        """Generate one merged candidate from all final beam survivors."""
        material = [
            RawProposal(proposal_id=c.candidate_id, optimizer_name=c.optimizer_name,
                        round_id=round_id, skill=c.skill, parent_id=c.candidate_id,
                        evidence_subset_id="beam_T",
                        provenance={"subset_task_ids": [], "beam_T_survivor": True})
            for c in survivors
        ]
        out = self._run(parents=(), proposals=material, evidence=(),
                        round_label=f"{round_id} (final merge of the Beam_T survivors)",
                        round_id=round_id, allocate_dir=allocate_dir, max_candidates=1,
                        mode="final_merge")
        return out[0] if out else None

    def _run(self, *, parents, proposals, evidence, round_label, round_id, allocate_dir,
             max_candidates: int, mode: str):
        unique = _unique_in_order(proposals)
        n_op = min(max_candidates, len(unique))
        report: dict[str, Any] = {
            "operator": self.name, "mode": mode, "round_id": round_id,
            "n_parents": len(parents), "n_proposals": len(proposals),
            "n_unique_proposals": len(unique), "n_op_target": n_op,
            "roles": list(ROLES[:n_op]), "llm_calls": 0, "fallback": False,
            "dropped_parent_identical": 0, "dropped_duplicate": 0, "dropped_invalid": 0,
        }
        if n_op == 0:
            self.last_report = report
            return []

        resources = self.require_resources("the recombination operator")
        work_dir = allocate_dir("recombination" if mode == "round" else "final_merge")
        user, budget = self.build_prompt(parents=parents, proposals=unique, evidence=evidence,
                                         round_id=round_label, n_op=n_op)
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
        report["n_emitted"] = len(out)
        self.last_report = report
        with open(os.path.join(work_dir, "operator_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        for op in out:
            op.provenance["operator_report"] = {k: v for k, v in report.items() if k != "parse_errors"}
        return out

    # ── outputs ─────────────────────────────────────────────────────────

    def _lineage_from(self, sources: list[RawProposal], parent: CandidateState | None) -> dict:
        for q in sources:
            found = {k: q.provenance.get(k) for k in _LINEAGE_KEYS if q.provenance.get(k)}
            if found:
                return found
        if parent is not None:
            for src in (parent.metadata, parent.operator_provenance, parent.proposal_provenance):
                found = {k: (src or {}).get(k) for k in _LINEAGE_KEYS if (src or {}).get(k)}
                if found:
                    return found
        return {}

    def _post_process(self, items: list, unique: list[RawProposal], parents, n_op: int,
                      report: dict) -> list[OperatorProposal]:
        by_id = {q.proposal_id: q for q in unique}
        parent_hashes = {p.skill_hash: p.candidate_id for p in parents}
        raw_hashes = {q.skill_hash: q.proposal_id for q in unique}
        parent_by_id = {p.candidate_id: p for p in parents}
        seen_hashes: set = set()
        out: list[OperatorProposal] = []
        for index, item in enumerate(items):
            if len(out) >= n_op:
                break
            if not isinstance(item, dict) or not str(item.get("skill") or "").strip():
                report["dropped_invalid"] += 1
                continue
            skill = str(item["skill"])
            role = str(item.get("role") or "")
            if role not in ROLES:
                role = ROLES[min(index, len(ROLES) - 1)]
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
            # no declared source: anchored to the first parent, exactly as the
            # controller anchors such a candidate
            anchor = (parent_by_id.get(source_parents[0]) if source_parents
                      else (parents[0] if parents else None))
            provenance = {
                "role": role,
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

    def _fallback(self, unique: list[RawProposal], n_op: int, parents, report: dict) -> list[OperatorProposal]:
        """No usable reply: the first N_OP unique RawProposals, unchanged."""
        report["fallback"] = True
        parent_by_id = {p.candidate_id: p for p in parents}
        out = []
        for q in unique[:n_op]:
            out.append(OperatorProposal(
                skill=q.skill,
                provenance={
                    "role": "fallback_pass_through", "rationale": "operator reply unusable",
                    "preserved_raw_proposal": q.proposal_id, "declared_source_proposal_ids": [],
                    "fallback": True, "operator_fallback_noop": True,
                    **self._lineage_from([q], parent_by_id.get(q.parent_id)),
                },
                source_proposal_ids=(q.proposal_id,),
                source_parent_ids=(q.parent_id,),
            ))
        return out


__all__ = ["ROLES", "SYSTEM_PROMPT", "GlobalDiversityRecombinationOperator", "OperatorOutputError"]
