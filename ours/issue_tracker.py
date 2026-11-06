"""Update semantic issue tables from branch evidence, with a deterministic fallback."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

from ours import history
from ours.operator_recombination import OperatorOutputError, _approx_tokens, _extract_json, _head_tail
from ours.registry import register
from ours.resources import ROLE_ISSUE_TRACKER, RunBound

SYSTEM_PROMPT = """You are the issue tracker of a skill-optimization search.

A "branch" is one candidate skill being optimized. Each round it is run on some training tasks and you are shown, per task, whether it was solved (when available), the task type, an error signal and a short rendering of what happened. You maintain a SMALL table of the distinct underlying ISSUES this branch keeps running into -- recurring failure modes, not one row per task.

You are given the branch's issue table from the PREVIOUS round and this round's per-task evidence. Return the UPDATED table. Rules:
- If a problem in the new evidence is the SAME underlying issue as one in the prior table, REUSE its issue_id and set status to "recurring"; increment rounds_seen and set last_seen_round to this round.
- If a problem is genuinely NEW, add a row with an EMPTY issue_id (the system assigns a stable id) and status "new".
- If a prior issue no longer appears in this round's evidence, keep its row but set status to "resolved".
- Keep the table small: merge tasks that share one underlying cause into one issue.

Return ONLY a JSON object of this exact shape, nothing else:
{"issues": [{"issue_id": "<id from the prior table, or empty for a new issue>", "description": "<one sentence>", "status": "new|recurring|resolved", "task_ids": ["<task id>", ...], "first_seen_round": <int>, "last_seen_round": <int>, "rounds_seen": <int>}]}
"""

_USER_TEMPLATE = """Round {round_id}, branch {branch_id}.

=== PRIOR ISSUE TABLE (previous round) ===
{prior}

=== THIS ROUND'S PER-TASK EVIDENCE ===
{evidence}
"""

_REPAIR_NOTE = (
    "\n\nYour previous reply could not be parsed. Return ONLY the JSON object described "
    "in the instructions, with no prose before or after it."
)

#: Fields of the per-task evidence view the tracker forwards to the model, in
#: order. ``solved`` is omitted for an optimizer (SSO) that has no such flag.
_EVIDENCE_FIELDS = ("solved", "task_type", "error_signal")


class IssueTracker(RunBound):
    """LLM-maintained per-branch issue table, with a deterministic fallback."""

    name = "abstract"
    is_placeholder = True

    def __init__(self, *, char_budget: int = 60_000, retries: int = 1,
                 max_tokens: int | None = None, error_signal_key_chars: int = 80) -> None:
        self.char_budget = int(char_budget)
        self.retries = max(0, int(retries))
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.error_signal_key_chars = int(error_signal_key_chars)
        self.last_report: dict | None = None

    # ── the round ────────────────────────────────────────────────────────────

    def update(self, *, branch_id: str, prior_issues: Sequence[dict],
               evidence_features: dict[str, dict[str, Any]], round_id: int) -> tuple[list[dict], dict]:
        """Return ``(issues, report)`` for one branch this round.

        ``prior_issues`` is the branch's issue table from last round; the result
        is the updated table (never mutating ``prior_issues``)."""
        prior = [dict(i) for i in prior_issues]
        report: dict[str, Any] = {
            "tracker": self.name, "round_id": round_id, "branch_id": branch_id,
            "n_tasks": len(evidence_features), "n_prior_issues": len(prior),
            "llm_calls": 0, "fallback": False, "parse_errors": [],
        }
        resources = self.require_resources("the issue tracker")
        user, budget = self._build_prompt(branch_id=branch_id, prior=prior,
                                          evidence_features=evidence_features, round_id=round_id)
        report["budget"] = budget
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

        parsed: list | None = None
        for attempt in range(1 + self.retries):
            attempt_messages = messages if attempt == 0 else [
                messages[0], {"role": "user", "content": user + _REPAIR_NOTE}]
            kwargs = {} if self.max_tokens is None else {"max_tokens": self.max_tokens}
            message = resources.chat(attempt_messages, role=ROLE_ISSUE_TRACKER, **kwargs)
            report["llm_calls"] += 1
            finish = getattr(resources, "last_finish_reason", None)
            report.setdefault("finish_reasons", []).append(finish)
            text = getattr(message, "content", None) or ""
            try:
                if finish == "length":
                    raise OperatorOutputError("reply truncated at max_tokens (finish_reason=length)")
                payload = _extract_json(text)
                items = payload.get("issues")
                if not isinstance(items, list):
                    raise OperatorOutputError("no issues list in reply")
                parsed = items
                break
            except OperatorOutputError as exc:
                report["parse_errors"].append(f"attempt {attempt}: {exc}")

        if parsed is None:
            issues = self._fallback(prior, evidence_features, round_id)
            report["fallback"] = True
        else:
            issues = self._normalise(parsed, prior, round_id)
        self._summarise(report, issues)
        self.last_report = report
        return issues, report

    # ── prompt ───────────────────────────────────────────────────────────────

    def _evidence_records(self, evidence_features: dict[str, dict[str, Any]]) -> list[dict]:
        records = []
        for tid, view in evidence_features.items():
            record: dict[str, Any] = {"task_id": str(tid)}
            for field in _EVIDENCE_FIELDS:
                if field in view and view.get(field) not in (None, ""):
                    record[field] = view.get(field)
            record["native_text"] = str(view.get("native_text") or "")
            records.append(record)
        return records

    def _build_prompt(self, *, branch_id: str, prior: Sequence[dict],
                      evidence_features: dict, round_id: int) -> tuple[str, dict]:
        prior_text = json.dumps([_public_issue(i) for i in prior], ensure_ascii=False, indent=2) \
            if prior else "(none)"
        records = self._evidence_records(evidence_features)
        fixed = _USER_TEMPLATE.format(round_id=round_id, branch_id=branch_id,
                                      prior=prior_text, evidence="")
        remaining = self.char_budget - len(SYSTEM_PROMPT) - len(fixed)
        original_chars = sum(len(r["native_text"]) for r in records)
        per_record = None
        truncated = False
        if records and original_chars > max(remaining, 0):
            truncated = True
            per_record = max(remaining, 0) // len(records)
        kept_chars = 0
        parts = []
        for r in records:
            text = r["native_text"] if per_record is None else _head_tail(r["native_text"], per_record)
            kept_chars += len(text)
            meta = {k: r[k] for k in _EVIDENCE_FIELDS if k in r}
            header = f"--- task {r['task_id']} {json.dumps(meta, ensure_ascii=False)} ---"
            parts.append(f"{header}\n{text}" if text else f"{header}\n(omitted: no budget left)")
        user = _USER_TEMPLATE.format(round_id=round_id, branch_id=branch_id, prior=prior_text,
                                     evidence="\n\n".join(parts) if parts else "(none)")
        budget = {
            "char_budget": self.char_budget,
            "prompt_chars": len(SYSTEM_PROMPT) + len(user),
            "prompt_tokens_approx": _approx_tokens(SYSTEM_PROMPT + user),
            "evidence_records": len(records),
            "evidence_chars_original": original_chars,
            "evidence_chars_kept": kept_chars,
            "evidence_truncated": truncated,
            "per_record_char_budget": per_record,
        }
        return user, budget

    # ── outputs ──────────────────────────────────────────────────────────────

    def _normalise(self, items: list, prior: Sequence[dict], round_id: int) -> list[dict]:
        """Trust the model's rows but repair ids, statuses and counters.

        A returned issue_id is reused only if it exists in the prior table;
        anything else (a new issue, or an id the model invented) is given a
        stable code-assigned id, so the model never introduces its own ids."""
        old_by_id = {str(i.get("issue_id")): i for i in prior}
        out: list[dict] = []
        new_counter = 0
        for raw in items:
            if not isinstance(raw, dict):
                continue
            claimed = str(raw.get("issue_id") or "").strip()
            if claimed and claimed in old_by_id:
                final_id, old = claimed, old_by_id[claimed]
            else:
                final_id = f"iss-r{round_id:03d}-{new_counter:03d}"
                new_counter += 1
                old = None
            out.append(_normalise_issue(raw, final_id=final_id, old=old, round_id=round_id))
        return out

    def _fallback(self, prior: Sequence[dict], evidence_features: dict, round_id: int) -> list[dict]:
        """Deterministic table: group tasks by ``(task_type, error_signal[:N])``.

        A group's id is a stable hash of its key, so the SAME failure mode gets
        the SAME id across rounds and "recurring" is observable even without the
        model. Prior issues whose id no longer appears become ``resolved``."""
        old_by_id = {str(i.get("issue_id")): i for i in prior}
        groups: dict[str, dict] = {}
        order: list[str] = []
        for tid, view in evidence_features.items():
            solved = view.get("solved")
            error = str(view.get("error_signal") or "")
            if solved is True or (solved is None and not error.strip()):
                # solved (when known) or no error signal at all -> no issue
                continue
            task_type = str(view.get("task_type") or "")
            key = f"{task_type}|{error[: self.error_signal_key_chars]}"
            gid = "fb-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
            if gid not in groups:
                groups[gid] = {"key": key, "task_type": task_type, "error": error, "task_ids": []}
                order.append(gid)
            groups[gid]["task_ids"].append(str(tid))

        out: list[dict] = []
        for gid in order:
            g = groups[gid]
            old = old_by_id.get(gid)
            first_seen = int(old["first_seen_round"]) if old else round_id
            rounds_seen = int(old.get("rounds_seen", 0)) + 1 if old else 1
            out.append({
                "issue_id": gid,
                "description": (f"{g['task_type']}: {g['error'][:120]}" if g["error"]
                                else f"{g['task_type']} failures"),
                "status": history.STATUS_RECURRING if old is not None else history.STATUS_NEW,
                "task_ids": g["task_ids"],
                "first_seen_round": first_seen,
                "last_seen_round": round_id,
                "rounds_seen": rounds_seen,
            })
        # prior issues that disappeared this round -> resolved
        present = set(groups)
        for iid, old in old_by_id.items():
            if iid in present:
                continue
            out.append({**{k: old.get(k) for k in _ISSUE_KEYS},
                        "issue_id": iid, "status": history.STATUS_RESOLVED,
                        "last_seen_round": int(old.get("last_seen_round", round_id))})
        return out

    @staticmethod
    def _summarise(report: dict, issues: Sequence[dict]) -> None:
        report["n_issues"] = len(issues)
        report["n_new"] = sum(1 for i in issues if i.get("status") == history.STATUS_NEW)
        report["n_recurring"] = sum(1 for i in issues if i.get("status") == history.STATUS_RECURRING)
        report["n_resolved"] = sum(1 for i in issues if i.get("status") == history.STATUS_RESOLVED)


@register("tracker", "llm_issue_tracker")
class LLMIssueTracker(IssueTracker):
    """Maintain semantic issues with an LLM and a deterministic parsing fallback."""

    name = "llm_issue_tracker"
    is_placeholder = False


_ISSUE_KEYS = ("issue_id", "description", "status", "task_ids",
               "first_seen_round", "last_seen_round", "rounds_seen")


def _public_issue(issue: dict) -> dict:
    return {k: issue.get(k) for k in _ISSUE_KEYS}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_issue(raw: dict, *, final_id: str, old: dict | None, round_id: int) -> dict:
    status = str(raw.get("status") or "").strip()
    if status not in history.ISSUE_STATUSES:
        status = history.STATUS_RECURRING if old is not None else history.STATUS_NEW
    first_seen = _coerce_int(raw.get("first_seen_round"),
                             int(old["first_seen_round"]) if old else round_id)
    last_seen = _coerce_int(raw.get("last_seen_round"),
                            round_id if status != history.STATUS_RESOLVED
                            else (int(old["last_seen_round"]) if old else round_id))
    default_rounds = (int(old.get("rounds_seen", 1)) + (1 if status != history.STATUS_RESOLVED else 0)) \
        if old else 1
    rounds_seen = _coerce_int(raw.get("rounds_seen"), default_rounds)
    return {
        "issue_id": final_id,
        "description": str(raw.get("description") or "").strip(),
        "status": status,
        "task_ids": [str(t) for t in (raw.get("task_ids") or [])],
        "first_seen_round": first_seen,
        "last_seen_round": last_seen,
        "rounds_seen": rounds_seen,
    }


__all__ = ["IssueTracker", "LLMIssueTracker", "SYSTEM_PROMPT"]
