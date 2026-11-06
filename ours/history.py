"""Maintain branch issues, repair attempts, and regeneration history."""
from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

#: The status vocabulary the issue tracker maintains for one issue.
STATUS_NEW = "new"
STATUS_RECURRING = "recurring"
STATUS_RESOLVED = "resolved"
ISSUE_STATUSES = (STATUS_NEW, STATUS_RECURRING, STATUS_RESOLVED)

#: The evolution operators an attempt can come from. ``raw_proposal`` is the
#: native one-step proposal; the other three are the operator family's roles.
KIND_RAW_PROPOSAL = "raw_proposal"
KIND_MUTATION = "mutation"
KIND_CROSSOVER = "crossover"
KIND_REGENERATION = "regeneration"
ATTEMPT_KINDS = (KIND_RAW_PROPOSAL, KIND_MUTATION, KIND_CROSSOVER, KIND_REGENERATION)

#: Where a branch's history lives inside ``CandidateState.metadata``.
METADATA_KEY = "history"


# ── construction / access ────────────────────────────────────────────────────


def new_history() -> dict:
    """An empty, well-formed history state."""
    return {"issues": [], "attempts": [], "restarts": 0}


def is_history(value: Any) -> bool:
    return isinstance(value, dict) and "issues" in value and "attempts" in value


def get(metadata: dict) -> dict:
    """The branch history stored in ``metadata``, creating an empty one in place
    if absent. Callers that must not mutate ``metadata`` use :func:`peek`."""
    history = metadata.get(METADATA_KEY)
    if not is_history(history):
        history = new_history()
        metadata[METADATA_KEY] = history
    return history


def peek(metadata: dict | None) -> dict:
    """The branch history in ``metadata`` if present, else a fresh empty one.
    Never mutates ``metadata`` (safe on a parent one must not touch)."""
    history = (metadata or {}).get(METADATA_KEY)
    return history if is_history(history) else new_history()


def deepcopy_history(history: dict | None) -> dict:
    """A deep, independent copy of a history state (or an empty one)."""
    if not is_history(history):
        return new_history()
    return copy.deepcopy(history)


def carry_forward_metadata(metadata: dict | None) -> dict:
    """A copy of ``metadata`` whose ``history`` entry is DEEP-copied.

    :meth:`ours.types.CandidateState.carried_forward` copies ``metadata``
    shallowly, which is correct for every other key but would make two sibling
    branches (both carried forward from the same parent object, or an operator
    candidate anchored to a parent) share one mutable history dict. Deep-copying
    the history alone keeps the rest of the shallow-copy behaviour unchanged."""
    out = dict(metadata or {})
    if METADATA_KEY in out:
        out[METADATA_KEY] = deepcopy_history(out[METADATA_KEY])
    return out


# ── issues ───────────────────────────────────────────────────────────────────


def set_issues(history: dict, issues: Sequence[dict]) -> None:
    """Replace the branch's issue table (the tracker owns its content)."""
    history["issues"] = [dict(i) for i in issues]


def issues(history: dict | None) -> list[dict]:
    return list((history or {}).get("issues") or [])


def issue_ids(history: dict | None) -> list[str]:
    return [str(i.get("issue_id")) for i in issues(history) if i.get("issue_id") is not None]


def unresolved_issues(history: dict | None) -> list[dict]:
    return [i for i in issues(history) if i.get("status") != STATUS_RESOLVED]


def relevant_issues(history: dict | None, task_ids: Iterable[str]) -> list[dict]:
    """The branch's issues whose tasks intersect ``task_ids`` (a subset)."""
    wanted = {str(t) for t in task_ids}
    if not wanted:
        return []
    out = []
    for issue in issues(history):
        its = {str(t) for t in (issue.get("task_ids") or [])}
        if its & wanted:
            out.append(issue)
    return out


def attempts_for_issue(history: dict | None, issue_id: str) -> list[dict]:
    """Every recorded attempt that targeted ``issue_id``."""
    return [a for a in (history or {}).get("attempts") or []
            if str(issue_id) in {str(t) for t in (a.get("target_issue_ids") or [])}]


def task_issue_features(history: dict | None) -> dict[str, dict]:
    """Map tasks to covering issue IDs, issue status, and prior attempt counts."""
    out: dict[str, dict] = {}
    for issue in issues(history):
        iid = str(issue.get("issue_id"))
        status = issue.get("status")
        n_attempts = len(attempts_for_issue(history, iid))
        for tid in issue.get("task_ids") or []:
            entry = out.setdefault(str(tid), {"issue_ids": [], "_statuses": [],
                                              "issue_attempt_count": 0})
            entry["issue_ids"].append(iid)
            entry["_statuses"].append(status)
            entry["issue_attempt_count"] += n_attempts
    for entry in out.values():
        statuses = entry.pop("_statuses")
        entry["issue_status"] = (STATUS_RECURRING if STATUS_RECURRING in statuses
                                 else STATUS_NEW if STATUS_NEW in statuses else STATUS_RESOLVED)
    return out


# ── attempts ─────────────────────────────────────────────────────────────────


def record_attempt(history: dict, *, round: int, kind: str, target_issue_ids: Sequence[str] = (),
                   summary: str = "", source_id: str = "") -> dict:
    """Append one attempt to the branch's log and return it.

    ``kind`` is one of :data:`ATTEMPT_KINDS` (``raw_proposal`` for the native
    one-step proposal, or the operator role that produced the candidate)."""
    attempt = {
        "round": int(round),
        "kind": str(kind),
        "target_issue_ids": [str(t) for t in target_issue_ids],
        "summary": str(summary or ""),
        "source_id": str(source_id or ""),
    }
    history.setdefault("attempts", []).append(attempt)
    return attempt


# ── inheritance / merge ──────────────────────────────────────────────────────


def _merge_issue(into: dict, other: dict) -> dict:
    """Keep the more recently seen copy of an issue (by ``last_seen_round``)."""
    a = int(into.get("last_seen_round", into.get("first_seen_round", 0)) or 0)
    b = int(other.get("last_seen_round", other.get("first_seen_round", 0)) or 0)
    return dict(other) if b >= a else dict(into)


def merge(histories: Sequence[dict]) -> dict:
    """Union of several branch histories, DEEP-copied (crossover: several source
    parents). Issues are de-duplicated by ``issue_id`` keeping the more recently
    seen copy; attempts are concatenated with exact-duplicate removal; the
    restart counter is the maximum. The result shares nothing with its inputs."""
    valid = [h for h in histories if is_history(h)]
    if not valid:
        return new_history()
    if len(valid) == 1:
        return deepcopy_history(valid[0])

    merged_issues: dict[str, dict] = {}
    order: list[str] = []
    for history in valid:
        for issue in issues(history):
            iid = str(issue.get("issue_id"))
            if iid in merged_issues:
                merged_issues[iid] = _merge_issue(merged_issues[iid], issue)
            else:
                merged_issues[iid] = dict(issue)
                order.append(iid)

    seen_attempts: set = set()
    merged_attempts: list[dict] = []
    for history in valid:
        for attempt in history.get("attempts") or []:
            key = (attempt.get("round"), attempt.get("kind"), attempt.get("source_id"),
                   attempt.get("summary"), tuple(attempt.get("target_issue_ids") or []))
            if key in seen_attempts:
                continue
            seen_attempts.add(key)
            merged_attempts.append(dict(attempt))

    restarts = max(int(h.get("restarts", 0) or 0) for h in valid)
    return copy.deepcopy({
        "issues": [merged_issues[iid] for iid in order],
        "attempts": merged_attempts,
        "restarts": restarts,
    })


def inherit(source_histories: Sequence[dict]) -> dict:
    """The history an operator candidate starts from: a deep copy of its single
    source parent's history, or the merged union of several (crossover)."""
    return merge(list(source_histories))


# ── rendering ────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n[... history truncated ...]"
    if limit <= len(marker):
        return text[:limit]
    return text[: limit - len(marker)] + marker


def _render_issue_line(issue: dict) -> str:
    iid = issue.get("issue_id", "?")
    status = issue.get("status", "?")
    rounds_seen = issue.get("rounds_seen", "?")
    desc = str(issue.get("description") or "").strip()
    tasks = ", ".join(str(t) for t in (issue.get("task_ids") or [])) or "?"
    return f"- [{iid}] status={status}, seen in {rounds_seen} round(s), tasks: {tasks}\n    {desc}"


def render_for_proposer(history: dict | None, task_ids: Iterable[str], *, limit: int = 4000) -> str:
    """Attempt-aware context for the native proposer on ONE evidence subset.

    Lists the issues relevant to this subset, the attempts already tried for
    them (so the proposer does not regenerate the same fix), and a single
    instruction to try something different. Empty when there is nothing to say,
    so a branch with no history hands the native proposer exactly what it got
    before (``""``)."""
    relevant = relevant_issues(history, task_ids)
    if not relevant:
        return ""
    lines = ["This branch has already run into these issues on the related tasks:"]
    for issue in relevant:
        lines.append(_render_issue_line(issue))
        tried = attempts_for_issue(history, str(issue.get("issue_id")))
        for attempt in tried:
            summary = str(attempt.get("summary") or "").strip() or "(no summary)"
            lines.append(f"    already tried [{attempt.get('kind')}]: {summary}")
    lines.append(
        "Do NOT regenerate a fix that was already tried above for these issues; "
        "propose a genuinely different change."
    )
    return _truncate("\n".join(lines), limit)


def render_for_operator(history: dict | None, *, branch_id: str = "", limit: int = 4000) -> str:
    """One branch's history for the operator's ``BRANCH HISTORIES`` section."""
    iss = issues(history)
    restarts = int((history or {}).get("restarts", 0) or 0)
    header = f"branch {branch_id}".strip() or "branch"
    if not iss and not (history or {}).get("attempts"):
        return _truncate(f"--- {header} --- (no recorded history)", limit)
    lines = [f"--- {header} (restarts: {restarts}) ---", "issues:"]
    for issue in iss:
        lines.append(_render_issue_line(issue))
    attempts = (history or {}).get("attempts") or []
    if attempts:
        lines.append("attempts already tried:")
        for attempt in attempts:
            targets = ", ".join(str(t) for t in (attempt.get("target_issue_ids") or [])) or "-"
            summary = str(attempt.get("summary") or "").strip() or "(no summary)"
            lines.append(f"    r{attempt.get('round')} [{attempt.get('kind')}] -> {targets}: {summary}")
    return _truncate("\n".join(lines), limit)


__all__ = [
    "STATUS_NEW", "STATUS_RECURRING", "STATUS_RESOLVED", "ISSUE_STATUSES",
    "KIND_RAW_PROPOSAL", "KIND_MUTATION", "KIND_CROSSOVER", "KIND_REGENERATION", "ATTEMPT_KINDS",
    "METADATA_KEY", "new_history", "is_history", "get", "peek", "deepcopy_history",
    "carry_forward_metadata", "set_issues", "issues", "issue_ids", "unresolved_issues",
    "relevant_issues", "attempts_for_issue", "task_issue_features", "record_attempt",
    "merge", "inherit", "render_for_proposer", "render_for_operator",
]
