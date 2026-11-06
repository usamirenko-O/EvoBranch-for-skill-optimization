"""Apply SkillOpt slow updates and meta-skill updates to selected branches."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path
from ours.types import CandidateState, Origin, skill_fingerprint

ensure_on_sys_path(SKILLOPT_ROOT)

import skillopt.optimizer.meta_skill as _meta_mod  # noqa: E402
import skillopt.optimizer.slow_update as _slow_mod  # noqa: E402
from skillopt.optimizer.meta_skill import format_meta_skill_context, run_meta_skill  # noqa: E402
from skillopt.optimizer.slow_update import (  # noqa: E402
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    _strip_all_slow_update_fields,
    extract_slow_update_field,
    has_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    run_slow_update,
    save_comparison_pairs,
)
from skillopt.utils import compute_score  # noqa: E402

#: Where a branch's native memory lives inside ``CandidateState.metadata``.
NATIVE_STATE_KEY = "native_state"


PARSE_RETRIES = 1


class _RecordedCall:
    """Context manager: route ``module.chat_optimizer`` through a recorder that
    appends each (system, user, reply) to ``log_path`` (jsonl)."""

    def __init__(self, module, log_path: str) -> None:
        self.module, self.log_path, self.calls = module, log_path, []

    def __enter__(self):
        self._orig = self.module.chat_optimizer

        def recorded(system, user, **kw):
            reply, meta = self._orig(system, user, **kw)
            rec = {"system": system, "user": user, "reply": reply if isinstance(reply, str) else str(reply)}
            self.calls.append(rec)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return reply, meta

        self.module.chat_optimizer = recorded
        return self

    def __exit__(self, *exc):
        self.module.chat_optimizer = self._orig
        return False


def _call_with_retry(fn, module, log_path: str, *args, **kwargs):
    """Call a native epoch function; on an empty result re-issue it up to
    :data:`PARSE_RETRIES` times. Returns ``(result, n_calls, raw_replies)``."""
    replies = []
    result = None
    for attempt in range(1 + PARSE_RETRIES):
        with _RecordedCall(module, log_path) as rec:
            result = fn(*args, **kwargs)
        replies.extend(c["reply"] for c in rec.calls)
        if result:
            break
    return result, len(replies), replies


def strip_protected_block(skill: str) -> str:
    return _strip_all_slow_update_fields(skill)


def inherit_protected_block(skill: str, anchor_skill: str) -> str:
    """``skill`` with exactly the protected block of ``anchor_skill`` (none if
    the anchor has none). Anything the model wrote between markers is removed."""
    body = strip_protected_block(skill)
    if not has_slow_update_field(anchor_skill):
        return body
    content = extract_slow_update_field(anchor_skill)
    return replace_slow_update_field(body, content) if content else inject_empty_slow_update_field(body)


def branch_meta_skill(candidate: CandidateState) -> str:
    state = (candidate.metadata or {}).get(NATIVE_STATE_KEY) or {}
    return str(state.get("meta_skill") or "")


@dataclass
class NativeEpochProtocol:
    """See the module docstring. Built by the SkillOpt+OURS driver from the
    live native runtime; bound to the adapter."""

    env_adapter: Any
    dataloader: Any
    out_root: str
    seed: int
    train_pool_ids: frozenset
    use_slow_update: bool = True
    use_meta_skill: bool = True
    samples: int = 20
    pair_policy: str = "mixed"
    #: (round_id, anchor skill hash) -> (rollout_dir, results): the previous
    #: generation's rollouts are shared by siblings of one anchor in a round.
    _prev_cache: dict = field(default_factory=dict)
    reports: list = field(default_factory=list)

    @classmethod
    def from_runtime(cls, runtime, *, train_pool_ids) -> "NativeEpochProtocol":
        """Built from the live native runtime with SkillOpt's OWN config values
        (``use_slow_update`` / ``use_meta_skill`` / ``slow_update_samples`` /
        ``longitudinal_pair_policy``, normalised by the trainer's own rule)."""
        from skillopt.engine.trainer import _normalise_longitudinal_pair_policy

        cfg = runtime.cfg
        return cls(
            env_adapter=runtime.env_adapter, dataloader=runtime.dataloader,
            out_root=runtime.out_root, seed=int(runtime.seed), train_pool_ids=frozenset(train_pool_ids),
            use_slow_update=bool(cfg.get("use_slow_update", False)),
            use_meta_skill=bool(cfg.get("use_meta_skill", False)),
            samples=int(cfg.get("slow_update_samples", 20) or 20),
            pair_policy=_normalise_longitudinal_pair_policy(cfg.get("longitudinal_pair_policy")),
        )

    # ── description for the artifact ─────────────────────────────────────

    def describe(self) -> dict:
        return {
            "enabled": True,
            "decision": "Update slow guidance and meta skills for new survivors against their anchor parents. Round 0 inserts a placeholder; the final round skips epoch updates.",
            "use_slow_update": self.use_slow_update, "use_meta_skill": self.use_meta_skill,
            "slow_update_samples": self.samples, "longitudinal_pair_policy": self.pair_policy,
            "sample_seed_rule": "seed + (round_id + 1) * 2000 (native: seed + epoch * 2000)",
            "score_policy": "survivor keeps its fixed-VAL score; the block is force-accepted without "
                            "re-validation (native slow_update_gate_with_selection=false)",
            "inheritance": "an operator candidate inherits its anchor branch's protected block "
                           "verbatim and its meta-skill memory; the operator prompt declares the "
                           "block protected",
            "unchanged_survivors": "not updated (no new generation to compare)",
        }

    # ── inheritance (operator candidates, final merge) ───────────────────

    def adopt(self, skill: str, anchor: CandidateState) -> tuple[str, dict]:
        state = dict((anchor.metadata or {}).get(NATIVE_STATE_KEY) or {})
        return inherit_protected_block(skill, anchor.skill), {NATIVE_STATE_KEY: state}

    def meta_context(self, parent: CandidateState) -> str:
        return format_meta_skill_context(branch_meta_skill(parent))

    # ── epoch end ────────────────────────────────────────────────────────

    def finish_round(self, survivors: Sequence[CandidateState], *, parents_by_id: dict,
                     round_id: int, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        report: dict[str, Any] = {
            "round_id": round_id, "native_epoch": round_id + 1,
            "action": "inject_placeholder" if round_id == 0 else "epoch_update",
            "survivors": [],
        }
        new = [c for c in survivors if c.origin is Origin.OPERATOR and c.round_id == round_id]
        for c in survivors:
            entry: dict[str, Any] = {"candidate_id": c.candidate_id, "origin": c.origin.value,
                                     "new_this_round": c in new, "updated": False,
                                     "skill_hash_before": c.skill_hash}
            if c in new:
                if round_id == 0:
                    self._set_skill(c, inject_empty_slow_update_field(c.skill))
                    entry["updated"] = c.skill_hash != entry["skill_hash_before"]
                else:
                    anchor = parents_by_id.get(c.parent_id)
                    if anchor is None:
                        entry["skipped"] = f"anchor {c.parent_id} not among this round's parents"
                    else:
                        entry.update(self._epoch_update(c, anchor, round_id=round_id,
                                                        out_dir=os.path.join(out_dir, c.candidate_id)))
            entry["skill_hash_after"] = c.skill_hash
            entry["score_kept"] = float(c.score.value) if c.score else None
            report["survivors"].append(entry)
        with open(os.path.join(out_dir, "native_epoch_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        self.reports.append(report)
        return report

    @staticmethod
    def _set_skill(candidate: CandidateState, skill: str) -> None:
        candidate.skill = skill
        candidate.skill_hash = skill_fingerprint(skill)

    def _sample(self, round_id: int):
        seed = self.seed + (round_id + 1) * 2000
        batch = self.dataloader.build_train_batch(batch_size=self.samples, seed=seed, out_root=self.out_root)
        env = self.env_adapter.build_env_from_batch(batch, out_root=self.out_root)
        items = list(env) if hasattr(env, "__iter__") else env
        ids = [str(i.get("id")) if isinstance(i, dict) else str(i) for i in items]
        outside = sorted({i for i in ids if i not in self.train_pool_ids})
        if outside:
            raise RuntimeError(f"slow-update sample names ids outside the TRAIN pool: {outside[:5]}")
        return env, items, ids, seed

    def _epoch_update(self, survivor: CandidateState, anchor: CandidateState, *, round_id: int,
                      out_dir: str) -> dict:
        from skillopt.engine.trainer import _build_longitudinal_pairs

        os.makedirs(out_dir, exist_ok=True)
        env, items, ids, seed = self._sample(round_id)
        entry: dict[str, Any] = {"anchor_id": anchor.candidate_id, "sample_seed": seed,
                                 "sample_task_ids": ids, "n_samples": len(items)}
        # previous generation (the anchor), shared by siblings of one anchor
        key = (round_id, anchor.skill_hash)
        if key not in self._prev_cache:
            prev_dir = os.path.join(out_dir, "rollout_prev")
            results_prev = self.env_adapter.rollout(env, anchor.skill, prev_dir)
            self._prev_cache[key] = (prev_dir, results_prev)
            entry["prev_rollout"] = "measured"
        else:
            prev_dir, results_prev = self._prev_cache[key]
            entry["prev_rollout"] = f"reused from {prev_dir}"
        curr_dir = os.path.join(out_dir, "rollout_curr")
        pre_update_skill = survivor.skill
        results_curr = self.env_adapter.rollout(env, pre_update_skill, curr_dir)
        entry["prev_hard"], _ = compute_score(results_prev)
        entry["curr_hard"], _ = compute_score(results_curr)

        pairs, all_pairs = _build_longitudinal_pairs(
            adapter=self.env_adapter, dataloader=self.dataloader,
            prev_skill=anchor.skill, curr_skill=pre_update_skill,
            initial_items=items, initial_prev_results=results_prev, initial_curr_results=results_curr,
            prev_rollout_dir=prev_dir, curr_rollout_dir=curr_dir,
            policy=self.pair_policy, target_n=self.samples, seed=seed, out_root=self.out_root,
        )
        save_comparison_pairs(pairs, os.path.join(out_dir, "comparison_pairs.json"))
        entry["pair_categories"] = {
            k: sum(1 for p in pairs if p.get("category") == k)
            for k in ("regressed", "improved", "persistent_fail", "stable_success")
        }

        # slow update -> protected block on the survivor
        if self.use_slow_update:
            result, n_calls, _ = _call_with_retry(
                run_slow_update, _slow_mod, os.path.join(out_dir, "slow_update_calls.jsonl"),
                pre_update_skill, results_prev, results_curr, items,
                prev_skill=anchor.skill,
                prev_slow_update_content=extract_slow_update_field(pre_update_skill),
                prev_rollout_dir=prev_dir, curr_rollout_dir=curr_dir, comparison_pairs=pairs,
            )
            content = (result or {}).get("slow_update_content") or ""
            if content.strip():
                self._set_skill(survivor, replace_slow_update_field(pre_update_skill, content))
                entry["slow_update"] = {"action": "force_accept", "chars": len(content), "llm_calls": n_calls,
                                        "reasoning": str((result or {}).get("reasoning", ""))[:2000]}
                entry["updated"] = True
            else:
                entry["slow_update"] = {"action": "no_content", "llm_calls": n_calls,
                                        "raw_replies": "slow_update_calls.jsonl"}
            with open(os.path.join(out_dir, "slow_result.json"), "w", encoding="utf-8") as f:
                json.dump(result or {"action": "no_content"}, f, indent=2, ensure_ascii=False, default=str)

        # meta skill -> branch memory (native compares the pre-slow-update skill)
        state = dict((survivor.metadata or {}).get(NATIVE_STATE_KEY) or {})
        if self.use_meta_skill:
            meta, n_calls, _ = _call_with_retry(
                run_meta_skill, _meta_mod, os.path.join(out_dir, "meta_skill_calls.jsonl"),
                anchor.skill, pre_update_skill, pairs, prev_meta_skill_content=branch_meta_skill(anchor))
            meta_content = (meta or {}).get("meta_skill_content") or ""
            if meta_content.strip():
                state["meta_skill"] = meta_content
                state["meta_skill_round"] = round_id
                entry["meta_skill"] = {"action": "updated", "chars": len(meta_content), "llm_calls": n_calls}
            else:
                entry["meta_skill"] = {"action": "no_content", "llm_calls": n_calls,
                                       "raw_replies": "meta_skill_calls.jsonl"}
            with open(os.path.join(out_dir, "meta_skill_result.json"), "w", encoding="utf-8") as f:
                json.dump(meta or {"action": "no_content"}, f, indent=2, ensure_ascii=False, default=str)
        survivor.metadata[NATIVE_STATE_KEY] = state
        with open(os.path.join(out_dir, "skill_after_epoch.md"), "w", encoding="utf-8") as f:
            f.write(survivor.skill)
        return entry


__all__ = ["NATIVE_STATE_KEY", "NativeEpochProtocol", "SLOW_UPDATE_START", "SLOW_UPDATE_END",
           "branch_meta_skill", "inherit_protected_block", "strip_protected_block"]
