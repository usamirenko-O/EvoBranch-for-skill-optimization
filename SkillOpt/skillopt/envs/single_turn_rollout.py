"""Run single-turn benchmark tasks and aggregate their scores and token usage."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from skillopt.model import chat_target, is_target_exec_backend


def _safe_id(item_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_.:" else "_" for ch in str(item_id))


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    build_system: Callable[[str], str],
    build_user: Callable[[dict], str],
    evaluate: Callable[[str, dict], dict],
    task_type_of: Callable[[dict], str],
    exec_timeout: int | None = 300,
    max_completion_tokens: int = 16384,
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "task_type": task_type_of(item),
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }
    try:
        if is_target_exec_backend():
            raise NotImplementedError(
                "single-turn cheap benchmarks only support chat backends "
                "(the main experiment routes through openai_compatible)"
            )
        pred_dir = os.path.join(out_root, "predictions", _safe_id(item_id))
        os.makedirs(pred_dir, exist_ok=True)
        llm_timeout = int(exec_timeout) if exec_timeout and int(exec_timeout) > 0 else None

        system = build_system(skill_content)
        user = build_user(item)
        resp_text, usage_info = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            timeout=llm_timeout,
        )
        result["response"] = resp_text
        result["agent_ok"] = True
        result["n_turns"] = 1
        result["finish_reason"] = usage_info.get("finish_reason")

        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)

        verdict = evaluate(resp_text, item)
        score = float(verdict.get("score", 0.0))
        result["hard"] = int(score >= 1.0)
        result["soft"] = score
        result["predicted_answer"] = str(verdict.get("predicted", "") or "")
        result["verdict"] = {k: v for k, v in verdict.items() if k not in ("predicted",)}
        if not result["hard"]:
            result["fail_reason"] = str(verdict.get("fail_reason") or "score=0")

        conversation = [
            {"type": "message", "turn": 1, "content": resp_text},
            {"role": "system", "content": "[EVALUATION RESULT]\n" + str(verdict.get("detail") or verdict)},
        ]
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {type(e).__name__}: {e}"
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    build_system: Callable[[str], str],
    build_user: Callable[[dict], str],
    evaluate: Callable[[str, dict], dict],
    task_type_of: Callable[[dict], str],
    exec_timeout: int | None = 300,
    workers: int = 16,
    max_completion_tokens: int = 16384,
    task_timeout: int | None = 600,
) -> list[dict]:
    exec_timeout_value = int(exec_timeout) if exec_timeout and int(exec_timeout) > 0 else 0
    task_timeout_value = int(task_timeout) if task_timeout and int(task_timeout) > 0 else 0
    if exec_timeout_value <= 0 or task_timeout_value <= 0:
        task_timeout = None
    else:
        task_timeout = max(task_timeout_value, exec_timeout_value + 60)

    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:  # noqa: BLE001
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        return existing

    total = len(existing) + len(pending)
    completed = len(existing)
    correct_count = sum(1 for r in existing if r.get("hard", 0))
    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)
    results = list(existing)
    started_at: dict[str, float] = {}

    def _run_one(it: dict) -> dict:
        started_at[str(it["id"])] = time.time()
        return process_one(
            it,
            out_root,
            skill_content,
            build_system=build_system,
            build_user=build_user,
            evaluate=evaluate,
            task_type_of=task_type_of,
            exec_timeout=exec_timeout,
            max_completion_tokens=max_completion_tokens,
        )

    def _failed_result(it: dict, reason: str) -> dict:
        return {
            "id": str(it["id"]),
            "task_type": task_type_of(it),
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": "",
            "response": "",
            "fail_reason": reason,
            "agent_ok": False,
            "n_turns": 0,
        }

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if task_timeout is not None
                    and str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa: BLE001
                        res = _failed_result(futs[fut], f"error: {type(e).__name__}: {e}")
                    results.append(res)
                    completed += 1
                    correct_count += 1 if res.get("hard", 0) else 0
                    acc = correct_count / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} (acc={acc:.3f}) id={res['id']} hard={res.get('hard', '?')}",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    res = _failed_result(futs[fut], f"task-timeout-{task_timeout}s")
                    results.append(res)
                    completed += 1
                    print(f"    [rollout] {completed}/{total} id={res['id']} TIMEOUT", flush=True)
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return results
