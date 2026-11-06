"""Load LiveCodeBench tasks, create difficulty-stratified splits, and grade code."""
from __future__ import annotations

from collections import Counter

from integration.benchmarks.manifest import SPLIT_ROLES, SplitManifest, source_inputs
from integration.benchmarks.paths import LIVECODEBENCH_ROOT
from integration.benchmarks.single_turn_base import (
    SPLIT_SEED,
    SPLIT_SIZES,
    SingleTurnBenchmarkAdapter,
    entries_for,
    stratified_split,
)
from skillopt.envs.livecodebench.dataloader import load_items
from skillopt.envs.livecodebench.evaluator import evaluate
from skillopt.envs.livecodebench.rollout import build_user

MANIFEST_ID = "livecodebench_v5v6_stratified_177_v2"
METRIC_NAME = "livecodebench_pass_at_1"
SCORER_REF = "skillopt.envs.livecodebench.evaluator:evaluate"
DATA_FILES = ("test5.jsonl", "test6.jsonl")


def build_manifest() -> SplitManifest:
    items = load_items(str(LIVECODEBENCH_ROOT))
    by_id = {i["id"]: i for i in items}
    by_difficulty: dict = {}
    for i in items:
        by_difficulty.setdefault(i["difficulty"], []).append(i["id"])
    splits, quotas = stratified_split(by_difficulty)
    pop_counts = {d: len(v) for d, v in sorted(by_difficulty.items())}

    def extra(task_id: str) -> dict:
        it = by_id[task_id]
        return {"difficulty": it["difficulty"], "platform": it["platform"],
                "contest_date": it["contest_date"], "release_file": it["source_file"].rsplit("/", 1)[-1]}

    return SplitManifest(
        benchmark="livecodebench",
        manifest_id=MANIFEST_ID,
        version="1",
        source={
            "upstream_dataset": "livecodebench/code_generation_lite",
            "upstream_files": list(DATA_FILES),
            "materialized_from": str(LIVECODEBENCH_ROOT),
            "inputs": source_inputs([LIVECODEBENCH_ROOT / f for f in DATA_FILES]),
            "official_scorer": {
                "ref": SCORER_REF,
                "upstream": "LiveCodeBench lcb_runner/evaluation/testing_util.py, vendored via "
                            "SkillGen/benchmarks/livecodebench_adapter.py into "
                            "SkillOpt/skillopt/envs/livecodebench/lcb_grader.py",
                "note": "pass@1: a problem counts only if every public and private test passes",
            },
        },
        role_map={
            "train": f"seeded stratified sample ({SPLIT_SIZES['train']})",
            "val": f"seeded stratified sample ({SPLIT_SIZES['val']})",
            "deployment": f"seeded stratified sample ({SPLIT_SIZES['deployment']})",
            "native_splits": "none -- LiveCodeBench publishes release increments, not train/test",
        },
        sampling={
            "method": "seeded_stratified_by_difficulty_largest_remainder",
            "seed": SPLIT_SEED,
            "population": len(items),
            "population_by_difficulty": pop_counts,
            "quota_by_role": quotas,
            "sizes": dict(SPLIT_SIZES),
        },
        splits={role: entries_for(splits[role], extra) for role in SPLIT_ROLES},
        notes=[
            "Population = release v5 + v6 problems present locally (test5.jsonl, test6.jsonl).",
            "Each role keeps the population's easy/medium/hard proportion (largest-remainder rounding).",
            "Task text is SkillGen's LiveCodeBench prompt wording; the solver returns one Python solution.",
        ],
    ).with_hash()


class LiveCodeBenchAdapter(SingleTurnBenchmarkAdapter):
    name = "livecodebench"
    metric_name = METRIC_NAME
    scorer_ref = SCORER_REF
    manifest_id = MANIFEST_ID
    data_path = str(LIVECODEBENCH_ROOT)
    _load_items = staticmethod(load_items)
    _build_user = staticmethod(build_user)
    _evaluate = staticmethod(evaluate)
    payload_meta_keys = ("difficulty", "platform", "question_title")

    def resource_status(self):
        status = super().resource_status()
        if status.ready:
            counts = Counter(i["difficulty"] for i in self.items_by_id().values())
            status.details["population_by_difficulty"] = dict(sorted(counts.items()))
        return status
