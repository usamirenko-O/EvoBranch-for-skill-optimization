"""Load PubMedQA tasks, construct shared splits, and score answers."""
from __future__ import annotations

from collections import Counter

from integration.benchmarks.manifest import SPLIT_ROLES, SplitManifest, source_inputs
from integration.benchmarks.paths import PUBMEDQA_PQAL_JSON, PUBMEDQA_ROOT
from integration.benchmarks.single_turn_base import (
    SPLIT_SEED,
    SPLIT_SIZES,
    SingleTurnBenchmarkAdapter,
    entries_for,
    uniform_split,
)
from skillopt.envs.pubmedqa.dataloader import load_items
from skillopt.envs.pubmedqa.evaluator import evaluate
from skillopt.envs.pubmedqa.rollout import build_user

MANIFEST_ID = "pubmedqa_pqal_uniform_177_v2"
METRIC_NAME = "pubmedqa_accuracy_yes_no_maybe"
SCORER_REF = "skillopt.envs.pubmedqa.evaluator:evaluate"


def build_manifest() -> SplitManifest:
    items = load_items(str(PUBMEDQA_ROOT))
    by_id = {i["id"]: i for i in items}
    splits = uniform_split(list(by_id))
    label_counts = dict(sorted(Counter(i["final_decision"] for i in items).items()))

    def extra(task_id: str) -> dict:
        return {"year": by_id[task_id]["year"]}

    return SplitManifest(
        benchmark="pubmedqa",
        manifest_id=MANIFEST_ID,
        version="1",
        source={
            "upstream_dataset": "PubMedQA PQA-L (pubmedqa/pqa_labeled, ori_pqal.json)",
            "materialized_from": str(PUBMEDQA_PQAL_JSON),
            "inputs": source_inputs([PUBMEDQA_PQAL_JSON]),
            "official_scorer": {
                "ref": SCORER_REF,
                "upstream": "PubMedQA accuracy: exact match on final_decision in {yes, no, maybe}; "
                            "extraction rules copied from SkillGen/benchmarks/pubmedqa_grader.py",
            },
        },
        role_map={
            "train": f"seeded uniform sample ({SPLIT_SIZES['train']})",
            "val": f"seeded uniform sample ({SPLIT_SIZES['val']})",
            "deployment": f"seeded uniform sample ({SPLIT_SIZES['deployment']})",
            "native_splits": "not used -- PQA-L's official 500-question test list is not present locally; "
                             "the population is all 1000 labelled questions",
        },
        sampling={
            "method": "seeded_uniform_sample_from_population",
            "seed": SPLIT_SEED,
            "population": len(items),
            "population_by_label": label_counts,
            "sizes": dict(SPLIT_SIZES),
        },
        splits={role: entries_for(splits[role], extra) for role in SPLIT_ROLES},
        notes=[
            "Reasoning-required setting: the solver sees the question and the abstract sections, never the conclusion.",
            "Ground truth is the official final_decision; scoring is exact match after yes/no/maybe extraction.",
        ],
    ).with_hash()


class PubMedQAAdapter(SingleTurnBenchmarkAdapter):
    name = "pubmedqa"
    metric_name = METRIC_NAME
    scorer_ref = SCORER_REF
    manifest_id = MANIFEST_ID
    data_path = str(PUBMEDQA_ROOT)
    _load_items = staticmethod(load_items)
    _build_user = staticmethod(build_user)
    _evaluate = staticmethod(evaluate)
    payload_meta_keys = ("question", "year")
