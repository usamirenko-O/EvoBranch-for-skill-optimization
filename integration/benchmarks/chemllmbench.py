"""Load ChemLLMBench yield tasks and create label-stratified splits."""
from __future__ import annotations

from collections import Counter

from integration.benchmarks.manifest import SPLIT_ROLES, SplitManifest, source_inputs
from integration.benchmarks.paths import CHEMLLMBENCH_ROOT, CHEMLLMBENCH_YIELD_DIR
from integration.benchmarks.single_turn_base import (
    SPLIT_SEED,
    SPLIT_SIZES,
    SingleTurnBenchmarkAdapter,
    entries_for,
    stratified_split,
)
from skillopt.envs.chemllmbench.dataloader import SUBSETS, load_items
from skillopt.envs.chemllmbench.evaluator import evaluate
from skillopt.envs.chemllmbench.rollout import build_user

MANIFEST_ID = "chemllmbench_yield_sample200_stratified_177_v2"
METRIC_NAME = "chemllmbench_yield_accuracy_yes_no"
SCORER_REF = "skillopt.envs.chemllmbench.evaluator:evaluate"


def build_manifest() -> SplitManifest:
    items = load_items(str(CHEMLLMBENCH_ROOT))
    by_id = {i["id"]: i for i in items}
    by_stratum: dict = {}
    for i in items:
        by_stratum.setdefault(f"{i['subset']}:{i['label']}", []).append(i["id"])
    splits, quotas = stratified_split(by_stratum)
    label_counts = dict(sorted(Counter((i["subset"], i["label"]) for i in items).items()))

    def extra(task_id: str) -> dict:
        return {"subset": by_id[task_id]["subset"]}

    return SplitManifest(
        benchmark="chemllmbench",
        manifest_id=MANIFEST_ID,
        version="1",
        source={
            "upstream_dataset": "ChemFoundationModels/ChemLLMBench yield_prediction (sampled 100-reaction test sets)",
            "upstream_files": [SUBSETS[k] for k in sorted(SUBSETS)],
            "materialized_from": str(CHEMLLMBENCH_YIELD_DIR),
            "inputs": source_inputs([CHEMLLMBENCH_YIELD_DIR / SUBSETS[k] for k in sorted(SUBSETS)]),
            "official_scorer": {
                "ref": SCORER_REF,
                "upstream": "ChemLLMBench yield prediction accuracy (high-yield Yes/No exact match); "
                            "extraction copied from SkillGen/benchmarks/chemllmbench_grader.py",
            },
        },
        role_map={
            "train": f"seeded stratified sample ({SPLIT_SIZES['train']})",
            "val": f"seeded stratified sample ({SPLIT_SIZES['val']})",
            "deployment": f"seeded stratified sample ({SPLIT_SIZES['deployment']})",
            "native_splits": "none -- the two *_sample_100_test.npz files are ChemLLMBench's evaluation sample",
        },
        sampling={
            "method": "seeded_stratified_by_subset_label_largest_remainder",
            "seed": SPLIT_SEED,
            "population": len(items),
            "population_by_subset_label": {f"{s}:{l}": n for (s, l), n in label_counts.items()},
            "quota_by_role": quotas,
            "sizes": dict(SPLIT_SIZES),
        },
        splits={role: entries_for(splits[role], extra) for role in SPLIT_ROLES},
        notes=[
            "Population = BH_sample_100_test + SU_sample_100_test (the SkillGen-paper sample); full BH/Suzuki sets unused.",
            "Task text is SkillGen's yield-prediction prompt wording; label = Yes iff yield > 30%.",
            "Label imbalance: 160 No / 40 Yes; a constant 'No' answer scores 0.80.",
            "Each role keeps the population's (subset, label) proportion (largest-remainder rounding), "
            "so train/val/deployment all contain Yes reactions from both BH and SU.",
        ],
    ).with_hash()


class ChemLLMBenchAdapter(SingleTurnBenchmarkAdapter):
    name = "chemllmbench"
    metric_name = METRIC_NAME
    scorer_ref = SCORER_REF
    manifest_id = MANIFEST_ID
    data_path = str(CHEMLLMBENCH_ROOT)
    _load_items = staticmethod(load_items)
    _build_user = staticmethod(build_user)
    _evaluate = staticmethod(evaluate)
    payload_meta_keys = ("subset", "reaction")
