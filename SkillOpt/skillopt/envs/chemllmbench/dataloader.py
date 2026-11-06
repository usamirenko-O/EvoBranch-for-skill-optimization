"""ChemLLMBench yield-prediction dataloader."""
from __future__ import annotations

import os

from skillopt.datasets.base import SplitDataLoader

SUBSETS = {
    "BH": "BH_sample_100_test.npz",
    "SU": "SU_sample_100_test.npz",
}
SUBSET_NAMES = {"BH": "Buchwald-Hartwig", "SU": "Suzuki-Miyaura"}


def _yield_dir(data_path: str) -> str:
    if os.path.isdir(os.path.join(data_path, "yield_prediction")):
        return os.path.join(data_path, "yield_prediction")
    if os.path.isdir(data_path):
        return data_path
    raise ValueError(f"ChemLLMBench yield_prediction directory not found under {data_path!r}")


def load_items(data_path: str) -> list[dict]:
    import numpy as np

    base = _yield_dir(data_path)
    items: list[dict] = []
    for subset, fname in SUBSETS.items():
        path = os.path.join(base, fname)
        if not os.path.isfile(path):
            raise ValueError(f"missing {path}")
        arr = np.load(path, allow_pickle=True)["data_df"]
        for idx, (reaction, label) in enumerate(arr):
            label_text = str(label).strip()
            if label_text.lower() not in ("yes", "no"):
                raise ValueError(f"{path} row {idx}: unexpected label {label!r}")
            items.append(
                {
                    "id": f"{subset}_{idx:03d}",
                    "subset": subset,
                    "reaction": str(reaction),
                    "label": "Yes" if label_text.lower() == "yes" else "No",
                }
            )
    if not items:
        raise ValueError(f"no ChemLLMBench yield items loaded from {base}")
    return items


class ChemLLMBenchDataLoader(SplitDataLoader):
    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(data_path)

    def get_task_types(self) -> list[str]:
        return sorted(SUBSETS)
