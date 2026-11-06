"""Resolve benchmark data and split-manifest paths relative to the repository."""
from __future__ import annotations

from integration.common.paths import BENCHMARKS_ROOT, WORKSPACE_ROOT

# --- optimizer repos whose frozen artifacts this package reads (never writes) ---
SKILLOPT_DATA_ROOT = WORKSPACE_ROOT / "SkillOpt" / "data"
EVOSKILL_ROOT = WORKSPACE_ROOT / "EvoSkill"

# --- raw benchmark corpora, exactly as downloaded in Phase 0 ---
OFFICEQA_ROOT = BENCHMARKS_ROOT / "officeqa"
OFFICEQA_PRO_CSV = OFFICEQA_ROOT / "officeqa_pro.csv"
OFFICEQA_FULL_CSV = OFFICEQA_ROOT / "officeqa_full.csv"  # not present locally (gated)
OFFICEQA_DOCS_ROOT = OFFICEQA_ROOT / "treasury_bulletins_parsed" / "transformed"

SPREADSHEETBENCH_ROOT = BENCHMARKS_ROOT / "spreadsheetbench" / "spreadsheetbench_verified_400"
SPREADSHEETBENCH_DATASET_JSON = SPREADSHEETBENCH_ROOT / "dataset.json"
SPREADSHEETBENCH_SPREADSHEET_ROOT = SPREADSHEETBENCH_ROOT / "spreadsheet"

ALFWORLD_ROOT = BENCHMARKS_ROOT / "alfworld"
ALFWORLD_JSON_ROOT = ALFWORLD_ROOT / "json_2.1.1"
ALFWORLD_LOGIC_ROOT = ALFWORLD_ROOT / "logic"

# --- frozen SkillOpt split manifests, the shared task-ID source of truth ---
OFFICEQA_ID_SPLIT = SKILLOPT_DATA_ROOT / "officeqa_id_split"
SPREADSHEETBENCH_ID_SPLIT = SKILLOPT_DATA_ROOT / "spreadsheetbench_id_split"
ALFWORLD_PATH_SPLIT = SKILLOPT_DATA_ROOT / "alfworld_path_split"
LIVEMATH_ID_SPLIT = SKILLOPT_DATA_ROOT / "livemathematicianbench_id_split"


LIVECODEBENCH_ROOT = BENCHMARKS_ROOT / "livecodebench"          
PUBMEDQA_ROOT = BENCHMARKS_ROOT / "pubmedqa"                    
PUBMEDQA_PQAL_JSON = PUBMEDQA_ROOT / "ori_pqal.json"
CHEMLLMBENCH_ROOT = BENCHMARKS_ROOT / "chemllmbench"            # yield_prediction/*.npz
CHEMLLMBENCH_YIELD_DIR = CHEMLLMBENCH_ROOT / "yield_prediction"

# --- where this package persists the manifests it builds ---
MANIFEST_DIR = WORKSPACE_ROOT / "integration" / "manifests"

__all__ = [
    "CHEMLLMBENCH_ROOT", "CHEMLLMBENCH_YIELD_DIR", "LIVECODEBENCH_ROOT", "PUBMEDQA_PQAL_JSON", "PUBMEDQA_ROOT",
    "ALFWORLD_JSON_ROOT", "ALFWORLD_LOGIC_ROOT", "ALFWORLD_PATH_SPLIT", "ALFWORLD_ROOT",
    "EVOSKILL_ROOT", "LIVEMATH_ID_SPLIT", "MANIFEST_DIR", "OFFICEQA_DOCS_ROOT",
    "OFFICEQA_FULL_CSV", "OFFICEQA_ID_SPLIT", "OFFICEQA_PRO_CSV", "OFFICEQA_ROOT",
    "SKILLOPT_DATA_ROOT", "SPREADSHEETBENCH_DATASET_JSON", "SPREADSHEETBENCH_ID_SPLIT",
    "SPREADSHEETBENCH_ROOT", "SPREADSHEETBENCH_SPREADSHEET_ROOT",
]
