"""Load each optimizer's native initial skill or no-skill starting state."""
from __future__ import annotations

from integration.common.paths import SKILLOPT_ROOT

#: SkillOpt 的原生初始 skill（只有 SkillOpt 用）
_SKILLOPT_INITIAL_RELPATH = {
    "officeqa": "skillopt/envs/officeqa/skills/initial.md",
    "spreadsheetbench": "skillopt/envs/spreadsheetbench/skills/initial.md",
    "alfworld": "skillopt/envs/alfworld/skills/initial.md",
    "livemathematicianbench": "skillopt/envs/livemathematicianbench/skills/initial.md",
    "livecodebench": "skillopt/envs/livecodebench/skills/initial.md",
    "pubmedqa": "skillopt/envs/pubmedqa/skills/initial.md",
    "chemllmbench": "skillopt/envs/chemllmbench/skills/initial.md",
}

#: optimizer -> S_0 的性质，写进 artifact 供审计
NATIVE_START_CONDITION = {
    "skillopt": "skillopt_initial_skill_document",
    "sso": "no_skill",
    "skillgen": "no_skill",
}


def skillopt_initial_skill_path(benchmark: str):
    """SkillOpt 的初始 skill 文件路径。只有 SkillOpt runner 会用到。"""
    return SKILLOPT_ROOT / _SKILLOPT_INITIAL_RELPATH[benchmark]


def initial_skill_text(optimizer: str, benchmark: str) -> str:
    """该 optimizer 在这个 benchmark 上的原生 S_0。

    SSO 和 SkillGen 返回空串 —— 这是 no-skill 条件本身，不是"文件没找到"的
    降级。四个 benchmark 的 harness 都能正确处理空 skill（不注入 skill section）。
    """
    if optimizer == "skillopt":
        path = skillopt_initial_skill_path(benchmark)
        if not path.is_file():
            raise FileNotFoundError(f"SkillOpt 的初始 skill 不存在: {path}")
        return path.read_text(encoding="utf-8")
    if optimizer in ("sso", "skillgen"):
        return ""
    raise KeyError(f"未知 optimizer: {optimizer!r}")
