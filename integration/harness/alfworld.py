"""Execute skills in ALFWorld with the TextWorld simulator."""
from __future__ import annotations

import os
from types import SimpleNamespace

import yaml

from integration.common.paths import SKILLOPT_ROOT, ensure_on_sys_path
from integration.harness.base import DirectHarness, SolverSession, TaskRun

ensure_on_sys_path(SKILLOPT_ROOT)

from skillopt.envs.alfworld.vendor.alfworld_projection import alfworld_projection  # noqa: E402
from skillopt.envs.alfworld.vendor.env_manager import AlfWorldEnvironmentManager  # noqa: E402

DEFAULT_CONFIG_PATH = SKILLOPT_ROOT / "skillopt" / "envs" / "alfworld" / "vendor" / "config_tw.yaml"
DEFAULT_MAX_STEPS = 50
DEFAULT_HISTORY_LENGTH = 3
SYSTEM_PROMPT = "You are an expert agent operating in the ALFRED Embodied Environment."

_NATIVE_SPLIT_TO_TRAIN_EVAL = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def load_alfworld_config(config_path=None, *, alfworld_data: str | None = None) -> dict:
    """Load SkillOpt's ALFWorld TextWorld config, with `$ALFWORLD_DATA` set.

    ALFWorld resolves every dataset/logic path through `$ALFWORLD_DATA`; this
    sets it from the caller's value (or leaves an already-set environment
    alone) so a run cannot silently read a different corpus than the manifest
    names.
    """
    if alfworld_data:
        os.environ["ALFWORLD_DATA"] = str(alfworld_data)
    with open(config_path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # CUDA is irrelevant to the text environment and the default config asks
    # for it; leaving it True makes alfworld emit a warning on CPU-only hosts.
    config.setdefault("general", {})["use_cuda"] = False
    return config


def _single_game_env(config: dict, gamefile: str, train_eval: str):
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    class _SingleGameEnv(AlfredTWEnv):
        def collect_game_files(self, verbose=False):  # noqa: D102 - see module docstring
            self.game_files = [gamefile]
            self.num_games = 1

    return _SingleGameEnv(config, train_eval=train_eval)


class _SingleGameEnvs:
    """The vectorized-env interface `AlfWorldEnvironmentManager` expects,
    backed by exactly one in-process TextWorld game (batch size 1)."""

    def __init__(self, config: dict, gamefile: str, train_eval: str) -> None:
        self._base = _single_game_env(config, gamefile, train_eval)
        self._env = self._base.init_env(batch_size=1)
        self.prev_admissible_commands = [None]

    def reset(self):
        obs, infos = self._env.reset()
        infos = {k: v[0] for k, v in infos.items()}
        self.prev_admissible_commands = [infos["admissible_commands"]]
        return [obs[0]], None, [infos]

    def step(self, actions):
        obs, scores, dones, infos = self._env.step(list(actions))
        infos = {k: v[0] for k, v in infos.items()}
        self.prev_admissible_commands = [infos["admissible_commands"]]
        reward = 10.0 * float(infos["won"])
        return [obs[0]], None, [reward], [bool(dones[0])], [infos]

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        try:
            self._env.close()
        except Exception:  # noqa: BLE001 - closing a finished env must never fail a run
            pass


class ALFWorldHarness(DirectHarness):
    benchmark = "alfworld"

    def __init__(
        self,
        *,
        alfworld_data: str,
        max_steps: int = DEFAULT_MAX_STEPS,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        config_path=None,
    ) -> None:
        self.alfworld_data = str(alfworld_data)
        self.max_steps = int(max_steps)
        self._config = load_alfworld_config(config_path, alfworld_data=self.alfworld_data)
        self._manager_config = SimpleNamespace(env=SimpleNamespace(history_length=int(history_length)))

    def run_task(self, skill_text: str, task, session: SolverSession) -> TaskRun:
        payload = task.payload
        gamefile = os.path.join(self.alfworld_data, payload["gamefile"])
        train_eval = _NATIVE_SPLIT_TO_TRAIN_EVAL.get(payload.get("native_split", ""), "eval_out_of_distribution")

        envs = _SingleGameEnvs(self._config, gamefile, train_eval)
        manager = AlfWorldEnvironmentManager(envs, alfworld_projection, self._manager_config)
        skill_prompt = f"## Skill\n{skill_text.strip()}\n" if skill_text.strip() else ""

        transcript: list = []
        calls = 0
        won = False
        steps = 0
        error = ""
        try:
            obs, _infos = manager.reset({})
            for step_idx in range(self.max_steps):
                prompt = obs["text"][0]
                if skill_prompt:
                    prompt = skill_prompt + "\n" + prompt
                msg = session.chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
                calls += 1
                response = msg.content or ""
                if "<action>" not in response.lower():
                    # SkillOpt's own fallback for a malformed response.
                    response = "<think>missing action tag</think><action>look</action>"
                transcript.append(f"[step {step_idx} obs]\n{obs['text'][0][:800]}")
                transcript.append(f"[step {step_idx} response]\n{response[:800]}")
                obs, _rewards, dones, infos = manager.step([response])
                steps = step_idx + 1
                won = bool(infos[0]["won"])
                if won or bool(dones[0]):
                    break
        except Exception as exc:  # noqa: BLE001 - an env failure is a failed episode, not a crashed run
            error = f"{type(exc).__name__}: {exc}"
        finally:
            envs.close()

        return TaskRun(
            task_id=task.task_id,
            output={"won": won, "steps": steps},
            transcript="\n\n".join(transcript),
            llm_calls=calls,
            error=error,
            extra={"task_type": payload.get("task_type", "")},
        )
