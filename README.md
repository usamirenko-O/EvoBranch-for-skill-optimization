# EvoBranch

Code for **Test Time Computing for Evolving Agent Skill Optimization**.
EvoBranch adds branch history, targeted proposals, cross-branch evolution, and
validation-based beam selection around SkillOpt, SSO, and SkillGen.

## Installation

Use Linux and Python 3.11. Run all commands from this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The runners load the included optimizer source directly; separate editable
installs are unnecessary. Additional environments such as ALFWorld require
their own dependencies. The requirements specify compatible lower bounds,
not a complete historical environment lock.

## Data

```bash
python scripts/prepare_data.py --download
python scripts/prepare_data.py --check
```

This prepares LiveMathematicianBench, LiveCodeBench, PubMedQA, and ChemLLMBench.
Each uses 35 train, 35 validation, and 107 deployment tasks, with exact IDs
fixed in `integration/manifests/`. See [data preparation](benchmarks/README.md)
for upstream sources, checksums, and importing existing local data.
Raw datasets are excluded from the source archive.

## Models

The runners use an API-compatible proxy, by default `http://localhost:4000/v1`.
`configs/proxy.example.yaml` defines the optimizer, embedding, and three solver
aliases. Set the following variables for the model endpoints you provide:

| Role | Model variable | Endpoint variable | Credential variable |
| --- | --- | --- | --- |
| Optimizer | `OPTIMIZER_MODEL` | `OPTIMIZER_BASE_URL` | `OPTIMIZER_API_KEY` |
| DeepSeek solver | `DEEPSEEK_MODEL` | `DEEPSEEK_BASE_URL` | `DEEPSEEK_API_KEY` |
| Luna solver | `LUNA_MODEL` | `LUNA_BASE_URL` | `LUNA_API_KEY` |
| Qwen solver | `QWEN_MODEL` | `QWEN_BASE_URL` | `QWEN_API_KEY` |

For example, an API-compatible model uses a LiteLLM provider prefix such as
`openai/deepseek-flash` and a base URL ending in `/v1`. Remove unused solver
entries from your local proxy configuration if you run only one model.
The optimizer and DeepSeek entries disable thinking, matching the included
experiment configuration. Model aliases identify configurations; access to
the corresponding backends must be supplied separately.

To host the proxy and the default `BAAI/bge-small-en-v1.5` embedding service:

```bash
python -m pip install -r requirements-services.txt
python -m integration.experiment.local_embedding_server --port 8318
# In another activated terminal, after setting the endpoint variables:
litellm --config configs/proxy.example.yaml --port 4000
```

The embedding service downloads model weights on first use. Credentials belong
in environment variables, not source files.

## Run an experiment

First resolve the configuration without calling any models:

```bash
bash main_experiment/run_cell.sh skillopt+ours pubmedqa solver-gpt-5-6-luna --dry-run
```

Run the native baseline and EvoBranch with the same benchmark and solver:

```bash
bash main_experiment/run_cell.sh skillopt pubmedqa solver-gpt-5-6-luna
bash main_experiment/run_cell.sh skillopt+ours pubmedqa solver-gpt-5-6-luna
```

Optimizer choices are `skillopt`, `sso`, and `skillgen`; append `+ours` for
EvoBranch. Main benchmark aliases are `livemath`, `livecode`, `pubmedqa`, and
`chem`. The paper configurations use solver aliases `solver-deepseek-flash`,
`solver-gpt-5-6-luna`, and `solver-qwen3-5-4b`.

`main_experiment/baseline_hparams.json` sets native optimizer parameters and
EvoBranch defaults: four rounds, beam width three, and three proposal subsets.
`main_experiment/main_experiment_protocol.json` selects the shared splits.
Use the launcher above to resolve these settings consistently.

An ablation or beam-width variant can be selected explicitly:

```bash
bash main_experiment/run_cell.sh skillopt+ours pubmedqa solver-gpt-5-6-luna \
  --design ablation/designs/wo_global_evolution.json --out-root integration/runs_ablation
```

Outputs go to `integration/runs/` by default. Run records separate optimization
and deployment token costs and retain the resolved configuration. Ordinary
native runs do not automatically reproduce budget-matched comparisons:
those comparisons also require the paper's budget-based selection/truncation
protocol. Historical results and logs are not included in this release.

## Method and source layout

| Component | Source |
| --- | --- |
| Search loop and branch expansion | `ours/controller.py`, `ours/search.py` |
| Branch history and issue tracking | `ours/history.py`, `ours/issue_tracker.py` |
| History-aware partitioning | `ours/partition.py` |
| Global evolution | `ours/operator_history_routed.py`, `ours/operator_recombination.py` |
| Beam selection and final deployment choice | `ours/beam.py`, `ours/deployment_rule.py` |
| Native optimizer adapters | `ours/adapters/`, `ours/experiment/` |
| Benchmark loading, scoring, shared execution | `integration/benchmarks/`, `integration/harness/` |
| Native baseline runners | `integration/experiment/runners/` |
| Fixed settings and ablations | `main_experiment/`, `ablation/designs/` |

`SkillOpt/` and `SkillGen/` contain their respective upstream implementations
with integration changes; `SSO/` contains an unofficial reproduction. Upstream
references: [SkillOpt](https://github.com/microsoft/SkillOpt) and
[SkillGen](https://github.com/yccm/SkillGen). Their existing license and
attribution files are retained. Other upstream utilities are optional; the
commands above are the entry points for the main experiment.

## Source archive

```bash
python scripts/build_release.py
```

This creates `dist/evobranch-anonymous-source.zip` with fixed archive timestamps
and relative paths. It excludes raw corpora, generated data copies, run output,
caches, local credentials, and Git metadata. Installation and dry-run checks
do not establish that the numerical paper results have been rerun.
