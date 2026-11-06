# Benchmark data

Raw benchmark data is downloaded separately from the source release. Run from
the repository root with Python 3.11:

```bash
python scripts/prepare_data.py --download
python scripts/prepare_data.py --check
```

To reuse an existing data directory with the layout below:

```bash
python scripts/prepare_data.py --from /path/to/benchmarks
```

The helper verifies every file against `assets.json` and rejects mismatched
downloads. It does not overwrite existing files. The main datasets require
approximately 670 MiB of disk space. Upstream dataset terms apply.

| Benchmark | Upstream | Files under `benchmarks/` |
| --- | --- | --- |
| LiveMathematicianBench | [Official dataset](https://huggingface.co/datasets/LiveMathematicianBench/LiveMathematicianBench) | `livemathematicianbench/data/{202511,202512,202601,202602}/qa_<month>_final.json` |
| LiveCodeBench | [Code generation lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) | `livecodebench/test5.jsonl`, `livecodebench/test6.jsonl` |
| PubMedQA | [PQA-L](https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json) | `pubmedqa/ori_pqal.json` |
| ChemLLMBench | [Yield prediction data](https://github.com/ChemFoundationModels/ChemLLMBench/tree/main/data/yield_prediction) | `chemllmbench/yield_prediction/BH_sample_100_test.npz`, `chemllmbench/yield_prediction/SU_sample_100_test.npz` |

The main experiment uses 35 train, 35 validation, and 107 deployment tasks per
benchmark. Exact task IDs are in `integration/manifests/`; the selected manifests
and their hashes are in `main_experiment/main_experiment_protocol.json`.
LiveMath uses the four-month snapshot: 17 original validation tasks plus 18
tasks sampled from the original test split form validation, leaving 107 for
deployment. The original ID splits remain in `SkillOpt/data/`.

Adapters for OfficeQA, SpreadsheetBench, and ALFWorld are also included. Their
additional datasets and environment dependencies are described in
[`SkillOpt/data/README.md`](../SkillOpt/data/README.md); the helper above prepares
only the four main benchmarks.
