# EvoSkillRec Amazon Books CTR autoresearch protocol

This file is the executable protocol for autonomous EvoSkillRec CTR evolution runs on the Amazon Books dataset. Follow the newest human instruction first; use this file only as the default operating policy for Books runs.

## Goal

Improve the Amazon Books CTR benchmark by evolving Stage 2 `SkillGenome` DAGs, training candidates, validating them by validation AUC, and writing per-run history. The default benchmark config is `examples/evolution/ctr_amazon_books_evolution.yaml`.

## Operating Rules

- Work from the repository root.
- Stay on the current branch unless the human explicitly asks to create or switch branches.
- Do not commit or push unless the human asks.
- Do not commit `outputs/`, run logs, local `results.tsv`, dataset files, model checkpoints, `__pycache__/`, or generated run artifacts.
- Use validation AUC (`validation_best_auc`) for keep/discard decisions. Test AUC and test logloss are reporting metrics only.
- Do not change label definitions, temporal split semantics, or evaluation metrics after a baseline for the same run.
- Do not hide failed runs. Record crashes, failed candidates, skipped duplicates, and Code-space failures.

## Environment

Use the `rechub` conda environment:

```bash
conda run -n rechub python -c "import torch, pandas, sklearn, yaml; print('ok')"
```

Long evolution runs should use all 8 GPUs for candidate training:

```bash
export CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7}
conda run -n rechub python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Pass `--device cuda:0` for long runs. Do not silently fall back to CPU unless the newest human message explicitly asks for CPU.

## Data

Use the prepared Amazon Books CTR file:

```text
examples/ranking/data/amazon-books/amazon_books_datasets.csv
```

The expected columns are:

```text
user_id,item_id,time,label
```

The configured CTR task uses:

- sparse categorical features: `user_id`, `item_id`;
- label column: `label`;
- temporal split column: `time`;
- split ratio: `0.7/0.1/0.2`.

If the prepared Books file is missing, stop and report the missing path. Do not substitute MovieLens, Beauty, sample data, or another Amazon category unless the human explicitly asks.

Do not commit Amazon Books CSV files.

## Standard 8-GPU Run

Use a run tag such as `books-ctr-jun24`. Each independent run writes to its own output directory.

```bash
mkdir -p outputs/autoresearch/<tag>
CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7} \
conda run -n rechub python examples/evolution/run_ctr_evolution.py \
  --config examples/evolution/ctr_amazon_books_evolution.yaml \
  --device cuda:0 \
  --output-dir outputs/autoresearch/<tag>/<experiment_id> \
  > outputs/autoresearch/<tag>/<experiment_id>.log 2>&1
```

The per-run output directory must contain:

- `results.tsv`: candidate rows plus summary rows for every generation.
- `round_history.tsv`: one baseline row and one row per completed generation, including candidate counts and best-so-far metrics.

Optional files such as `summary.json`, `dataset_metadata.json`, `baseline_genome.json`, and `evolution_memory.jsonl` are controlled by config flags. Per-candidate `model.pth` files must not be committed.

## Evolution Shape

The Books config should run 25 rounds with 8 candidates per generation:

- Initial split: 5 Skill-space candidates and 3 Code-space candidates (`code_space_probability: 0.375`).
- Adaptive budget may shift stagnant rounds to 4 Skill-space and 4 Code-space candidates while keeping total candidates at 8.
- With 8 GPUs, run candidate training in one wave of 8. Each worker gets one visible GPU and uses `--device cuda:0`.
- Skill-space should keep a mix of add, replace, hybridize, and specialize operations.
- Duplicate architectures must be skipped before training and still recorded as `skipped_duplicate`.

If Code-space generation fails or produces too few valid candidates, retry according to `evolution.code_space.max_retries`, record diagnostics, and continue with valid candidates when possible. The workflow should write Code-space failure details instead of silently reporting zero candidates.

## Code-space Policy

Code-space is enabled and macro-only in the Books autoresearch config:

- Provider: `creative_command`.
- Planner: `python tools/code_space_llm_provider.py --mode planner`.
- Synthesizer: `python tools/code_space_llm_provider.py --mode synthesizer`.
- Structural scope: `macro`.
- Built-in deterministic fallback templates are disabled (`allow_fallback_templates: false`).
- Promoted generated skills may be reused (`reuse_promoted_generated_skills: true`).

Every valid Code-space proposal must:

- set `structural_scope: macro`;
- include `metadata.macro_judgment`;
- use macro wiring such as `replace_node`, `insert_between`, `branch_to_fusion`, or `replace_fusion`;
- consume tensor keys available in the retained parent genome;
- pass static safety checks, import/instantiate checks, forward/shape checks, genome validation, and training.

The wrapper backend defaults to Codex:

```bash
export EVOSKILLREC_CODE_SPACE_PROVIDER=codex
# Optionally set EVOSKILLREC_CODE_SPACE_MODEL to an available model.
```

For an external backend, configure `EVOSKILLREC_CODE_SPACE_PROVIDER` and the required API key or command before launching. If no usable backend exists, stop and report the blocker.

## Result Review

After a run, inspect:

- `round_history.tsv` for one row per generation and best-so-far validation AUC.
- `results.tsv` for candidate metrics, failures, skipped duplicates, architecture fingerprints, and promotion status.
- `round*_code_space_proposals.json` or `round*_code_space_error.json` when Code-space candidates are missing or failed.
- The run log only when the workflow crashes or stalls.

Keep an experiment only if its best survivor improves the incumbent validation AUC by at least `training.min_delta`. Otherwise record it as discard and continue with a new hypothesis.

## Error Handling

Run non-interactively once a research run starts.

- If a fast check fails, inspect the failure, make a minimal fix, and rerun the focused check.
- If individual candidates fail during generation, verification, import, compilation, training, or evaluation, mark those candidates failed and continue the generation when possible.
- If a workflow crashes before candidate metadata is written, inspect the last log lines, fix obvious syntax/import/shape/config errors, and rerun once or twice.
- If GPUs fail or OOM, reduce parallelism or batch size for that run and record the resource change.
- Stop only for hard blockers: missing Books dataset, no usable Code-space provider, unavailable GPUs after retry, missing conda environment, missing dependencies requiring installation approval, or human changes that make the branch unsafe to modify.

## Useful Commands

Focused checks:

```bash
conda run -n rechub python -m pytest tests/evolution/test_code_space_flow.py tests/evolution/test_ctr_workflow.py -q
```

Config/data smoke check:

```bash
conda run -n rechub python -c "from recskill.evolution.ctr_workflow import load_workflow_config, prepare_movielens_ctr_data, _candidate_targets; c=load_workflow_config('examples/evolution/ctr_amazon_books_evolution.yaml'); print(_candidate_targets(c.evolution)); c.dataset.limit_rows=1000; b=prepare_movielens_ctr_data(c.dataset, c.training); print(b.metadata['num_rows'], b.metadata['feature_names'], b.metadata['positive_rate'])"
```

Open-ended ingestion debug:

```bash
conda run -n rechub python -m recskill.evolution.cli ingest-open-ended-proposal \
  --proposal path/to/proposal.json \
  --genome path/to/genome.json \
  --out outputs/autoresearch/<tag>/open_ended
```

Repeat-run template:

```bash
for seed in 2022 2023 2024; do
  CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7} \
  conda run -n rechub python examples/evolution/run_ctr_evolution.py \
    --config examples/evolution/ctr_amazon_books_evolution.yaml \
    --device cuda:0 \
    --seed "${seed}" \
    --output-dir "outputs/autoresearch/<tag>/books_seed${seed}"
done
```
