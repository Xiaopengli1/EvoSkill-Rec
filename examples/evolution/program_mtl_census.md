# EvoSkillRec Census-Income MTL autoresearch protocol

This file is the executable protocol for autonomous EvoSkillRec multi-task evolution runs on the census-income dataset. Follow the newest human instruction first; use this file only as the default operating policy for census MTL runs.

## Goal

Improve the census-income multi-task benchmark by evolving Stage 2 `SkillGenome` DAGs, training candidates, validating them by mean validation AUC across tasks, and writing per-run history. The default autoresearch workflow config is `examples/evolution/mtl_census_codespace.yaml`.

The initial genome uses `genome.template: aitm`.

The task contract is fixed:

- `income` is task 0 and is treated as the CTR-style task.
- `marital status` is task 1 and is treated as the CVR-style task.
- Task names are `income_ctr` and `marital_cvr`.
- Both tasks are binary classification tasks.

## Operating Rules

- Work from the repository root.
- Stay on the current branch unless the human explicitly asks to create or switch branches.
- Do not commit or push unless the human asks.
- Do not commit `outputs/`, run logs, local `results.tsv`, dataset files, model checkpoints, `__pycache__/`, or generated run artifacts.
- Use validation mean AUC (`validation_best_auc` / `auc`) for keep/discard decisions. Per-task AUCs, test AUC, and test logloss are reporting metrics only.
- Do not change label definitions, split semantics, dense/sparse feature handling, task order, or evaluation metrics after a baseline for the same run.
- Do not hide failed runs. Record crashes, failed candidates, skipped duplicates, Code-space failures, and generated-skill promotion failures.

## Environment

Use the `rechub` conda environment:

```bash
conda run -n rechub python -c "import torch, pandas, sklearn, yaml; print('ok')"
```

Long MTL runs must use all 8 GPUs for candidate training:

```bash
export CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7}
conda run -n rechub python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Pass `--device cuda:0` for long runs. Do not silently fall back to CPU unless the newest human message explicitly asks for CPU.
Do not launch long MTL runs with `CUDA_VISIBLE_DEVICES=${GPU_ID:-0}` because that restricts the workflow to a single physical GPU. The MTL runner uses `CTR_EVOLUTION_GPU_IDS` to assign one candidate worker per GPU; each worker sees one GPU and uses `--device cuda:0`.

## Data

Use the prepared census-income split files:

```text
examples/ranking/data/census-income/census_income_train.csv
examples/ranking/data/census-income/census_income_val.csv
examples/ranking/data/census-income/census_income_test.csv
```

The configured MTL task uses:

- labels: `income`, `marital status`;
- dense continuous features: `age`, `wage per hour`, `capital gains`, `capital losses`, `divdends from stocks`, `num persons worked for employer`, `weeks worked in year`;
- sparse categorical features: all non-label feature columns except the dense continuous columns.

Feature handling must follow the Census preprocessing contract:

- sparse categorical columns stay as sparse ids and feed `sparse_features -> field_embedding`;
- dense continuous columns stay continuous and feed `dense_values -> dense_feature_path`;
- dense columns must not be binned into categorical ids;
- the combined field stack flows through `field_embeddings -> flatten -> flat_embeddings`.

If any prepared census split file is missing, stop and report the missing path. Do not substitute MovieLens, Amazon, sample data, or dense-only data unless the human explicitly asks.

Do not commit census CSV files.

## Standard Autoresearch Run

Use a run tag such as `mtl-census-jun25`. Each independent run writes to its own output directory.

The standard MTL autoresearch config uses live `creative_command` Code-space generation. It must ask the LLM in real time using the current parent genome, current adaptive budget state, recent round history, recent failed candidates, Code-space provider/ingestion errors, and census runtime dimensions.

```bash
mkdir -p outputs/autoresearch/<tag>
CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7} \
conda run -n rechub python examples/evolution/run_mtl_evolution.py \
  --config examples/evolution/mtl_census_codespace.yaml \
  --device cuda:0 \
  --output-dir outputs/autoresearch/<tag>/<experiment_id> \
  > outputs/autoresearch/<tag>/<experiment_id>.log 2>&1
```

For a Skill-space-only run, use:

```bash
CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7} \
conda run -n rechub python examples/evolution/run_mtl_evolution.py \
  --config examples/evolution/mtl_census_evolution.yaml \
  --device cuda:0 \
  --output-dir outputs/autoresearch/<tag>/<experiment_id> \
  > outputs/autoresearch/<tag>/<experiment_id>.log 2>&1
```

Use the live Code-space run only after confirming `EVOSKILLREC_CODE_SPACE_PROVIDER` and model/API credentials are configured. If no usable live backend exists, stop and report the blocker.

The per-run output directory must contain:

- `results.tsv`: baseline and candidate rows plus summary rows for every generation.
- `round_history.tsv`: one baseline row and one row per completed generation, including candidate counts and best-so-far metrics.

Optional files such as `summary.json`, `dataset_metadata.json`, `baseline_genome.json`, `evolution_memory.jsonl`, and Code-space diagnostics are controlled by config flags. Per-candidate `model.pth` files must not be committed.

## Evolution Shape

The default MTL Code-space config should run 25 rounds with 8 candidates per generation:

- Initial split: 6 Skill-space candidates and 2 Code-space candidates (`code_space_probability: 0.25`).
- Adaptive budget keeps the total candidate budget fixed and can shift stagnant rounds from 4 Skill-space / 4 Code-space to 2 Skill-space / 6 Code-space.
- At least 2 Skill-space candidates must remain in every adaptive round.
- With 8 GPUs, run candidate training in one wave of 8. Each worker gets one visible GPU and uses `--device cuda:0`.
- Skill-space should search shared-transform variants: `aitm`, `shared_bottom`, `mmoe`, `ple`, and `task_specific`, plus capacity specializations.
- Skill-space must actively sample reusable promoted generated skills from the global generated-skill pool when `reuse_promoted_generated_skills: true`.
- Duplicate architectures must be skipped before training and still recorded as `skipped_duplicate`.

With the live `creative_command` provider, adaptive budget increases must result in fresh LLM planner/synthesizer calls for the larger Code-space quota. If the provider cannot return enough valid candidates, record diagnostics, feed provider/ingestion failures back into the next provider attempt, and fill missing quota with Skill-space supplement candidates only after the configured live attempts fail.

## Code-space Policy

Live Code-space is enabled and macro-only:

- Fresh open-ended autoresearch: `creative_command`, using `examples/evolution/mtl_census_codespace.yaml`.
- Planner: `python tools/code_space_llm_provider.py --mode planner --task multitask`.
- Synthesizer: `python tools/code_space_llm_provider.py --mode synthesizer --task multitask`.
- Structural scope: `macro`.
- Generated skills may be promoted by survivor policy.
- Promoted generated skills must re-enter Skill-space reuse (`reuse_promoted_generated_skills: true`).

Every live Code-space call must include:

- current parent genome and MTL architecture profile;
- current census dataset/runtime dimensions, including sparse fields, dense fields, `num_fields`, `embedding_dim`, and `flat_input_dim`;
- adaptive quota state and stagnation count;
- recent round summaries, including validation mean AUC, test metrics, candidate space counts, and failures;
- recent provider, ingestion, generation, duplicate, shape, import, and training errors;
- retry feedback from the immediately previous failed provider/ingestion attempt in the same round.

Every valid MTL Code-space proposal must:

- set `task_types: [multitask]`;
- set `structural_scope: macro`;
- include `metadata.macro_judgment`;
- use only topology-generic wiring: `insert_between` or `replace_node`;
- never use CTR logits/fusion wiring such as `branch_to_fusion`, `replace_fusion`, or `reroute_logits`;
- never replace protected nodes: `field_embedding`, `task_tower`, or `loss`;
- target one of the advertised MTL lanes: `field_embeddings`, `flat_embeddings`, or `task_representations`;
- preserve the intercepted tensor shape exactly;
- pass static safety checks, import/instantiate checks, forward/shape checks, genome validation, and training.

The wrapper backend defaults to Codex for live Code-space:

```bash
export EVOSKILLREC_CODE_SPACE_PROVIDER=codex
# Optionally set EVOSKILLREC_CODE_SPACE_MODEL to an available model.
```

For an external backend, configure `EVOSKILLREC_CODE_SPACE_PROVIDER` and the required API key or command before launching. If no usable backend exists for a live Code-space run, stop and report the blocker.

## Skill Sedimentation

Generated Code-space skills that survive must be promoted according to `promotion_policy: survivor`. Promotion should persist the generated PyTorch implementation and skill card, mark the skill reusable when portability checks allow it, and register it so later Skill-space rounds can sample it.

MTL Skill-space reuse must support these stable lanes:

- `field_embeddings` before flattening;
- `flat_embeddings` before the shared transform;
- `task_representations` before `task_tower`.

Template mutations must carry compatible generated insertions forward across rebuilt MTL templates.

## Result Review

After a run, inspect:

- `round_history.tsv` for one row per generation and best-so-far validation mean AUC.
- `results.tsv` for candidate metrics, failures, skipped duplicates, architecture fingerprints, validation/test task AUCs, structural scope, and promotion status.
- `dataset_metadata.json` to confirm `num_dense_binned=0`, raw dense handling, and the expected sparse/dense split.
- `round*_mtl_code_space_proposals.json` when Code-space candidates are missing or failed.
- The run log only when the workflow crashes or stalls.

Keep an experiment only if its best survivor improves the incumbent validation mean AUC by at least `training.min_delta`. Otherwise record it as discard and continue with a new hypothesis.

## Error Handling

Run non-interactively once a research run starts.

- If a fast check fails, inspect the failure, make a minimal fix, and rerun the focused check.
- If individual candidates fail during generation, verification, import, compilation, training, or evaluation, mark those candidates failed and continue the generation when possible.
- If a workflow crashes before candidate metadata is written, inspect the last log lines, fix obvious syntax/import/shape/config errors, and rerun once or twice.
- If GPUs fail or OOM, reduce parallelism or batch size for that run and record the resource change.
- Stop only for hard blockers: missing prepared census split files, no usable live Code-space provider when live generation is requested, unavailable GPUs after retry, missing conda environment, missing dependencies requiring installation approval, or human changes that make the branch unsafe to modify.

## Useful Commands

Focused checks:

```bash
conda run -n rechub python -m pytest tests/evolution/test_mtl_workflow.py -q
```

Full evolution checks:

```bash
conda run -n rechub python -m pytest tests/evolution -q
```

Config/data smoke check:

```bash
conda run -n rechub python -c "from recskill.evolution.mtl_workflow import load_mtl_workflow_config, prepare_census_mtl_data, build_mtl_genome; c=load_mtl_workflow_config('examples/evolution/mtl_census_codespace.yaml'); c.dataset.limit_rows=1000; c.training.batch_size=128; b=prepare_census_mtl_data(c.dataset, c.training); g=build_mtl_genome(b, c.genome); print(b.metadata['num_sparse_features'], b.metadata['num_dense_features'], b.metadata['num_dense_binned'], g.metadata.extras['num_fields'], g.metadata.extras['flat_input_dim'])"
```

CPU smoke run (data/training path only; live Code-space requires a configured backend):

```bash
conda run -n rechub python -c "from recskill.evolution.mtl_workflow import load_mtl_workflow_config, MTLModelEvolutionRunner; c=load_mtl_workflow_config('examples/evolution/mtl_census_codespace.yaml'); c.output_dir='outputs/autoresearch/mtl-census-smoke/cpu_smoke'; c.dataset.limit_rows=300; c.training.device='cpu'; c.training.epoch=1; c.training.batch_size=128; c.training.max_train_batches=1; c.training.max_eval_batches=1; c.evolution.rounds=1; c.evolution.candidate_budget=2; c.evolution.code_space.provider='disabled'; c.evolution.code_space_probability=0.0; c.evolution.adaptive_candidate_budget.enabled=False; c.write_survivor_artifacts=False; print(MTLModelEvolutionRunner(c).run()['task_label_cols'])"
```

Codex entry invocation example:

```bash
codex exec --cd "$PWD" --json - \
  < examples/evolution/program_mtl_census.md
```
