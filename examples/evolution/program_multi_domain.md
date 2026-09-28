# EvoSkillRec Multi-Domain autoresearch protocol

This file is the executable protocol for autonomous EvoSkillRec multi-domain
evolution runs. Follow the newest human instruction first; use this file only as
the default operating policy for multi-domain runs.

## Goal

Improve multi-domain recommendation benchmarks by evolving Stage 2 `SkillGenome`
DAGs, training candidates, validating them by validation AUC, and writing
per-run history. This is a generic multi-domain entrypoint: dataset-specific
behavior must live in YAML config, not in the workflow code.

Supported starter configs:

- Amazon: `examples/evolution/multi_domain_amazon_evolution.yaml`
- MovieLens: `examples/evolution/multi_domain_movielens_evolution.yaml`

Select the dataset and its YAML from the human's starting instruction.
Use that dataset consistently throughout the run. If no dataset is specified,
ask which starter configuration to use before launching training.

The task contract is common:

- model input includes `sparse_features`, `domain_indicator`, and `labels`;
- optional dense fields feed `dense_values -> dense_feature_path`;
- domain ids are routed through `domain_indicator_adapter -> domain_id`;
- active domain prediction is selected by `domain_select`;
- objective is binary BCE over `prediction` with `from_logits: false`.

## Environment

Use the `rechub` conda environment and verify PyTorch, pandas, scikit-learn,
and PyYAML before launching. Use the GPU pool provided by the human through
`CTR_EVOLUTION_GPU_IDS`; the commands below show an eight-GPU example.
Live Code-space defaults to the Codex backend. Verify that the Codex CLI is
available and authenticated in the environment used by candidate generation.

## Dataset Contracts

Any CSV dataset can be used when the config provides a label column, a domain
column or derivation rule, and sparse/dense feature rules. The starter configs
below are examples, not separate code paths.

Amazon uses an existing `domain_indicator` column:

```text
examples/multi_domain/data/amazon_5_core/amazon.csv
```

MovieLens derives `domain_indicator` from `age` using the rules in
`examples/evolution/multi_domain_movielens_evolution.yaml`:

- domain 0: ages `1, 18`;
- domain 1: age `25`;
- domain 2: ages `35, 45, 50, 56`;
- label: `rating > 3`;
- `genres` is converted to first-genre `cate_id`;
- `title`, `timestamp`, and raw `genres` are excluded from sparse features;
- `age` is used as a normalized dense feature.

If the configured dataset is missing, stop and report the missing path. Do not
substitute a different dataset unless the human explicitly asks.

## Standard Runs

Use a descriptive run tag. Each independent run writes to its own output
directory. For an Amazon task:

```bash
mkdir -p outputs/autoresearch/<tag>
CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7} \
conda run -n rechub python examples/evolution/run_multi_domain_evolution.py \
  --config examples/evolution/multi_domain_amazon_evolution.yaml \
  --device cuda:0 \
  --output-dir outputs/autoresearch/<tag>/<experiment_id> \
  > outputs/autoresearch/<tag>/<experiment_id>.log 2>&1
```

For a MovieLens task:

```bash
mkdir -p outputs/autoresearch/<tag>
CTR_EVOLUTION_GPU_IDS=${CTR_EVOLUTION_GPU_IDS:-0,1,2,3,4,5,6,7} \
conda run -n rechub python examples/evolution/run_multi_domain_evolution.py \
  --config examples/evolution/multi_domain_movielens_evolution.yaml \
  --device cuda:0 \
  --output-dir outputs/autoresearch/<tag>/<experiment_id> \
  > outputs/autoresearch/<tag>/<experiment_id>.log 2>&1
```

Both starter configs use `limit_rows: null`, so standard evolution runs use the
full configured dataset. Keep row limits only for local smoke checks by
overriding `c.dataset.limit_rows` in the check command or by using a throwaway
debug config.

## Evolution Shape

The current multi-domain Skill-space uses the multi-domain skills already
sedimented in `recskill/skills/scenario`:

- `star`;
- `shared_bottom`;
- `mmoe`;
- `ple`;
- `sarnet`;
- `adaptdhm`;
- `hamur_small`;
- `hamur_large`;
- `m3oe`;
- `ppnet`;
- `m2m`;
- `adasparse`;
- `epnet`.

The Amazon autoresearch config starts from STAR and runs 25 rounds with
8 candidates per generation:

- Initial split: 6 Skill-space candidates and 2 Code-space candidates
  (`code_space_probability: 0.25`).
- Adaptive budget keeps the total candidate budget fixed at 8 and can shift
  stagnant rounds toward Code-space up to 2 Skill-space / 6 Code-space.
- At least 2 Skill-space candidates remain in every adaptive round, while the
  open-ended Code-space quota can expand to 6 candidates when skill-space
  progress stalls.
- Skill-space candidates are selected from a unified scored pool built from
  YAML templates, default sedimented multi-domain skills, and reusable promoted
  generated skills. Selection uses skill-card task metadata, retrieval text,
  composition hints, parent architecture state, failure modes, and candidate
  metrics when available.
- The selector enforces source/family diversity before applying a small
  round-aware exploration slot, so the run does not collapse onto only the
  highest-scoring families while still remaining score-driven.
- With 8 GPUs, run candidate training in one wave of 8. Each worker gets one
  visible GPU and uses `--device cuda:0`.

The MovieLens configuration starts from SARNET and runs 25 rounds with
10 candidates per generation. Its adaptive budget starts with 2 Code-space
candidates and can increase to 8 while retaining at least 2 Skill-space
candidates. Read these values from the selected YAML; do not copy the Amazon
budget into a MovieLens run.

Code-space is enabled with the live `creative_command` provider. The planner
and synthesizer task flag is `multi_domain`:

```bash
python tools/code_space_llm_provider.py --mode planner --task multi_domain
python tools/code_space_llm_provider.py --mode synthesizer --task multi_domain
```

If no usable live backend is configured for `tools/code_space_llm_provider.py`,
stop and report the blocker instead of treating the run as a valid Code-space
experiment.

## Code-space Policy

Multi-domain Code-space should be open-ended over the parent `SkillGenome` DAG,
with hard constraints only for safety, compilation, and fair evaluation.

Every executable multi-domain Code-space proposal must:

- set `task_types: [multi_domain]`;
- set `structural_scope: macro`;
- include `metadata.macro_judgment`;
- use current executable DAG integration wiring: `insert_between` or `replace_node`;
- target any suitable parent node or edge, not just the example lanes;
- preserve the downstream boundary contract, not necessarily every internal
  hidden shape;
- use valid proposal types accepted by `OpenEndedProposal`, such as
  `NEW_SKILL_INVENTION`, `NEW_EMBEDDING_DESIGN`, `NEW_INTERACTION_DESIGN`,
  `NEW_BRANCH_DESIGN`, `NEW_ROUTING_OR_GATING_DESIGN`, or `LOCAL_CODE_SURGERY`;
- pass static safety checks, import/instantiate checks, forward/shape checks, genome validation, and training.

The search lanes are intentionally broad:

- embedding and dense feature processing;
- field interaction and representation transforms;
- multi-domain backbone replacement;
- domain adapter/routing;
- domain expert routing;
- domain tower/head design;
- domain selection;
- regularization signals that can be expressed as generated DAG modules.

The advertised tensors such as `field_embeddings`, `flat_embeddings`,
`domain_representations`, and `domain_outputs` are stable examples, not the full
search space. A valid proposal can target other available context keys or nodes
when the generated module can be compiled into the current parent DAG.

Do not use CTR-only logits/fusion shortcuts such as `branch_to_fusion`,
`replace_fusion`, or `reroute_logits` unless the parent multi-domain genome
actually exposes a compatible terminal. The current generated-module engine does
not yet perform arbitrary repo patches or terminal objective rewrites in this
multi-domain path; those ideas should be recorded as future engine extensions
rather than emitted as executable candidates.

Generated Code-space skills that survive are promoted through the shared
EvoSkillRec generated-skill promotion path under `recskill/generated_skills`.

## Useful Commands

Focused checks:

```bash
conda run -n rechub python -m pytest tests/evolution/test_multi_domain_workflow.py
```

Amazon config/data smoke:

```bash
conda run -n rechub python -c "from recskill.evolution.multi_domain_workflow import load_multi_domain_workflow_config, prepare_multi_domain_data, build_multi_domain_genome; c=load_multi_domain_workflow_config('examples/evolution/multi_domain_amazon_evolution.yaml'); c.dataset.limit_rows=3000; c.training.batch_size=256; b=prepare_multi_domain_data(c.dataset, c.training); g=build_multi_domain_genome(b, c.genome); print(b.domain_num, b.domain_names, b.metadata['domain_counts'], b.sparse_feature_names, b.dense_feature_names, g.metadata.extras['flat_input_dim'])"
```

MovieLens config/data smoke:

```bash
conda run -n rechub python -c "from recskill.evolution.multi_domain_workflow import load_multi_domain_workflow_config, prepare_multi_domain_data, build_multi_domain_genome; c=load_multi_domain_workflow_config('examples/evolution/multi_domain_movielens_evolution.yaml'); c.dataset.limit_rows=3000; c.training.batch_size=256; b=prepare_multi_domain_data(c.dataset, c.training); g=build_multi_domain_genome(b, c.genome); print(b.domain_num, b.domain_names, b.metadata['domain_counts'], b.sparse_feature_names, b.dense_feature_names, g.metadata.extras['flat_input_dim'])"
```

## Result Review and Recovery

Use validation AUC (`validation_best_auc`) for survivor selection. Test AUC and
logloss are reporting metrics. Preserve label definitions, domain mappings,
feature handling, and split semantics across comparisons.

Inspect `round_history.tsv` and `results.tsv` after each run. Review enabled
Code-space diagnostics when proposals fail, and inspect the run log if training
crashes or stalls. Record failed candidates and skipped duplicates. Apply small
runtime fixes and retry recoverable failures; record any batch-size or GPU-pool
changes. Stop and report missing data, an unavailable live provider, or a GPU
failure that persists after recovery attempts.

Keep outputs under `outputs/`; generated modules belong to
`recskill/generated_skills/`. Do not commit datasets, checkpoints, logs, or
experimental artifacts. Stay on the current branch and do not commit or push
unless the human requests it.
