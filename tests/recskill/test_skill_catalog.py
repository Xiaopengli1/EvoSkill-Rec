from pathlib import Path

import yaml

from recskill.evolution import assert_model_skill_coverage, assert_torch_rechub_skill_implementations
from tools.build_skill_index import build_skill_catalog, build_skill_index
from tools.search_skills import search_catalog


ROOT = Path(__file__).resolve().parents[2]


def test_every_skill_manifest_has_retrieval_and_composition_metadata():
    index = build_skill_index(ROOT / "recskill" / "skills", repo_root=ROOT)

    for name, manifest in index.items():
        retrieval = manifest.get("retrieval")
        composition = manifest.get("composition")
        assert retrieval, f"{name} missing retrieval metadata"
        assert composition, f"{name} missing composition metadata"
        for key in ["summary", "use_when", "avoid_when", "task_types", "architecture_roles", "input_modalities", "objectives", "aliases"]:
            assert retrieval.get(key), f"{name} retrieval.{key} is empty"
        for key in ["requires", "produces", "common_upstream", "common_downstream", "example_genome_fragment"]:
            assert composition.get(key), f"{name} composition.{key} is empty"


def test_skill_catalog_contains_searchable_facets():
    index = build_skill_index(ROOT / "recskill" / "skills", repo_root=ROOT)
    catalog = build_skill_catalog(index)

    assert len(catalog) == len(index)
    assert "explicit_cross" in catalog["crossnet_v1"]["architecture_roles"]
    assert "target_attention" in catalog["din_target_attention"]["architecture_roles"]
    assert "SASRec" in catalog["sasrec_sequence_encoder"]["model_families"]
    assert "candidate_scoring" in catalog["candidate_dot_logits"]["architecture_roles"]
    assert "NARM" in catalog["narm_session_encoder"]["model_families"]
    assert "task_gate" in catalog["mmoe_gate"]["architecture_roles"]
    assert catalog["field_embedding"]["searchable_text"]


def test_search_catalog_recalls_skills_by_task_purpose_and_composition():
    index = build_skill_index(ROOT / "recskill" / "skills", repo_root=ROOT)
    catalog = build_skill_catalog(index)

    cross_results = search_catalog(catalog, task="ctr", role="explicit_cross")
    assert "crossnet_v1" in [entry["name"] for _, entry in cross_results]

    attention_results = search_catalog(catalog, query="target attention", task="ranking")
    assert "din_target_attention" in [entry["name"] for _, entry in attention_results]

    sasrec_results = search_catalog(catalog, model="SASRec")
    assert {"shared_sequence_embedding", "sasrec_sequence_encoder", "sasrec_pairwise_logits"}.issubset(
        {entry["name"] for _, entry in sasrec_results}
    )

    downstream_results = search_catalog(catalog, upstream="field_embedding", limit=50)
    assert {"flatten_field_embeddings", "fm_interaction"}.issubset({entry["name"] for _, entry in downstream_results})


def test_model_coverage_tracks_claimed_genomes():
    coverage = yaml.safe_load((ROOT / "recskill" / "model_coverage.yaml").read_text(encoding="utf-8"))
    models = coverage["models"]
    by_name = {model["name"]: model for model in models}

    for model_name in ["DeepFM", "DCN", "DIN", "SASRec", "MMOE"]:
        assert by_name[model_name]["status"] in {"L1", "L2", "L3", "L4", "L5"}
        assert (ROOT / by_name[model_name]["genome"]).exists()
        assert by_name[model_name]["tests"]

    for model in models:
        if model["domain"] != "generative":
            assert model["status"] in {"L1", "L2", "L3", "L4", "L5"}, model["name"]
            assert (ROOT / model["genome"]).exists(), model["name"]
            assert model["covered_components"], model["name"]

    for model in models:
        if model["status"] == "L0":
            assert model["genome"] is None
        else:
            assert model["covered_components"], model["name"]


def test_model_skill_coverage_audit_passes():
    assert_model_skill_coverage(ROOT)


def test_torch_rechub_model_abilities_have_skill_implementations():
    assert_torch_rechub_skill_implementations(ROOT)
