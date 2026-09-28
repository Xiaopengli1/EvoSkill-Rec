from recskill.evolution import SkillGenome

from .helpers import base_genome


def test_load_save_clone_and_validate_genome(tmp_path):
    genome = base_genome()
    path = tmp_path / "genome.json"
    genome.save(path)

    loaded = SkillGenome.load(path)
    assert loaded.to_dict() == genome.to_dict()
    assert loaded.validate_graph() == []

    cloned = loaded.clone()
    assert cloned.metadata.genome_id != loaded.metadata.genome_id
    assert loaded.metadata.genome_id in cloned.metadata.parent_genome_ids

    roundtrip = SkillGenome.from_json(loaded.to_json())
    assert roundtrip.to_dict() == loaded.to_dict()
