from recskill.evolution import EvolutionMemory, EvolutionRecord


def test_evolution_memory_records_and_retrieves_attempts(tmp_path):
    memory = EvolutionMemory(tmp_path / "memory.jsonl")
    memory.append(
        EvolutionRecord(
            parent_genome_id="g0",
            child_genome_id="g1",
            mutation_type="add_skill",
            status="success",
            failure_mode="scenario_gap",
            task_type="ctr",
            artifact_paths={"skill_card": "card.yaml"},
        )
    )
    memory.append(
        EvolutionRecord(
            parent_genome_id="g1",
            child_genome_id="g2",
            mutation_type="NEW_SKILL_INVENTION",
            status="failed",
            failure_mode="scenario_gap",
            task_type="ctr",
        )
    )

    assert len(memory.get_successful_mutations(failure_mode="scenario_gap", task_type="ctr")) == 1
    assert len(memory.get_failed_mutations(failure_mode="scenario_gap", task_type="ctr")) == 1
    assert len(memory.get_generated_skills(task_type="ctr")) == 1
    assert [record.child_genome_id for record in memory.get_lineage("g2")] == ["g1", "g2"]
