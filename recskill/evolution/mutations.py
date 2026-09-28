from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .exceptions import GenomeValidationError, MutationValidationError
from .genome import GenomeMutation, MutationPlan, MutationResult, SkillEdge, SkillGenome, SkillNode, new_id, utc_now_iso
from .skill_library import SkillLibrary


class BaseMutation:
    mutation_type = "base"

    def __init__(self, skill_library: SkillLibrary | None = None) -> None:
        self.skill_library = skill_library or SkillLibrary.from_repo()

    def apply(self, genome: SkillGenome, plan: MutationPlan) -> MutationResult:
        try:
            mutated = self._apply(genome, plan)
            mutation = _build_mutation_record(genome, mutated, plan)
            mutated.record_mutation(mutation)
            mutated.assert_valid_graph()
            return MutationResult(success=True, genome=mutated, message="mutation applied", mutation=mutation)
        except Exception as exc:
            return MutationResult(success=False, genome=None, message=f"{exc.__class__.__name__}: {exc}")

    def _apply(self, genome: SkillGenome, plan: MutationPlan) -> SkillGenome:
        raise NotImplementedError


class AddSkillMutation(BaseMutation):
    mutation_type = "add_skill"

    def _apply(self, genome: SkillGenome, plan: MutationPlan) -> SkillGenome:
        node = _node_from_plan(plan, self.skill_library)
        if node.node_id in genome.node_ids():
            raise MutationValidationError(f"Node '{node.node_id}' already exists")
        mutated = genome.clone()
        mutated.nodes.append(node)
        mutated.edges.extend(_edges_with_new_node(plan.edges, node.node_id))
        return mutated


class RemoveSkillMutation(BaseMutation):
    mutation_type = "remove_skill"

    def _apply(self, genome: SkillGenome, plan: MutationPlan) -> SkillGenome:
        target = _require_target(plan)
        genome.get_node(target)
        mutated = genome.clone()
        mutated.nodes = [node for node in mutated.nodes if node.node_id != target]
        mutated.edges = [edge for edge in mutated.edges if edge.src_node_id != target and edge.dst_node_id != target]
        mutated.edges.extend(plan.edges)
        return mutated


class ReplaceSkillMutation(BaseMutation):
    mutation_type = "replace_skill"

    def _apply(self, genome: SkillGenome, plan: MutationPlan) -> SkillGenome:
        target = _require_target(plan)
        mutated = genome.clone()
        node = mutated.get_node(target)
        replacement = _node_from_plan(plan, self.skill_library, node_id=target)
        node.skill_id = replacement.skill_id
        node.skill_name = replacement.skill_name
        node.category = replacement.category
        node.params = replacement.params
        node.input_keys = replacement.input_keys or node.input_keys
        node.output_keys = replacement.output_keys or node.output_keys
        node.task_types = replacement.task_types
        node.source = replacement.source
        node.metadata = {**node.metadata, **replacement.metadata, "replaced_at": utc_now_iso()}
        if plan.edges:
            incident = {target}
            mutated.edges = [edge for edge in mutated.edges if edge.src_node_id not in incident and edge.dst_node_id not in incident]
            mutated.edges.extend(plan.edges)
        return mutated


class RewireSkillMutation(BaseMutation):
    mutation_type = "rewire_skill"

    def _apply(self, genome: SkillGenome, plan: MutationPlan) -> SkillGenome:
        mutated = genome.clone()
        for edge in plan.edges:
            if edge.src_node_id not in mutated.node_ids() or edge.dst_node_id not in mutated.node_ids():
                raise MutationValidationError(f"Rewire edge references missing node: {edge}")
        if plan.metadata.get("replace_edges"):
            mutated.edges = list(plan.edges)
            return mutated

        remove_edges = [_edge_from_any(item) for item in plan.metadata.get("remove_edges", [])]
        if remove_edges:
            remove_set = set(remove_edges)
            mutated.edges = [edge for edge in mutated.edges if edge not in remove_set]
        mutated.edges.extend(edge for edge in plan.edges if edge not in mutated.edges)
        return mutated


class HybridizeSkillMutation(BaseMutation):
    mutation_type = "hybridize_skill"

    def _apply(self, genome: SkillGenome, plan: MutationPlan) -> SkillGenome:
        mutated = genome.clone()
        donor_nodes: list[SkillNode] = list(plan.nodes)
        donor_edges: list[SkillEdge] = list(plan.edges)
        for donor in plan.parent_genomes:
            donor_genome = SkillGenome.from_dict(donor)
            donor_nodes.extend(donor_genome.nodes)
            donor_edges.extend(donor_genome.edges)

        id_map: dict[str, str] = {}
        existing = mutated.node_ids()
        for node in donor_nodes:
            node_copy = SkillNode.from_dict(asdict(node))
            old_id = node_copy.node_id
            if old_id in existing or old_id in id_map.values():
                node_copy.node_id = new_id(f"hybrid_{old_id}")
            id_map[old_id] = node_copy.node_id
            node_copy.metadata = {**node_copy.metadata, "hybridized_from": old_id}
            mutated.nodes.append(node_copy)
            existing.add(node_copy.node_id)

        for edge in donor_edges:
            src = id_map.get(edge.src_node_id, edge.src_node_id)
            dst = id_map.get(edge.dst_node_id, edge.dst_node_id)
            if src in existing and dst in existing:
                mutated.edges.append(
                    SkillEdge(
                        src_node_id=src,
                        dst_node_id=dst,
                        src_output_key=edge.src_output_key,
                        dst_input_key=edge.dst_input_key,
                        tensor_semantics=edge.tensor_semantics,
                    )
                )
        return mutated


class SpecializeSkillMutation(AddSkillMutation):
    mutation_type = "specialize_skill"

    def _apply(self, genome: SkillGenome, plan: MutationPlan) -> SkillGenome:
        plan = MutationPlan.from_dict(plan.to_dict())
        plan.metadata = {**plan.metadata, "specialization": True}
        if "source" not in plan.metadata:
            plan.metadata["source"] = "modified_skill"
        return super()._apply(genome, plan)


MUTATION_OPERATORS = {
    "ADD_SKILL": AddSkillMutation,
    "add_skill": AddSkillMutation,
    "REMOVE_SKILL": RemoveSkillMutation,
    "remove_skill": RemoveSkillMutation,
    "REPLACE_SKILL": ReplaceSkillMutation,
    "replace_skill": ReplaceSkillMutation,
    "REWIRE_SKILL": RewireSkillMutation,
    "rewire_skill": RewireSkillMutation,
    "HYBRIDIZE_SKILL": HybridizeSkillMutation,
    "hybridize_skill": HybridizeSkillMutation,
    "SPECIALIZE_SKILL": SpecializeSkillMutation,
    "specialize_skill": SpecializeSkillMutation,
}


def apply_mutation(genome: SkillGenome, plan: MutationPlan, skill_library: SkillLibrary | None = None) -> MutationResult:
    operator_cls = MUTATION_OPERATORS.get(plan.mutation_type)
    if operator_cls is None:
        return MutationResult(success=False, message=f"Unknown mutation_type: {plan.mutation_type}")
    return operator_cls(skill_library=skill_library).apply(genome, plan)


def _node_from_plan(plan: MutationPlan, skill_library: SkillLibrary, node_id: str | None = None) -> SkillNode:
    if plan.nodes and node_id is None:
        return SkillNode.from_dict(asdict(plan.nodes[0]))
    skill_id = plan.skill_id or plan.skill_name
    if not skill_id:
        raise MutationValidationError("Mutation plan must include skill_id or nodes")
    if skill_library.has(skill_id):
        node = skill_library.build_node(
            skill_id,
            node_id=node_id or plan.metadata.get("node_id"),
            params=plan.params,
            source=plan.metadata.get("source", "existing_skill"),
            metadata=plan.metadata.get("node_metadata", {}),
        )
    else:
        if plan.category is None:
            raise MutationValidationError(f"Unknown skill '{skill_id}' and no category was supplied")
        node = SkillNode(
            node_id=node_id or plan.metadata.get("node_id") or new_id("node"),
            skill_id=skill_id,
            skill_name=plan.skill_name or skill_id,
            category=plan.category,
            params=dict(plan.params),
            input_keys=list(plan.input_keys),
            output_keys=list(plan.output_keys),
            task_types=list(plan.task_types),
            source=plan.metadata.get("source", "generated_skill"),
            metadata=dict(plan.metadata.get("node_metadata", {})),
        )
    if plan.input_keys:
        node.input_keys = list(plan.input_keys)
    if plan.output_keys:
        node.output_keys = list(plan.output_keys)
    if plan.task_types:
        node.task_types = list(plan.task_types)
    if plan.category:
        node.category = plan.category
    return node


def _edges_with_new_node(edges: list[SkillEdge], node_id: str) -> list[SkillEdge]:
    return [
        SkillEdge(
            src_node_id=node_id if edge.src_node_id in {"NEW_NODE", "$new_node", "${new_node_id}"} else edge.src_node_id,
            dst_node_id=node_id if edge.dst_node_id in {"NEW_NODE", "$new_node", "${new_node_id}"} else edge.dst_node_id,
            src_output_key=edge.src_output_key,
            dst_input_key=edge.dst_input_key,
            tensor_semantics=edge.tensor_semantics,
        )
        for edge in edges
    ]


def _edge_from_any(value: Any) -> SkillEdge:
    if isinstance(value, SkillEdge):
        return value
    if isinstance(value, dict):
        return SkillEdge.from_dict(value)
    raise TypeError(f"Cannot parse edge from {type(value).__name__}")


def _require_target(plan: MutationPlan) -> str:
    if not plan.target_node_id:
        raise MutationValidationError("Mutation plan must include target_node_id")
    return plan.target_node_id


def _build_mutation_record(parent: SkillGenome, child: SkillGenome, plan: MutationPlan) -> GenomeMutation:
    return GenomeMutation(
        mutation_type=plan.mutation_type,
        parent_genome_id=parent.metadata.genome_id,
        child_genome_id=child.metadata.genome_id,
        plan_id=plan.plan_id,
        description=plan.rationale,
        details=plan.to_dict(),
    )
