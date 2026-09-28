from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from recskill.core import BaseSkill, SkillContext, build_skill

from .genome import SkillGenome
from .skill_library import SkillLibrary


class SkillModule(torch.nn.Module):
    """Common Stage 2 skill interface for dict-in/dict-out modules."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class Stage1SkillAdapter(SkillModule):
    """Wrap a Stage 1 BaseSkill so the compiler can treat all nodes uniformly."""

    def __init__(self, module: BaseSkill) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = inputs if isinstance(inputs, SkillContext) else SkillContext(inputs)
        result = self.module(ctx)
        if not isinstance(result, SkillContext):
            raise TypeError(f"{self.module.__class__.__name__}.forward must return SkillContext")
        return result


class DictSkillAdapter(SkillModule):
    """Wrap a generic torch.nn.Module that accepts a tensor dict or single tensor."""

    def __init__(self, module: torch.nn.Module, input_keys: list[str], output_keys: list[str]) -> None:
        super().__init__()
        self.module = module
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        payload = {key: inputs[key] for key in self.input_keys} if self.input_keys else dict(inputs)
        try:
            result = self.module(payload)
        except TypeError:
            if len(payload) == 1:
                result = self.module(next(iter(payload.values())))
            else:
                raise
        if isinstance(result, dict):
            return result
        if not self.output_keys:
            raise TypeError("Generic skill returned a tensor but no output_keys were declared")
        return {self.output_keys[0]: result}


class CompiledSkillGenomeModel(torch.nn.Module):
    """Executable DAG model compiled from a SkillGenome."""

    def __init__(self, genome: SkillGenome, modules: dict[str, SkillModule]) -> None:
        super().__init__()
        self.genome = genome
        self.node_order = genome.topological_order()
        self.modules_by_node = torch.nn.ModuleDict(modules)
        self.incoming_edges = {
            node.node_id: [edge for edge in genome.edges if edge.dst_node_id == node.node_id]
            for node in genome.nodes
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> SkillContext:
        ctx = SkillContext(batch)
        for node_id in self.node_order:
            for edge in self.incoming_edges.get(node_id, []):
                if edge.src_output_key in ctx:
                    ctx.put(edge.dst_input_key, ctx[edge.src_output_key])
            outputs = self.modules_by_node[node_id](ctx)
            for key, value in outputs.items():
                ctx.put(key, value)
        return ctx


class SkillGenomeCompiler:
    """Compile Stage 2 DAG genomes using Stage 1 registry and generated skill cards."""

    def __init__(self, skill_library: SkillLibrary | None = None) -> None:
        self.skill_library = skill_library or SkillLibrary.from_repo()

    def compile(self, genome: SkillGenome) -> CompiledSkillGenomeModel:
        genome.assert_valid_graph()
        modules = {}
        for node in genome.nodes:
            module = self._instantiate_node(node.skill_id, node.params, node.input_keys, node.output_keys)
            modules[node.node_id] = module
        return CompiledSkillGenomeModel(genome, modules)

    def _instantiate_node(self, skill_id: str, params: dict[str, Any], input_keys: list[str], output_keys: list[str]) -> SkillModule:
        try:
            module = build_skill(skill_id, **params)
            return Stage1SkillAdapter(module)
        except KeyError:
            self.skill_library.register_generated_skill(skill_id)
            module = build_skill(skill_id, **params)
            return Stage1SkillAdapter(module)
        except TypeError:
            raise


def compile_genome(genome: SkillGenome, skill_library: SkillLibrary | None = None) -> CompiledSkillGenomeModel:
    return SkillGenomeCompiler(skill_library=skill_library).compile(genome)
