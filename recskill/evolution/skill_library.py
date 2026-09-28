from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from recskill.core import BaseSkill, SkillContext, SkillSpec
from recskill.core.registry import SKILL_REGISTRY, register_skill
from recskill.core.spec import TensorSpec as CoreTensorSpec

from .genome import SkillNode, new_id


@dataclass
class SkillCard:
    skill_id: str
    skill_name: str
    category: str
    manifest: dict[str, Any]
    path: Path | None = None

    @property
    def task_types(self) -> list[str]:
        return list(self.manifest.get("task_types") or [])

    @property
    def implementation_path(self) -> str | None:
        return self.manifest.get("implementation_path") or (self.manifest.get("source") or {}).get("file")

    @property
    def class_name(self) -> str | None:
        return self.manifest.get("class_name") or (self.manifest.get("source") or {}).get("class_or_function")

    @property
    def input_keys(self) -> list[str]:
        return _io_keys_from_manifest(self.manifest, direction="input")

    @property
    def output_keys(self) -> list[str]:
        return _io_keys_from_manifest(self.manifest, direction="output")


class SkillLibrary:
    """Stage 2 view over Stage 1 YAML skill cards and runtime registry."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        skill_roots: list[str | Path] | None = None,
        include_generated: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        self.include_generated = include_generated
        default_roots = [self.repo_root / "recskill" / "skills"]
        if include_generated:
            default_roots.append(self.repo_root / "recskill" / "generated_skills")
        self.skill_roots = [Path(path) for path in (skill_roots or default_roots)]
        self.cards: dict[str, SkillCard] = {}
        self.load_errors: list[dict[str, str]] = []
        self.reload()

    @classmethod
    def from_repo(cls, repo_root: str | Path | None = None, include_generated: bool = True) -> "SkillLibrary":
        return cls(repo_root=repo_root, include_generated=include_generated)

    def reload(self) -> None:
        self.cards = {}
        self.load_errors = []
        for root in self.skill_roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("**/*.skill.yaml")):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        manifest = yaml.safe_load(f) or {}
                    if not isinstance(manifest, dict):
                        raise TypeError(f"Skill card manifest must be a mapping: {path}")
                    skill_id = manifest.get("skill_id") or manifest.get("name")
                    if not skill_id:
                        raise ValueError(f"Skill card missing skill_id/name: {path}")
                except Exception as exc:
                    self.load_errors.append({"path": str(path), "error": f"{exc.__class__.__name__}: {exc}"})
                    continue
                self.cards[skill_id] = SkillCard(
                    skill_id=skill_id,
                    skill_name=manifest.get("skill_name") or manifest.get("name") or skill_id,
                    category=manifest.get("category", "unknown"),
                    manifest=manifest,
                    path=path,
                )
        self._add_registered_skill_specs()

    def has(self, skill_id: str) -> bool:
        return skill_id in self.cards or self._registered_skill_visible(skill_id)

    def get(self, skill_id: str) -> SkillCard:
        if skill_id in self.cards:
            return self.cards[skill_id]
        if self._registered_skill_visible(skill_id):
            return _card_from_registered_skill(skill_id)
        available = ", ".join(self.list_skill_ids()[:20])
        raise KeyError(f"Unknown skill '{skill_id}'. Available examples: {available}")

    def list_skill_ids(self) -> list[str]:
        registered = {skill_id for skill_id in SKILL_REGISTRY if self._registered_skill_visible(skill_id)}
        return sorted(set(self.cards) | registered)

    def build_node(
        self,
        skill_id: str,
        *,
        node_id: str | None = None,
        params: dict[str, Any] | None = None,
        source: str = "existing_skill",
        metadata: dict[str, Any] | None = None,
    ) -> SkillNode:
        params = dict(params or {})
        card = self.get(skill_id)
        return SkillNode(
            node_id=node_id or new_id("node"),
            skill_id=skill_id,
            skill_name=card.skill_name,
            category=card.category,
            params=params,
            input_keys=_resolve_parametric_keys(card.input_keys, params, direction="input"),
            output_keys=_resolve_parametric_keys(card.output_keys, params, direction="output"),
            task_types=card.task_types,
            source=source,
            metadata=dict(metadata or {}),
        )

    def add_card(self, card_path: str | Path) -> SkillCard:
        path = Path(card_path)
        with path.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        skill_id = manifest.get("skill_id") or manifest.get("name")
        if not skill_id:
            raise ValueError(f"Skill card missing skill_id/name: {path}")
        card = SkillCard(
            skill_id=skill_id,
            skill_name=manifest.get("skill_name") or manifest.get("name") or skill_id,
            category=manifest.get("category", "generated"),
            manifest=manifest,
            path=path,
        )
        self.cards[skill_id] = card
        return card

    def register_generated_skill(self, skill_id: str) -> type[BaseSkill]:
        card = self.get(skill_id)
        return register_generated_skill_from_card(card, repo_root=self.repo_root)

    def _add_registered_skill_specs(self) -> None:
        for skill_id in SKILL_REGISTRY:
            if skill_id not in self.cards:
                if not self._registered_skill_visible(skill_id):
                    continue
                self.cards[skill_id] = _card_from_registered_skill(skill_id)

    def _registered_skill_visible(self, skill_id: str) -> bool:
        if skill_id not in SKILL_REGISTRY:
            return False
        return self.include_generated or not _registered_skill_looks_generated(skill_id)


def register_generated_skill_from_card(card: SkillCard, repo_root: str | Path | None = None) -> type[BaseSkill]:
    """Register a generated skill card with Stage 1's decorator registry."""
    if card.skill_id in SKILL_REGISTRY:
        return SKILL_REGISTRY[card.skill_id]
    implementation_path = card.implementation_path
    class_name = card.class_name
    if not implementation_path or not class_name:
        raise ValueError(f"Generated skill '{card.skill_id}' is missing implementation_path or class_name")

    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    module_path = Path(implementation_path)
    if not module_path.is_absolute():
        if card.path is not None:
            for base in [card.path.parent, card.path.parent.parent]:
                candidate = base / module_path
                if candidate.exists():
                    module_path = candidate
                    break
            else:
                module_path = repo_root / module_path
        else:
            module_path = repo_root / module_path
    module = load_module_from_path(module_path)
    impl_cls = getattr(module, class_name)
    if isinstance(impl_cls, type) and issubclass(impl_cls, BaseSkill):
        return register_skill(card.skill_id)(impl_cls)

    adapter_cls = build_stage1_adapter_class(card, impl_cls)
    return register_skill(card.skill_id)(adapter_cls)


def build_stage1_adapter_class(card: SkillCard, impl_cls: type) -> type[BaseSkill]:
    input_keys = list(card.input_keys)
    output_keys = list(card.output_keys)
    skill_spec = SkillSpec(
        name=card.skill_id,
        category=card.category,
        description=card.manifest.get("function_description") or card.manifest.get("description", ""),
        input_specs=[CoreTensorSpec(item.get("name", key), str(item.get("shape", "")), item.get("dtype", "float32")) for key, item in _signature_items(card.manifest, "input_signature", input_keys)],
        output_specs=[CoreTensorSpec(item.get("name", key), str(item.get("shape", "")), item.get("dtype", "float32")) for key, item in _signature_items(card.manifest, "output_signature", output_keys)],
        task_types=list(card.manifest.get("applicable_tasks") or card.task_types),
        inductive_bias=_as_list(card.manifest.get("inductive_bias")),
        failure_signatures=_as_list(card.manifest.get("failure_modes_addressed")),
        implementation=card.implementation_path,
    )

    class GeneratedSkillAdapter(BaseSkill):
        pass

    GeneratedSkillAdapter.__name__ = f"{_camel(card.skill_id)}Adapter"
    GeneratedSkillAdapter.skill_spec = skill_spec

    def __init__(self, **kwargs: Any) -> None:
        BaseSkill.__init__(self)
        self.input_keys = _runtime_io_keys(input_keys, kwargs, direction="input")
        self.output_keys = _runtime_io_keys(output_keys, kwargs, direction="output")
        self.impl = impl_cls(**_constructor_kwargs(impl_cls, kwargs))

    def forward(self, ctx: SkillContext) -> SkillContext:
        payload = {key: ctx.get_required(key) for key in self.input_keys} if self.input_keys else dict(ctx)
        try:
            result = self.impl(payload)
        except TypeError:
            if len(payload) == 1:
                result = self.impl(next(iter(payload.values())))
            else:
                raise
        if isinstance(result, SkillContext):
            return result
        if isinstance(result, dict):
            for key, value in result.items():
                ctx.put(key, value)
            return ctx
        if not self.output_keys:
            raise TypeError(f"Generated skill '{card.skill_id}' returned a tensor but has no output_signature")
        return ctx.put(self.output_keys[0], result)

    GeneratedSkillAdapter.__init__ = __init__  # type: ignore[method-assign]
    GeneratedSkillAdapter.forward = forward  # type: ignore[method-assign]
    GeneratedSkillAdapter.__abstractmethods__ = frozenset()
    return GeneratedSkillAdapter


def _constructor_kwargs(impl_cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Return only params accepted by a generated module constructor."""
    try:
        signature = inspect.signature(impl_cls.__init__)
    except (TypeError, ValueError):
        return dict(params)
    accepted: set[str] = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return dict(params)
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            accepted.add(name)
    return {key: value for key, value in params.items() if key in accepted}


def load_module_from_path(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"recskill_generated_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def _card_from_registered_skill(skill_id: str) -> SkillCard:
    cls = SKILL_REGISTRY[skill_id]
    spec = getattr(cls, "skill_spec", None)
    manifest = {
        "name": skill_id,
        "skill_id": skill_id,
        "skill_name": getattr(spec, "name", skill_id),
        "category": getattr(spec, "category", "unknown"),
        "description": getattr(spec, "description", ""),
        "inputs": [vars(item) for item in getattr(spec, "input_specs", [])],
        "outputs": [vars(item) for item in getattr(spec, "output_specs", [])],
        "task_types": list(getattr(spec, "task_types", []) or []),
        "inductive_bias": list(getattr(spec, "inductive_bias", []) or []),
        "failure_signatures": list(getattr(spec, "failure_signatures", []) or []),
        "implementation_path": getattr(spec, "implementation", None),
        "class_name": cls.__name__,
    }
    return SkillCard(skill_id=skill_id, skill_name=manifest["skill_name"], category=manifest["category"], manifest=manifest)


def _registered_skill_looks_generated(skill_id: str) -> bool:
    cls = SKILL_REGISTRY[skill_id]
    spec = getattr(cls, "skill_spec", None)
    implementation = str(getattr(spec, "implementation", "") or "")
    category = str(getattr(spec, "category", "") or "").lower()
    return (
        skill_id.startswith(("generated_", "mtl_r"))
        or "generated_skills" in implementation.replace("\\", "/")
        or category == "generated"
    )


def _io_keys_from_manifest(manifest: dict[str, Any], *, direction: str) -> list[str]:
    composition = manifest.get("composition") or {}
    if direction == "input":
        keys = _as_list(composition.get("requires"))
        fallback_field = "inputs"
        signature_field = "input_signature"
    else:
        keys = _as_list(composition.get("produces"))
        fallback_field = "outputs"
        signature_field = "output_signature"
    if keys:
        return [str(key) for key in keys]
    signature = manifest.get(signature_field)
    if signature:
        return [str(item.get("name")) for item in signature if item.get("name")]
    return [str(item.get("name")) for item in manifest.get(fallback_field, []) if item.get("name")]


def _resolve_parametric_keys(keys: list[str], params: dict[str, Any], *, direction: str) -> list[str]:
    resolved: list[str] = []
    for key in keys:
        if key == "input_key" and "input_key" in params:
            resolved.append(str(params["input_key"]))
        elif key == "output_key" and "output_key" in params:
            resolved.append(str(params["output_key"]))
        elif key == "input_keys" and "input_keys" in params:
            resolved.extend(str(item) for item in params["input_keys"])
        elif key == "output_keys" and "output_keys" in params:
            resolved.extend(str(item) for item in params["output_keys"])
        elif key.endswith("_key") and key in params:
            resolved.append(str(params[key]))
        elif key.endswith("_keys") and key in params:
            value = params[key]
            if isinstance(value, list):
                resolved.extend(str(item) for item in value)
            else:
                resolved.append(str(value))
        else:
            resolved.append(str(key))
    if direction == "output" and not resolved and "output_key" in params:
        resolved.append(str(params["output_key"]))
    return list(dict.fromkeys(resolved))


def _runtime_io_keys(keys: list[str], params: dict[str, Any], *, direction: str) -> list[str]:
    if direction == "input":
        if "input_keys" in params and isinstance(params["input_keys"], list):
            return [str(item) for item in params["input_keys"]]
        if "input_key" in params and len(keys) <= 1:
            return [str(params["input_key"])]
    else:
        if "output_keys" in params and isinstance(params["output_keys"], list):
            return [str(item) for item in params["output_keys"]]
        if "output_key" in params and len(keys) <= 1:
            return [str(params["output_key"])]
    return _resolve_parametric_keys(keys, params, direction=direction)


def _signature_items(manifest: dict[str, Any], field: str, keys: list[str]) -> list[tuple[str, dict[str, Any]]]:
    signature = manifest.get(field) or []
    if signature:
        return [(str(item.get("name")), item) for item in signature if item.get("name")]
    return [(key, {"name": key, "shape": [], "dtype": "float32"}) for key in keys]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _camel(value: str) -> str:
    return "".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)
