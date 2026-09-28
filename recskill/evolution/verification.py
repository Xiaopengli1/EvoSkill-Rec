from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from .exceptions import GenomeValidationError, StaticCodeError
from .genome import SkillGenome, utc_now_iso
from .skill_library import SkillCard, SkillLibrary, load_module_from_path


@dataclass
class TensorSpec:
    name: str
    shape: list[str | int] = field(default_factory=list)
    dtype: str = "float32"
    semantic: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorSpec":
        return cls(
            name=data["name"],
            shape=_parse_shape(data.get("shape", [])),
            dtype=data.get("dtype", "float32"),
            semantic=data.get("semantic") or data.get("description"),
        )

    @property
    def rank(self) -> int | None:
        return len(self.shape) if self.shape else None


@dataclass
class SkillSignature:
    input_signature: list[TensorSpec] = field(default_factory=list)
    output_signature: list[TensorSpec] = field(default_factory=list)
    class_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_signature": [asdict(item) for item in self.input_signature],
            "output_signature": [asdict(item) for item in self.output_signature],
            "class_name": self.class_name,
        }


class GenomeVerifier:
    """Structural and lightweight semantic verifier for Stage 2 DAG genomes."""

    def __init__(self, skill_library: SkillLibrary | None = None) -> None:
        self.skill_library = skill_library or SkillLibrary.from_repo()

    def verify(self, genome: SkillGenome) -> dict[str, Any]:
        issues = genome.validate_graph()
        issues.extend(self._verify_skill_references(genome))
        issues.extend(self._verify_task_compatibility(genome))
        issues.extend(self._verify_dataflow(genome))
        return {"valid": not issues, "issues": issues}

    def assert_valid(self, genome: SkillGenome) -> None:
        result = self.verify(genome)
        if not result["valid"]:
            raise GenomeValidationError("; ".join(result["issues"]))

    def _verify_skill_references(self, genome: SkillGenome) -> list[str]:
        issues = []
        for node in genome.nodes:
            if node.source == "existing_skill" and not self.skill_library.has(node.skill_id):
                issues.append(f"Node {node.node_id} references unknown existing skill '{node.skill_id}'")
        return issues

    def _verify_task_compatibility(self, genome: SkillGenome) -> list[str]:
        issues = []
        required_tasks = set(genome.constraints.task_types)
        if not required_tasks:
            return issues
        for node in genome.nodes:
            node_tasks = set(node.task_types)
            if node_tasks and not (required_tasks & node_tasks):
                issues.append(f"Node {node.node_id} task_types {sorted(node_tasks)} do not match constraints {sorted(required_tasks)}")
        return issues

    def _verify_dataflow(self, genome: SkillGenome) -> list[str]:
        issues: list[str] = []
        try:
            order = genome.topological_order()
        except GenomeValidationError as exc:
            return [str(exc)]
        node_by_id = {node.node_id: node for node in genome.nodes}
        incoming_edges = {
            node.node_id: [edge for edge in genome.edges if edge.dst_node_id == node.node_id]
            for node in genome.nodes
        }
        available = set(genome.constraints.required_inputs)
        for node_id in order:
            for edge in incoming_edges.get(node_id, []):
                if edge.src_output_key in available:
                    available.add(edge.dst_input_key)
            node = node_by_id[node_id]
            missing = [key for key in node.input_keys if key not in available]
            if missing:
                issues.append(f"Node {node.node_id} missing required inputs: {missing}")
            available.update(node.output_keys)
        missing_outputs = sorted(set(genome.constraints.required_outputs) - available)
        if missing_outputs:
            issues.append(f"Genome missing required outputs after dataflow: {missing_outputs}")
        return issues


class StaticCodeChecker:
    """AST-based safety checker for generated skill code."""

    def __init__(
        self,
        allowed_imports: set[str] | None = None,
        banned_imports: set[str] | None = None,
        banned_calls: set[str] | None = None,
    ) -> None:
        self.allowed_imports = allowed_imports or {
            "__future__",
            "math",
            "typing",
            "dataclasses",
            "collections",
            "collections.abc",
            "torch",
            "torch.nn",
            "torch.nn.functional",
        }
        self.banned_imports = banned_imports or {
            "os",
            "sys",
            "subprocess",
            "socket",
            "requests",
            "urllib",
            "http",
            "ftplib",
            "shutil",
            "pathlib",
        }
        self.banned_calls = banned_calls or {
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",
            "os.system",
            "os.popen",
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "os.getenv",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
            "subprocess.run",
            "shutil.rmtree",
            "Path.unlink",
            "Path.rmdir",
        }

    def check(self, code: str) -> dict[str, Any]:
        issues: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {"passed": False, "issues": [f"SyntaxError: {exc}"]}

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                issues.extend(self._check_import(node))
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                if (
                    call_name in self.banned_calls
                    or call_name.split(".")[0] in self.banned_imports
                    or call_name.endswith((".unlink", ".rmdir", ".remove", ".rmtree"))
                ):
                    issues.append(f"Dangerous call rejected: {call_name}")
            elif isinstance(node, ast.Attribute):
                name = _call_name(node)
                if name in {"os.environ", "environ"}:
                    issues.append("Environment variable access rejected")
        return {"passed": not issues, "issues": issues}

    def assert_safe(self, code: str) -> None:
        result = self.check(code)
        if not result["passed"]:
            raise StaticCodeError("; ".join(result["issues"]))

    def _check_import(self, node: ast.Import | ast.ImportFrom) -> list[str]:
        issues = []
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in self.banned_imports:
                issues.append(f"Import rejected: {name}")
            elif not any(name == allowed or name.startswith(f"{allowed}.") or root == allowed for allowed in self.allowed_imports):
                issues.append(f"Import not allowlisted: {name}")
        return issues


class SignatureInferer:
    """Infer minimal generated-skill signatures from proposals or Python code."""

    def infer_from_proposal(self, proposal: Any) -> SkillSignature:
        input_signature = [TensorSpec.from_dict(item) for item in getattr(proposal, "expected_input_signature", [])]
        output_signature = [TensorSpec.from_dict(item) for item in getattr(proposal, "expected_output_signature", [])]
        code_signature = self.infer_from_code(getattr(proposal, "code", ""))
        return SkillSignature(
            input_signature=input_signature or code_signature.input_signature,
            output_signature=output_signature or code_signature.output_signature,
            class_name=getattr(proposal, "class_name", None) or code_signature.class_name,
        )

    def infer_from_code(self, code: str) -> SkillSignature:
        tree = ast.parse(code)
        class_name = None
        input_keys: set[str] = set()
        output_keys: set[str] = set()
        for node in ast.walk(tree):
            if class_name is None and isinstance(node, ast.ClassDef):
                class_name = node.name
            if isinstance(node, ast.Subscript):
                key = _literal_subscript_key(node)
                if key and _call_name(node.value) in {"inputs", "batch", "x"}:
                    input_keys.add(key)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                for key_node in node.value.keys:
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        output_keys.add(key_node.value)
        return SkillSignature(
            input_signature=[TensorSpec(name=key, shape=[]) for key in sorted(input_keys)],
            output_signature=[TensorSpec(name=key, shape=[]) for key in sorted(output_keys)],
            class_name=class_name,
        )


class SkillUnitTestRunner:
    """Run import, instantiate, forward, and optional backward checks."""

    def __init__(self, static_checker: StaticCodeChecker | None = None) -> None:
        self.static_checker = static_checker or StaticCodeChecker()

    def run(
        self,
        code_path: str | Path,
        class_name: str,
        input_signature: list[TensorSpec],
        output_signature: list[TensorSpec],
        init_params: dict[str, Any] | None = None,
        unit_test_code: str | None = None,
    ) -> dict[str, Any]:
        results = {"passed": False, "checks": {}, "issues": []}
        try:
            module = load_module_from_path(Path(code_path))
            cls = getattr(module, class_name)
            skill = cls(**(init_params or {}))
            results["checks"]["import"] = True
            results["checks"]["instantiate"] = True
            payload = _synthetic_payload(input_signature, init_params)
            output = _call_skill(skill, payload)
            if not isinstance(output, dict):
                raise TypeError("forward must return a dict[str, torch.Tensor]")
            expected_outputs = {spec.name for spec in output_signature}
            missing = sorted(expected_outputs - set(output))
            if missing:
                raise AssertionError(f"Missing expected outputs: {missing}")
            results["checks"]["forward"] = True
            floating_outputs = [value for value in output.values() if torch.is_tensor(value) and value.is_floating_point()]
            trainable = hasattr(skill, "parameters") and any(param.requires_grad for param in skill.parameters())
            if trainable and floating_outputs:
                loss = sum(value.sum() for value in floating_outputs)
                if loss.requires_grad:
                    loss.backward()
                results["checks"]["backward"] = True
            if unit_test_code:
                self._run_pytest_snippet(unit_test_code, Path(code_path), class_name)
                results["checks"]["pytest_snippet"] = True
            results["passed"] = True
        except Exception as exc:
            results["issues"].append(f"{exc.__class__.__name__}: {exc}")
        return results

    def _run_pytest_snippet(self, code: str, code_path: Path, class_name: str) -> None:
        self.static_checker.assert_safe(code)
        test_path = code_path.parent / f"test_{code_path.stem}.py"
        test_path.write_text(code, encoding="utf-8")
        env = os.environ.copy()
        env["GENERATED_SKILL_PATH"] = str(code_path)
        env["GENERATED_SKILL_CLASS"] = class_name
        proc = subprocess.run([sys.executable, "-m", "pytest", str(test_path), "-q"], text=True, capture_output=True, env=env, check=False)
        if proc.returncode != 0:
            raise AssertionError(proc.stdout + proc.stderr)


class ShapeTestRunner:
    """Run deterministic synthetic shape checks for generated skills."""

    def run(
        self,
        code_path: str | Path,
        class_name: str,
        input_signature: list[TensorSpec],
        output_signature: list[TensorSpec],
        init_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        results = {"passed": False, "output_shapes": {}, "issues": []}
        try:
            module = load_module_from_path(Path(code_path))
            cls = getattr(module, class_name)
            skill = cls(**(init_params or {}))
            if hasattr(skill, "eval"):
                skill.eval()
            payload = _synthetic_payload(input_signature, init_params)
            with torch.no_grad():
                first = _call_skill(skill, payload)
                second = _call_skill(skill, payload)
            if set(first) != set(second):
                raise AssertionError("Output keys are not deterministic")
            for spec in output_signature:
                if spec.name not in first:
                    raise AssertionError(f"Missing output '{spec.name}'")
                value = first[spec.name]
                if not torch.is_tensor(value):
                    raise TypeError(f"Output '{spec.name}' is not a tensor")
                shape = list(value.shape)
                results["output_shapes"][spec.name] = shape
                expected = _resolve_shape(spec.shape)
                if expected and len(expected) == len(shape):
                    for expected_dim, actual_dim in zip(expected, shape):
                        if isinstance(expected_dim, int) and expected_dim != actual_dim:
                            raise AssertionError(f"Output '{spec.name}' shape {shape} does not match expected {expected}")
            results["passed"] = True
        except Exception as exc:
            results["issues"].append(f"{exc.__class__.__name__}: {exc}")
        return results


class SkillCardConsolidator:
    """Create Stage 1-compatible YAML skill cards for validated generated skills."""

    def __init__(self, repo_root: str | Path | None = None, generated_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        self.generated_root = Path(generated_root) if generated_root is not None else self.repo_root / "recskill" / "generated_skills"

    def consolidate(
        self,
        proposal: Any,
        code_path: str | Path,
        signature: SkillSignature,
        validation_results: dict[str, Any],
        init_params: dict[str, Any] | None = None,
    ) -> SkillCard:
        self.generated_root.mkdir(parents=True, exist_ok=True)
        skill_id = getattr(proposal, "skill_id", None) or _sanitize_identifier(getattr(proposal, "proposal_id"))
        card_path = self.generated_root / f"{skill_id}.skill.yaml"
        if card_path.exists():
            raise FileExistsError(f"Generated skill card already exists: {card_path}")
        code_path = Path(code_path)
        implementation_path = _relative_or_absolute(code_path, self.repo_root)
        example_params = _portable_example_params(
            dict(init_params if init_params is not None else (getattr(proposal, "init_params", {}) or {})),
            signature,
        )
        portability = _infer_portability(signature, example_params, validation_results)
        manifest = {
            "skill_id": skill_id,
            "name": skill_id,
            "skill_name": getattr(proposal, "skill_name", None) or skill_id,
            "category": _category_from_proposal(getattr(proposal, "proposal_type", "")),
            "function_description": getattr(proposal, "architecture_hypothesis", ""),
            "description": getattr(proposal, "architecture_hypothesis", ""),
            "inductive_bias": _as_list(getattr(proposal, "architecture_hypothesis", "")),
            "applicable_tasks": _as_list(getattr(proposal, "task_types", None)),
            "task_types": _as_list(getattr(proposal, "task_types", None)),
            "failure_modes_addressed": _as_list(getattr(proposal, "target_failure_mode", None)),
            "input_signature": [asdict(item) for item in signature.input_signature],
            "output_signature": [asdict(item) for item in signature.output_signature],
            "implementation_path": implementation_path,
            "class_name": signature.class_name,
            "source_proposal_id": getattr(proposal, "proposal_id", None),
            "validation_status": "passed",
            "validation_results": validation_results,
            "portability": portability,
            "created_at": utc_now_iso(),
            "parent_skills": list(getattr(proposal, "affected_genome_nodes", []) or []),
            "mutation_lineage": {
                "proposal_type": getattr(proposal, "proposal_type", None),
                "author": getattr(proposal, "author", None),
            },
            "composition": {
                "requires": [item.name for item in signature.input_signature],
                "produces": [item.name for item in signature.output_signature],
                "common_upstream": list(getattr(proposal, "affected_genome_nodes", []) or []),
                "common_downstream": [],
                "example_genome_fragment": {
                    "skill": skill_id,
                    "params": example_params,
                },
            },
            "retrieval": {
                "summary": getattr(proposal, "architecture_hypothesis", ""),
                "use_when": [getattr(proposal, "target_failure_mode", "")],
                "avoid_when": _as_list(getattr(proposal, "expected_risks", None)),
                "task_types": _as_list(getattr(proposal, "task_types", None)),
                "architecture_roles": [_category_from_proposal(getattr(proposal, "proposal_type", ""))],
                "input_modalities": [item.name for item in signature.input_signature],
                "output_semantics": [item.semantic or item.name for item in signature.output_signature],
                "objectives": [],
                "aliases": [skill_id, getattr(proposal, "proposal_type", "")],
            },
        }
        with card_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        return SkillCard(skill_id=skill_id, skill_name=manifest["skill_name"], category=manifest["category"], manifest=manifest, path=card_path)


def _synthetic_payload(signature: list[TensorSpec], init_params: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    if not signature:
        signature = [TensorSpec(name="x", shape=["batch_size", 4])]
    return {spec.name: _synthetic_tensor(spec, init_params or {}) for spec in signature}


def _portable_example_params(params: dict[str, Any], signature: SkillSignature) -> dict[str, Any]:
    portable = dict(params)
    symbolic = _signature_uses_symbolic_ctr_shapes(signature)
    if symbolic:
        for key, placeholder in {
            "num_fields": "${num_fields}",
            "embedding_dim": "${embedding_dim}",
            "input_dim": "${flat_input_dim}",
            "flat_input_dim": "${flat_input_dim}",
            "domain_num": "${domain_num}",
            "num_domains": "${domain_num}",
            "domain_representation_dim": "${domain_representation_dim}",
            "representation_dim": "${domain_representation_dim}",
        }.items():
            if key in portable and _is_positive_int_like(portable[key]):
                portable[key] = placeholder
    return portable


def _infer_portability(signature: SkillSignature, params: dict[str, Any], validation_results: dict[str, Any] | None = None) -> dict[str, Any]:
    issues: list[str] = []
    symbolic = _signature_uses_symbolic_ctr_shapes(signature)
    if not symbolic and _signature_has_fixed_ctr_shapes(signature):
        issues.append("input_signature contains fixed CTR field or flat dimensions")
    for key in (
        "num_fields",
        "embedding_dim",
        "input_dim",
        "flat_input_dim",
        "domain_num",
        "num_domains",
        "domain_representation_dim",
        "representation_dim",
    ):
        value = params.get(key)
        if _is_positive_int_like(value):
            issues.append(f"example_genome_fragment.params.{key} is fixed")
    portability_tests = (validation_results or {}).get("portability_tests") or {}
    if symbolic and not portability_tests.get("passed", False):
        issues.append("portability_tests did not pass")
    if issues:
        return {
            "scope": "dataset_specific",
            "reuse_enabled": False,
            "constraints": {},
            "issues": issues,
        }
    return {
        "scope": "schema_agnostic" if symbolic else "unknown",
        "reuse_enabled": True,
        "constraints": {
            "min_num_fields": _infer_min_num_fields(signature),
        },
        "issues": [],
    }


def _signature_uses_symbolic_ctr_shapes(signature: SkillSignature) -> bool:
    symbols = {
        "num_fields",
        "embedding_dim",
        "input_dim",
        "flat_input_dim",
        "domain_num",
        "num_domains",
        "domain_representation_dim",
        "representation_dim",
    }
    for spec in signature.input_signature:
        for dim in spec.shape:
            if isinstance(dim, str) and dim in symbols:
                return True
    return False


def _signature_has_fixed_ctr_shapes(signature: SkillSignature) -> bool:
    for spec in signature.input_signature:
        name = spec.name.lower()
        if len(spec.shape) == 3 and "field" in name:
            if _is_positive_int_like(spec.shape[1]) or _is_positive_int_like(spec.shape[2]):
                return True
        if len(spec.shape) == 2 and ("flat" in name or "input" in name):
            if _is_positive_int_like(spec.shape[1]):
                return True
    return False


def _infer_min_num_fields(signature: SkillSignature) -> int:
    text = json.dumps([asdict(item) for item in signature.input_signature], sort_keys=True)
    lowered = text.lower()
    if "triple" in lowered or "order3" in lowered or "third" in lowered:
        return 3
    return 1


def _is_positive_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        return value.isdigit() and int(value) > 0
    return False


def _synthetic_tensor(spec: TensorSpec, init_params: dict[str, Any] | None = None) -> torch.Tensor:
    shape = _resolve_shape(spec.shape) or [2, 4]
    dtype = spec.dtype.lower()
    int_upper_bound = _synthetic_int_upper_bound(spec, init_params or {})
    if dtype in {"int64", "long"}:
        return torch.randint(0, int_upper_bound, shape, dtype=torch.long)
    if dtype in {"int32", "int"}:
        return torch.randint(0, int_upper_bound, shape, dtype=torch.int)
    tensor = torch.randn(*shape, dtype=torch.float32)
    tensor.requires_grad_(True)
    return tensor


def _synthetic_int_upper_bound(spec: TensorSpec, init_params: dict[str, Any]) -> int:
    lowered = spec.name.lower()
    if lowered in {"domain_id", "domain_indicator"} or "domain" in lowered:
        for key in ("domain_num", "num_domains", "n_task"):
            value = init_params.get(key)
            if _is_positive_int_like(value):
                return max(1, int(value))
        return 3
    return 8


def _resolve_shape(shape: list[str | int]) -> list[int]:
    symbols = {
        "batch_size": 2,
        "batch": 2,
        "num_fields": 3,
        "embedding_dim": 4,
        "input_dim": 4,
        "hidden_dim": 4,
        "seq_len": 5,
        "sequence_length": 5,
        "num_tasks": 2,
        "domain_num": 3,
        "num_domains": 3,
        "domain_representation_dim": 4,
        "representation_dim": 4,
    }
    resolved = []
    for dim in shape:
        if isinstance(dim, int):
            resolved.append(dim)
        elif isinstance(dim, str):
            if dim.isdigit():
                resolved.append(int(dim))
            elif dim in symbols:
                resolved.append(symbols[dim])
            else:
                resolved.append(4)
    return resolved


def _call_skill(skill: Any, payload: dict[str, torch.Tensor]) -> Any:
    try:
        return skill(payload)
    except TypeError:
        if len(payload) == 1:
            return skill(next(iter(payload.values())))
        raise


def _parse_shape(value: Any) -> list[str | int]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        return [int(part.strip()) if part.strip().isdigit() else part.strip() for part in stripped.split(",") if part.strip()]
    return []


def _literal_subscript_key(node: ast.Subscript) -> str | None:
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _call_name(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    return ""


def _sanitize_identifier(value: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in value.lower()]
    sanitized = "".join(chars).strip("_")
    return sanitized or "generated_skill"


def _category_from_proposal(proposal_type: str) -> str:
    mapping = {
        "NEW_SKILL_INVENTION": "adapter",
        "LOCAL_CODE_SURGERY": "adapter",
        "NEW_FUSION_DESIGN": "fusion",
        "NEW_OBJECTIVE_DESIGN": "objective",
        "NEW_TRAINING_MECHANISM": "training",
        "NEW_ROUTING_OR_GATING_DESIGN": "routing",
    }
    return mapping.get(proposal_type, "adapter")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
