#!/usr/bin/env python3
"""LLM command wrapper for macro-only code-space evolution.

The evolution workflow sends prompt JSON on stdin and expects JSON on stdout.
This script keeps that contract stable while letting the actual model backend be
configured outside the repo through environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROVIDER_ENV = "EVOSKILLREC_CODE_SPACE_PROVIDER"
PLANNER_PROVIDER_ENV = "EVOSKILLREC_CODE_SPACE_PLANNER_PROVIDER"
SYNTHESIZER_PROVIDER_ENV = "EVOSKILLREC_CODE_SPACE_SYNTHESIZER_PROVIDER"
PLANNER_ENV = "EVOSKILLREC_CODE_SPACE_PLANNER_BACKEND"
SYNTHESIZER_ENV = "EVOSKILLREC_CODE_SPACE_SYNTHESIZER_BACKEND"
SHARED_ENV = "EVOSKILLREC_CODE_SPACE_BACKEND"
MODEL_ENV = "EVOSKILLREC_CODE_SPACE_MODEL"
PLANNER_MODEL_ENV = "EVOSKILLREC_CODE_SPACE_PLANNER_MODEL"
SYNTHESIZER_MODEL_ENV = "EVOSKILLREC_CODE_SPACE_SYNTHESIZER_MODEL"
CODEX_COMMAND_ENV = "EVOSKILLREC_CODE_SPACE_CODEX_COMMAND"
CODEX_ARGS_ENV = "EVOSKILLREC_CODE_SPACE_CODEX_ARGS"
CODEX_PROFILE_ENV = "EVOSKILLREC_CODE_SPACE_CODEX_PROFILE"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
TIMEOUT_ENV = "EVOSKILLREC_CODE_SPACE_TIMEOUT_SEC"
TEMPERATURE_ENV = "EVOSKILLREC_CODE_SPACE_TEMPERATURE"
MAX_TOKENS_ENV = "EVOSKILLREC_CODE_SPACE_MAX_TOKENS"

MACRO_WIRING = {
    "replace_node",
    "insert_between",
    "branch_to_fusion",
    "replace_fusion",
    "generated_fusion",
    "fusion_replacement",
    "macro_fusion",
}
# Multi-task and multi-domain evolution only use topology-generic wiring modes
# (no CTR fusion/logit rerouting).
TOPOLOGY_GENERIC_MACRO_WIRING = {
    "replace_node",
    "insert_between",
}


def _allowed_wiring(domain: str) -> set[str]:
    return TOPOLOGY_GENERIC_MACRO_WIRING if domain in {"multitask", "multi_domain"} else MACRO_WIRING
MACRO_JUDGMENT_FIELDS = {
    "current_architecture_family",
    "diagnosed_limitation",
    "target_architecture_transformation",
    "why_local_edit_is_insufficient",
    "parent_evidence",
    "proposal_insight",
    "preconditions",
    "wiring_plan",
    "ablation_plan",
}
FORBIDDEN_CODE_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bsubprocess\b",
    r"\bsocket\b",
    r"\brequests\b",
    r"\bhttpx\b",
    r"\burllib\b",
    r"\bopen\s*\(",
    r"\bos\.",
    r"\bsys\.",
]


def main() -> int:
    args = _parse_args()
    prompt = _read_json_stdin()
    domain = _resolve_domain(args.task, prompt)
    provider = _provider_for_mode(args.mode)
    system_prompt = _system_prompt(args.mode, domain)
    model_input = {
        "system": system_prompt,
        "task": args.mode,
        "domain": domain,
        "prompt": prompt,
    }
    raw = _run_provider(provider, args.mode, model_input)
    payload = _extract_json(raw)
    if args.mode == "planner":
        payload = _validate_planner_payload(payload, prompt, domain)
    else:
        payload = _validate_synthesizer_payload(payload, prompt, domain)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["planner", "synthesizer"], required=True)
    parser.add_argument(
        "--task",
        choices=["ctr", "multitask", "multi_domain", "auto"],
        default="auto",
        help=(
            "Evolution domain. 'ctr' (single-task logits/fusion), 'multitask' "
            "(task_tower/multitask_loss), or 'multi_domain' "
            "(domain routing/domain_select). "
            "'auto' (default) infers it from the prompt's design brief."
        ),
    )
    return parser.parse_args()


def _resolve_domain(task_arg: str, prompt: dict[str, Any]) -> str:
    """Resolve the evolution domain, preferring the explicit flag then the prompt."""
    if task_arg in {"ctr", "multitask", "multi_domain"}:
        return task_arg
    brief = _design_brief(prompt)
    constraints = brief.get("design_constraints") if isinstance(brief, dict) else None
    declared = ""
    if isinstance(constraints, dict):
        declared = str(constraints.get("task_type") or "").lower()
    family = str(brief.get("task_family") or prompt.get("task_family") or "").lower() if isinstance(brief, dict) else ""
    if declared in {"multi_domain", "multi-domain"} or "multi_domain" in family or "multi-domain" in family:
        return "multi_domain"
    if declared in {"multitask", "multi_task"} or "multi_task" in family or "multitask" in family:
        return "multitask"
    return "ctr"


def _read_json_stdin() -> dict[str, Any]:
    text = sys.stdin.read()
    if not text.strip():
        raise SystemExit("Expected prompt JSON on stdin.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid stdin JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("Prompt JSON must be an object.")
    return data


def _provider_for_mode(mode: str) -> str:
    stage_env = PLANNER_PROVIDER_ENV if mode == "planner" else SYNTHESIZER_PROVIDER_ENV
    provider = os.environ.get(stage_env) or os.environ.get(PROVIDER_ENV)
    if provider:
        return provider.strip().lower().replace("-", "_")
    if _command_backend_for_mode(mode, required=False):
        return "command"
    return "codex"


def _run_provider(provider: str, mode: str, payload: dict[str, Any]) -> str:
    if provider in {"command", "custom_command", "local_command"}:
        return _run_command_backend(_command_backend_for_mode(mode, required=True), payload)
    if provider in {"codex", "codex_cli"}:
        return _run_codex_cli(mode, payload)
    if provider in {"gemini", "google_gemini", "google"}:
        return _run_gemini(mode, payload)
    if provider in {"deepseek", "deep_seek"}:
        return _run_deepseek(mode, payload)
    raise SystemExit(
        f"Unknown code-space model provider: {provider!r}. "
        "Use codex, command, gemini, or deepseek."
    )


def _command_backend_for_mode(mode: str, *, required: bool) -> str:
    env_name = PLANNER_ENV if mode == "planner" else SYNTHESIZER_ENV
    command = os.environ.get(env_name) or os.environ.get(SHARED_ENV)
    if not command and required:
        raise SystemExit(
            f"Missing model backend. Set {env_name} or {SHARED_ENV} to a command "
            "that reads JSON from stdin and writes JSON to stdout."
        )
    return command or ""


def _run_command_backend(command: str, payload: dict[str, Any]) -> str:
    timeout = float(os.environ.get(TIMEOUT_ENV, "300"))
    proc = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload, sort_keys=True),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise SystemExit(stderr or f"Backend command exited with {proc.returncode}.")
    return proc.stdout


def _run_codex_cli(mode: str, payload: dict[str, Any]) -> str:
    command = shlex.split(os.environ.get(CODEX_COMMAND_ENV, "codex"))
    model = _model_for_mode(mode, provider="codex", default="")
    profile = os.environ.get(CODEX_PROFILE_ENV)
    extra_args = shlex.split(os.environ.get(CODEX_ARGS_ENV, ""))
    timeout = float(os.environ.get(TIMEOUT_ENV, "300"))
    prompt = _render_text_prompt(payload)

    output_path = Path(tempfile.NamedTemporaryFile(prefix="evoskillrec_codex_", suffix=".txt", delete=False).name)
    cmd = [
        *command,
        "exec",
        "--cd",
        os.getcwd(),
        "--sandbox",
        "read-only",
        "-c",
        'approval_policy="never"',
        "--ephemeral",
        "--output-last-message",
        str(output_path),
    ]
    if model:
        cmd.extend(["--model", model])
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(extra_args)
    cmd.append("-")
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if proc.returncode != 0:
            detail = "\n".join(part for part in [proc.stderr.strip(), proc.stdout.strip()] if part)
            raise SystemExit(detail or f"Codex CLI exited with {proc.returncode}.")
        if output_path.exists():
            final = output_path.read_text(encoding="utf-8").strip()
            if final:
                return final
        return proc.stdout
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def _run_gemini(mode: str, payload: dict[str, Any]) -> str:
    api_key = os.environ.get(GEMINI_API_KEY_ENV) or os.environ.get(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Missing Gemini API key. Set {GEMINI_API_KEY_ENV} or {GOOGLE_API_KEY_ENV}.")
    model = _model_for_mode(mode, provider="gemini", default="gemini-2.5-pro")
    endpoint = os.environ.get(
        "EVOSKILLREC_CODE_SPACE_GEMINI_ENDPOINT",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    )
    url = endpoint.format(model=urllib.parse.quote(model, safe=""))
    separator = "&" if "?" in url else "?"
    url = f"{url}{separator}key={urllib.parse.quote(api_key)}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": _render_text_prompt(payload)}]}],
        "generationConfig": {
            "temperature": _temperature(),
            "responseMimeType": "application/json",
        },
    }
    text = _post_json(url, body, headers={"Content-Type": "application/json"})
    data = json.loads(text)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise SystemExit(f"Gemini response did not contain text content: {data}") from exc


def _run_deepseek(mode: str, payload: dict[str, Any]) -> str:
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Missing DeepSeek API key. Set {DEEPSEEK_API_KEY_ENV}.")
    model = _model_for_mode(mode, provider="deepseek", default="deepseek-chat")
    base_url = os.environ.get("EVOSKILLREC_CODE_SPACE_DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    url = f"{base_url}/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": str(payload.get("system") or "")},
            {"role": "user", "content": _render_user_payload(payload)},
        ],
        "temperature": _temperature(),
        "response_format": {"type": "json_object"},
    }
    max_tokens = os.environ.get(MAX_TOKENS_ENV)
    if max_tokens:
        body["max_tokens"] = int(max_tokens)
    text = _post_json(
        url,
        body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    data = json.loads(text)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise SystemExit(f"DeepSeek response did not contain message content: {data}") from exc


def _post_json(url: str, body: dict[str, Any], *, headers: dict[str, str]) -> str:
    timeout = float(os.environ.get(TIMEOUT_ENV, "300"))
    request = urllib.request.Request(
        url,
        data=json.dumps(body, sort_keys=True).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP provider failed with {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"HTTP provider failed: {exc}") from exc


def _model_for_mode(mode: str, *, provider: str, default: str) -> str:
    stage_env = PLANNER_MODEL_ENV if mode == "planner" else SYNTHESIZER_MODEL_ENV
    provider_env = f"EVOSKILLREC_CODE_SPACE_{provider.upper()}_MODEL"
    return os.environ.get(stage_env) or os.environ.get(provider_env) or os.environ.get(MODEL_ENV) or default


def _temperature() -> float:
    return float(os.environ.get(TEMPERATURE_ENV, "0.2"))


def _render_text_prompt(payload: dict[str, Any]) -> str:
    contract = _response_contract(str(payload.get("task") or ""), str(payload.get("domain") or "ctr"))
    return (
        f"{payload.get('system', '')}\n\n"
        "Return only one JSON object. Do not include markdown unless the JSON object is inside a single json code fence.\n\n"
        f"{contract}\n\n"
        "Input JSON:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


def _render_user_payload(payload: dict[str, Any]) -> str:
    contract = _response_contract(str(payload.get("task") or ""), str(payload.get("domain") or "ctr"))
    return (
        "Return only one JSON object matching the requested schema.\n\n"
        f"{contract}\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _response_contract(task: str, domain: str = "ctr") -> str:
    wiring_hint = (
        "insert_between or replace_node"
        if domain in {"multitask", "multi_domain"}
        else "replace_node, insert_between, branch_to_fusion, or replace_fusion"
    )
    if task == "planner":
        return (
            "Required planner response contract: return {\"sketches\": [...]} only. "
            "Every sketch object must include sketch_id, innovation_lane, target_failure_mode, hypothesis, "
            "affected_genome_nodes, input_keys, output_keys, wiring, structural_scope, and metadata. "
            f"Use structural_scope=\"macro\" and a macro wiring value such as {wiring_hint}. "
            "input_keys must be drawn only from design_brief.available_context_keys."
        )
    if task == "synthesizer":
        judgment_fields = ", ".join(sorted(MACRO_JUDGMENT_FIELDS))
        if domain == "multi_domain":
            return (
                "Required synthesizer response contract: return {\"proposals\": [...]} only. "
                "Every proposal object must include proposal_id, proposal_type, structural_scope, target_failure_mode, "
                "architecture_hypothesis, affected_genome_nodes, code, expected_input_signature, "
                "expected_output_signature, class_name, skill_id, task_types, init_params, and metadata. "
                "Use structural_scope=\"macro\" and metadata.wiring set to insert_between or replace_node. "
                f"metadata.macro_judgment must include: {judgment_fields}. expected_input_signature names must be drawn "
                "only from design_brief.available_context_keys. Set task_types=[\"multi_domain\"]. The generated module must "
                "PRESERVE the shape of the tensor it intercepts: field_embeddings [batch_size, num_fields, embedding_dim], "
                "flat_embeddings [batch_size, flat_input_dim], domain_representations [batch_size, domain_num, representation_dim], "
                "or domain_outputs [batch_size, domain_num]. Do not target protected nodes field_embedding, domain_adapter, "
                "domain_select, or loss."
            )
        if domain == "multitask":
            return (
                "Required synthesizer response contract: return {\"proposals\": [...]} only. "
                "Every proposal object must include proposal_id, proposal_type, structural_scope, target_failure_mode, "
                "architecture_hypothesis, affected_genome_nodes, code, expected_input_signature, "
                "expected_output_signature, class_name, skill_id, task_types, init_params, and metadata. "
                "Use structural_scope=\"macro\" and metadata.wiring set to insert_between or replace_node. "
                f"metadata.macro_judgment must include: {judgment_fields}. expected_input_signature names must be drawn "
                "only from design_brief.available_context_keys. Set task_types=[\"multitask\"]. The generated module must "
                "PRESERVE the shape of the tensor it intercepts (flat_embeddings [batch_size, flat_input_dim] stays the same "
                "width; task_representations [batch_size, n_task, representation_dim] stays the same shape). For flat_embeddings "
                "skills use symbolic init_params ${num_fields}, ${embedding_dim}, and ${flat_input_dim}; for task_representations "
                "skills set n_task and representation_dim explicitly. Do not target the protected nodes task_tower, loss, or field_embedding."
            )
        return (
            "Required synthesizer response contract: return {\"proposals\": [...]} only. "
            "Every proposal object must include proposal_id, proposal_type, structural_scope, target_failure_mode, "
            "architecture_hypothesis, affected_genome_nodes, code, expected_input_signature, "
            "expected_output_signature, class_name, skill_id, task_types, init_params, and metadata. "
            f"Use structural_scope=\"macro\" and metadata.wiring with a macro wiring value such as {wiring_hint}. "
            "metadata.macro_judgment must include: "
            f"{judgment_fields}. expected_input_signature names must be drawn only from design_brief.available_context_keys. "
            "Portable CTR dimension init_params may use ${num_fields}, ${embedding_dim}, and ${flat_input_dim}; "
            "do not hard-code dataset-specific field counts or flattened dimensions."
        )
    return "Return only the requested JSON object."


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise SystemExit("Backend returned empty output.")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
        if not match:
            raise SystemExit("Backend output did not contain a JSON object.")
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Backend JSON could not be parsed: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("Backend JSON must be an object.")
    return data


def _validate_planner_payload(payload: dict[str, Any], prompt: dict[str, Any], domain: str = "ctr") -> dict[str, Any]:
    sketches = payload.get("sketches") or payload.get("architecture_sketches")
    if not isinstance(sketches, list) or not sketches:
        raise SystemExit("Planner must return a non-empty sketches list.")
    allowed_keys = set(_design_brief(prompt).get("available_context_keys") or [])
    allowed_wiring = _allowed_wiring(domain)
    for idx, sketch in enumerate(sketches):
        if not isinstance(sketch, dict):
            raise SystemExit(f"Sketch {idx} must be an object.")
        if not sketch.get("sketch_id"):
            raise SystemExit(f"Sketch {idx} is missing sketch_id.")
        if str(sketch.get("structural_scope") or "").lower() != "macro":
            raise SystemExit(f"Sketch {sketch.get('sketch_id')} is not macro scope.")
        wiring = _normalized_wiring(sketch)
        if wiring not in allowed_wiring:
            raise SystemExit(
                f"Sketch {sketch.get('sketch_id')} uses wiring {wiring!r} not allowed for {domain}; "
                f"expected one of {sorted(allowed_wiring)}."
            )
        input_keys = [str(key) for key in sketch.get("input_keys") or []]
        missing = sorted(key for key in input_keys if key not in allowed_keys)
        if missing:
            raise SystemExit(f"Sketch {sketch.get('sketch_id')} uses unavailable input keys: {missing}.")
        if not sketch.get("output_keys"):
            raise SystemExit(f"Sketch {sketch.get('sketch_id')} is missing output_keys.")
    return {"sketches": sketches}


def _validate_synthesizer_payload(payload: dict[str, Any], prompt: dict[str, Any], domain: str = "ctr") -> dict[str, Any]:
    proposals = payload.get("proposals") or payload.get("open_ended_proposals")
    if not isinstance(proposals, list) or not proposals:
        raise SystemExit("Synthesizer must return a non-empty proposals list.")
    allowed_keys = set(_design_brief(prompt).get("available_context_keys") or [])
    allowed_wiring = _allowed_wiring(domain)
    if domain == "multitask":
        protected_nodes = {"task_tower", "loss", "field_embedding"}
    elif domain == "multi_domain":
        protected_nodes = {"field_embedding", "domain_adapter", "domain_select", "loss"}
    else:
        protected_nodes = set()
    for idx, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            raise SystemExit(f"Proposal {idx} must be an object.")
        proposal_id = proposal.get("proposal_id") or f"index {idx}"
        if str(proposal.get("structural_scope") or "").lower() != "macro":
            raise SystemExit(f"Proposal {proposal_id} is not macro scope.")
        metadata = proposal.get("metadata") or {}
        wiring = _normalized_wiring(proposal)
        if wiring not in allowed_wiring:
            raise SystemExit(
                f"Proposal {proposal_id} uses wiring {wiring!r} not allowed for {domain}; "
                f"expected one of {sorted(allowed_wiring)}."
            )
        if domain in {"multitask", "multi_domain"} and wiring == "replace_node":
            target = str(metadata.get("target_node_id") or "")
            if target in protected_nodes:
                raise SystemExit(
                    f"Proposal {proposal_id} replace_node target {target!r} is protected for {domain}; "
                    "target a transform or representation node instead."
                )
        if domain == "multitask":
            task_types = [str(t).lower() for t in (proposal.get("task_types") or [])]
            if task_types and "multitask" not in task_types:
                raise SystemExit(
                    f"Proposal {proposal_id} must set task_types=['multitask'] for multi-task evolution."
                )
        if domain == "multi_domain":
            task_types = [str(t).lower().replace("-", "_") for t in (proposal.get("task_types") or [])]
            if task_types and "multi_domain" not in task_types:
                raise SystemExit(
                    f"Proposal {proposal_id} must set task_types=['multi_domain'] for multi-domain evolution."
                )
        judgment = metadata.get("macro_judgment") or {}
        missing_judgment = sorted(field for field in MACRO_JUDGMENT_FIELDS if not judgment.get(field))
        if missing_judgment:
            raise SystemExit(f"Proposal {proposal_id} missing macro_judgment fields: {missing_judgment}.")
        input_keys = [
            str(item.get("name"))
            for item in proposal.get("expected_input_signature") or []
            if isinstance(item, dict) and item.get("name")
        ]
        missing_inputs = sorted(key for key in input_keys if key not in allowed_keys)
        if missing_inputs:
            raise SystemExit(f"Proposal {proposal_id} uses unavailable input keys: {missing_inputs}.")
        for item in proposal.get("expected_input_signature") or []:
            if isinstance(item, dict):
                _validate_portable_ctr_signature(proposal_id, item)
        if not proposal.get("expected_output_signature"):
            raise SystemExit(f"Proposal {proposal_id} is missing expected_output_signature.")
        code = str(proposal.get("code") or "")
        if not code.strip():
            raise SystemExit(f"Proposal {proposal_id} is missing code.")
        if "nn.Module" not in code or "def forward" not in code:
            raise SystemExit(f"Proposal {proposal_id} code must define an nn.Module with forward.")
        for pattern in FORBIDDEN_CODE_PATTERNS:
            if re.search(pattern, code):
                raise SystemExit(f"Proposal {proposal_id} code contains forbidden pattern: {pattern}.")
    return {"proposals": proposals}


def _validate_portable_ctr_signature(proposal_id: str, item: dict[str, Any]) -> None:
    name = str(item.get("name") or "")
    shape = item.get("shape") or []
    if not isinstance(shape, list):
        return
    if name == "field_embeddings" and len(shape) >= 3:
        if _is_fixed_dim(shape[1]) or _is_fixed_dim(shape[2]):
            raise SystemExit(
                f"Proposal {proposal_id} hard-codes field_embeddings shape {shape}; use "
                "[batch_size, num_fields, embedding_dim] for reusable generated skills."
            )
    if name == "flat_embeddings" and len(shape) >= 2:
        if _is_fixed_dim(shape[1]):
            raise SystemExit(
                f"Proposal {proposal_id} hard-codes flat_embeddings shape {shape}; use "
                "[batch_size, flat_input_dim] for reusable generated skills."
            )


def _is_fixed_dim(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        return value.isdigit() and int(value) > 0
    return False


def _design_brief(prompt: dict[str, Any]) -> dict[str, Any]:
    if isinstance(prompt.get("design_brief"), dict):
        return prompt["design_brief"]
    nested = prompt.get("prompt")
    if isinstance(nested, dict) and isinstance(nested.get("design_brief"), dict):
        return nested["design_brief"]
    return {}


def _normalized_wiring(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return _normalize_wiring_value(item.get("wiring") or metadata.get("wiring") or metadata.get("integration") or "")


def _normalize_wiring_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("op", "operation", "mode", "type", "wiring"):
            nested = value.get(key)
            if nested:
                return _normalize_wiring_value(nested)
        return ""
    return str(value).lower().replace("-", "_")


def _system_prompt(mode: str, domain: str = "ctr") -> str:
    if domain == "multi_domain":
        if mode == "planner":
            return (
                "You are the macro architecture planner for EvoSkillRec multi-domain recommendation evolution. "
                "The model learns shared and domain-specific behavior, uses domain_indicator/domain_id for "
                "sample-wise scenario routing, and terminates through domain_select plus BCE loss. Read the prompt "
                "JSON and return only JSON with a top-level sketches list. Every sketch must be structural_scope='macro', "
                "must use wiring 'insert_between' or 'replace_node' at one of design_brief.injection_points, and must "
                "preserve the intercepted tensor shape. Do not touch protected nodes field_embedding, domain_adapter, "
                "domain_select, or loss. Do not propose CTR fusion/logit rerouting."
            )
        return (
            "You are the macro code synthesizer for EvoSkillRec multi-domain recommendation evolution. "
            "Read selected macro sketches and return only JSON with a top-level proposals list. Each proposal must "
            "be safe local PyTorch code implementing nn.Module.forward(inputs), must set task_types=['multi_domain'], "
            "must use wiring 'insert_between' or 'replace_node', and must preserve the intercepted tensor shape. "
            "Never target field_embedding, domain_adapter, domain_select, or loss. Do not use file IO, network, "
            "subprocess, eval, exec, os, or sys."
        )
    if domain == "multitask":
        if mode == "planner":
            return (
                "You are the macro architecture planner for EvoSkillRec multi-task recommendation evolution. "
                "The model jointly trains several binary task heads (e.g. income as a CTR-style task and marital "
                "status as a CVR-style task) by splitting a shared representation into per-task representations. "
                "Read the prompt JSON and return only JSON with a top-level sketches list. Every sketch must be "
                "structural_scope='macro', justified by the retained parent genome, and must use wiring 'insert_between' "
                "or 'replace_node' at one of design_brief.injection_points. Sketches must preserve the intercepted "
                "tensor's shape and must not touch the protected nodes task_tower, loss, or field_embedding. "
                "Do not propose calibration-only, scalar-only, or residual-only local edits."
            )
        return (
            "You are the macro code synthesizer for EvoSkillRec multi-task recommendation evolution. "
            "Read selected macro sketches and return only JSON with a top-level proposals list. "
            "Each proposal must be safe local PyTorch code implementing nn.Module.forward(inputs), must preserve "
            "metadata.macro_judgment, must set task_types=['multitask'], and must use wiring 'insert_between' or "
            "'replace_node' while preserving the intercepted tensor's shape. Never target task_tower, loss, or "
            "field_embedding. Do not use file IO, network, subprocess, eval, exec, os, or sys."
        )
    if mode == "planner":
        return (
            "You are the macro architecture planner for EvoSkillRec CTR evolution. "
            "Read the prompt JSON and return only JSON with a top-level sketches list. "
            "Every sketch must be structural_scope='macro', justified by the retained parent genome, "
            "and must use macro wiring such as replace_node, insert_between, branch_to_fusion, or replace_fusion. "
            "Do not propose calibration-only, scalar-only, post-logit, or residual-only local edits."
        )
    return (
        "You are the macro code synthesizer for EvoSkillRec CTR evolution. "
        "Read selected macro sketches and return only JSON with a top-level proposals list. "
        "Each proposal must be safe local PyTorch code implementing nn.Module.forward(inputs), "
        "must preserve metadata.macro_judgment, and must use macro topology wiring. "
        "Do not use file IO, network, subprocess, eval, exec, os, or sys."
    )


if __name__ == "__main__":
    raise SystemExit(main())
