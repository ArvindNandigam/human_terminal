"""Natural-language interpretation for hcmd.

Trained local models are used when available.  A deterministic parser remains
available for installation, help, and common command workflows when model
artifacts are not installed.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..constants import (
    DST_SPAN_MODEL_PATH,
    GIT_ADD_MODEL_PATH,
    GIT_CHECKOUT_MODEL_PATH,
    GIT_CLONE_MODEL_PATH,
    INTENT_CONF_THRESHOLD,
    INTENT_MODEL_PATH,
    NAVIGATION_SPAN_MODEL_PATH,
    OBJECT_SPAN_MODEL_PATH,
    RENAME_SPAN_MODEL_PATH,
    SRC_SPAN_MODEL_PATH,
    SPAN_CONF_THRESHOLD,
    TARGET_SPAN_MODEL_PATH,
)
from .intent_rules import rule_intent
from .memory import memory
from .patterns import detect_pattern

MODEL_PATHS = (
    INTENT_MODEL_PATH,
    NAVIGATION_SPAN_MODEL_PATH,
    SRC_SPAN_MODEL_PATH,
    DST_SPAN_MODEL_PATH,
    OBJECT_SPAN_MODEL_PATH,
    RENAME_SPAN_MODEL_PATH,
    GIT_ADD_MODEL_PATH,
    GIT_CHECKOUT_MODEL_PATH,
    GIT_CLONE_MODEL_PATH,
    TARGET_SPAN_MODEL_PATH,
)

INTENT_VERBS = {
    "CREATE_FILE": ("create file", "create a file", "make file", "make a file", "new file", "touch"),
    "DELETE_FILE": ("delete file", "delete", "remove file", "remove", "erase"),
    "MOVE_FILE": ("move", "relocate"),
    "COPY_FILE": ("copy", "duplicate"),
    "RENAME_FILE": ("rename", "change filename"),
    "CREATE_DIR": ("create directory", "make directory", "new directory", "create folder", "make folder", "new folder", "mkdir"),
    "DELETE_DIR": ("delete directory", "remove directory", "erase directory", "delete folder", "remove folder", "erase folder", "rmdir"),
    "NAVIGATION": ("go to", "navigate to", "change directory to", "cd "),
    "GIT_ADD": ("git add", "stage "),
    "GIT_COMMIT": ("git commit", "commit"),
    "GIT_CHECKOUT": ("git checkout", "checkout"),
    "GIT_BRANCH": ("git branch", "branch"),
    "GIT_CLONE": ("git clone", "clone"),
    "GIT_STATUS": ("git status", "status"),
    "GIT_LOG": ("git log", "log"),
    "GIT_PULL": ("git pull", "pull"),
    "GIT_PUSH": ("git push", "push"),
    "GIT_RESET": ("git reset", "reset"),
    "GIT_STASH": ("git stash", "stash"),
    "PROCESS_KILL": ("kill process", "terminate process", "kill ", "terminate "),
    "PROCESS_LIST": ("list processes", "show running processes", "tasklist"),
    "SYSTEM_INFO": ("system info", "system information", "hostname", "whoami"),
    "NETWORK_INFO": ("network info", "network configuration", "ipconfig", "ip address"),
    "PWD": ("where am i", "current directory", "print working directory", "pwd"),
    "LIST_FILES": ("list files", "show files", "list directory", "show directory", "ls", "dir"),
    "READ_FILE": ("read file", "show contents", "cat "),
}


def _models_available() -> bool:
    return all(Path(path).is_dir() for path in MODEL_PATHS)


def _load_models() -> Optional[dict[str, Any]]:
    if not _models_available():
        return None
    try:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import (
            AutoModelForQuestionAnswering,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        from .rename_model import DualSpanQA

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH, local_files_only=True)
        intent_model = AutoModelForSequenceClassification.from_pretrained(
            INTENT_MODEL_PATH, local_files_only=True
        ).to(device).eval()
        span_tokenizer = AutoTokenizer.from_pretrained(
            NAVIGATION_SPAN_MODEL_PATH, local_files_only=True
        )
        qa_paths = {
            "nav_model": NAVIGATION_SPAN_MODEL_PATH,
            "src_model": SRC_SPAN_MODEL_PATH,
            "dst_model": DST_SPAN_MODEL_PATH,
            "object_model": OBJECT_SPAN_MODEL_PATH,
            "target_model": TARGET_SPAN_MODEL_PATH,
            "git_add_model": GIT_ADD_MODEL_PATH,
            "git_checkout_model": GIT_CHECKOUT_MODEL_PATH,
            "git_clone_model": GIT_CLONE_MODEL_PATH,
        }
        models = {
            "intent_tokenizer": intent_tokenizer,
            "intent_model": intent_model,
            "span_tokenizer": span_tokenizer,
            "device": device,
        }
        for name, path in qa_paths.items():
            models[name] = AutoModelForQuestionAnswering.from_pretrained(
                path, local_files_only=True
            ).to(device).eval()
        rename_model = DualSpanQA("distilbert-base-uncased").to(device)
        rename_model.load_state_dict(
            torch.load(
                Path(RENAME_SPAN_MODEL_PATH) / "pytorch_model.bin",
                map_location=device,
            )
        )
        models["rename_model"] = rename_model.eval()
        return models
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


MODELS = _load_models()


def in_git_repo() -> bool:
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _first_match(patterns: tuple[str, ...], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"\'') if match.groups() else None
    return None


def _fallback_intent(text: str) -> Optional[str]:
    normalized = text.lower().strip()
    rule = rule_intent(text)
    if rule:
        return rule
    if normalized in ("ls", "dir", "list", "list files", "show files"):
        return "LIST_FILES"
    for intent, phrases in INTENT_VERBS.items():
        if any(phrase in normalized for phrase in phrases):
            return intent
    return None


def _fallback_spans(intent: str, text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    quoted_values = [a or b for a, b in quoted]

    if intent in {"MOVE_FILE", "COPY_FILE"}:
        match = re.search(r"\b(?:move|copy|relocate|duplicate)\s+(.+?)\s+(?:to|into)\s+(.+)$", text, re.I)
        if match:
            result.update(src=match.group(1).strip().strip('"\' '), dst=match.group(2).strip().strip('"\' '))
    elif intent == "RENAME_FILE":
        match = re.search(r"\brename\s+(.+?)\s+to\s+(.+)$", text, re.I)
        if match:
            result.update(src=match.group(1).strip().strip('"\' '), dst=match.group(2).strip().strip('"\' '))
    elif intent in {"NAVIGATION", "CREATE_DIR", "DELETE_DIR"}:
        match = re.search(r"(?:to|directory|folder|mkdir|cd)\s+(.+)$", text, re.I)
        if match:
            result["path"] = match.group(1).strip().strip('"\' ')
    elif intent in {"CREATE_FILE", "DELETE_FILE", "READ_FILE"}:
        if quoted_values:
            result["path"] = quoted_values[-1]
        else:
            match = re.search(
                r"(?:file|delete|remove|erase|touch|cat)\s+(?:named\s+)?([^\s]+)",
                text,
                re.I,
            )
            if match:
                result["path"] = match.group(1).strip().strip('"\' ')
    elif intent == "GIT_ADD":
        match = re.search(r"(?:git\s+add|stage)\s+(.+)$", text, re.I)
        if match:
            result["path"] = match.group(1).strip().strip('"\' ')
    elif intent == "GIT_CHECKOUT":
        match = re.search(r"(?:git\s+checkout|checkout)\s+(.+)$", text, re.I)
        if match:
            result["branch"] = match.group(1).strip().strip('"\' ')
    elif intent == "GIT_CLONE":
        match = re.search(r"(?:git\s+clone|clone)(?:\s+repository)?\s+(?:from\s+)?(.+)$", text, re.I)
        if match:
            result["repo"] = match.group(1).strip().strip('"\' ')
    elif intent == "PROCESS_KILL":
        match = re.search(r"(?:kill|terminate)(?:\s+process)?\s+(.+)$", text, re.I)
        if match:
            result["target"] = match.group(1).strip().strip('"\' ')
    return result


def _extract_span(text: str, model: Any) -> tuple[Optional[str], float]:
    import torch

    inputs = MODELS["span_tokenizer"](
        text, return_tensors="pt", return_offsets_mapping=True, truncation=True
    )
    offsets = inputs.pop("offset_mapping")[0]
    inputs = {key: value.to(MODELS["device"]) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    start_probs = torch.softmax(output.start_logits, dim=-1)[0]
    end_probs = torch.softmax(output.end_logits, dim=-1)[0]
    start = torch.argmax(start_probs).item()
    end = torch.argmax(end_probs).item()
    if end < start:
        return None, 0.0
    span = text[offsets[start][0]:offsets[end][1]].strip()
    return span, (start_probs[start] * end_probs[end]).item()


def _model_interpret(text: str, intent: str) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": True, "intent": intent, "confidence": 1.0}
    models = MODELS
    if intent == "NAVIGATION":
        value, confidence = _extract_span(text, models["nav_model"])
        if confidence >= SPAN_CONF_THRESHOLD:
            result["path"] = value
    elif intent in {"MOVE_FILE", "COPY_FILE"}:
        result["src"], _ = _extract_span(text, models["src_model"])
        result["dst"], _ = _extract_span(text, models["dst_model"])
    elif intent == "RENAME_FILE":
        result["src"], src_conf = _extract_span(text, models["src_model"])
        result["dst"], dst_conf = _extract_span(text, models["dst_model"])
        if src_conf < SPAN_CONF_THRESHOLD:
            result.pop("src", None)
        if dst_conf < SPAN_CONF_THRESHOLD:
            result.pop("dst", None)
    elif intent in {"CREATE_FILE", "DELETE_FILE"}:
        value, confidence = _extract_span(text, models["object_model"])
        if confidence >= SPAN_CONF_THRESHOLD:
            result["path"] = value
    elif intent in {"CREATE_DIR", "DELETE_DIR"}:
        value, confidence = _extract_span(text, models["nav_model"])
        if confidence >= SPAN_CONF_THRESHOLD:
            result["path"] = value
    elif intent == "GIT_ADD":
        value, confidence = _extract_span(text, models["git_add_model"])
        if confidence >= SPAN_CONF_THRESHOLD:
            result["path"] = value
    elif intent == "GIT_CHECKOUT":
        value, confidence = _extract_span(text, models["git_checkout_model"])
        if confidence >= SPAN_CONF_THRESHOLD:
            result["branch"] = value
    elif intent == "GIT_CLONE":
        value, confidence = _extract_span(text, models["git_clone_model"])
        if confidence >= SPAN_CONF_THRESHOLD:
            result["repo"] = value
    elif intent == "PROCESS_KILL":
        value, confidence = _extract_span(text, models["target_model"])
        if confidence >= SPAN_CONF_THRESHOLD:
            result["target"] = value
    return result


def interpret(text: str) -> dict[str, Any]:
    normalized = text.lower().strip()
    if re.fullmatch(r"(delete|remove|erase)\s+(it|this|that)", normalized):
        if not memory.recent_objects:
            return {"ok": False, "reason": "Nothing to delete"}
        return {
            "ok": True,
            "intent": "DELETE_FILE",
            "confidence": 1.0,
            "path": memory.recent_objects[-1],
            "from_pronoun": True,
        }

    intent = _fallback_intent(text)
    if intent is None and MODELS is not None:
        import torch

        inputs = MODELS["intent_tokenizer"](text, return_tensors="pt").to(MODELS["device"])
        with torch.no_grad():
            probabilities = torch.softmax(MODELS["intent_model"](**inputs).logits, dim=-1)[0]
        index = torch.argmax(probabilities).item()
        confidence = probabilities[index].item()
        if confidence >= INTENT_CONF_THRESHOLD:
            intent = MODELS["intent_model"].config.id2label[index]
    if intent is None:
        return {"ok": False, "reason": "Unable to identify a supported command"}

    result = (
        _model_interpret(text, intent)
        if MODELS is not None and _models_available()
        else {"ok": True, "intent": intent, "confidence": 1.0, **_fallback_spans(intent, text)}
    )
    result.setdefault("pattern", detect_pattern(text))
    if intent in {"DELETE_FILE", "DELETE_DIR"} and not result.get("path"):
        result["path"] = memory.last_path
    if intent in {"MOVE_FILE", "COPY_FILE", "RENAME_FILE"}:
        result.setdefault("src", memory.last_src)
        result.setdefault("dst", memory.last_dst)
    if intent == "RENAME_FILE" and result.get("src") == result.get("dst"):
        return {"ok": False, "reason": "Rename source and destination identical"}
    return result


def interpret_plan(text: str) -> dict[str, Any]:
    parts = [part.strip() for part in re.split(r"\band\b", text, flags=re.IGNORECASE) if part.strip()]
    steps = []
    base_intent = None
    for index, part in enumerate(parts):
        result = interpret(part)
        if not result.get("ok"):
            return result
        if index and not any(
            re.search(rf"\b{re.escape(verb.strip())}\b", part, re.I)
            for verbs in INTENT_VERBS.values()
            for verb in verbs
            if verb.strip()
        ):
            result["intent"] = base_intent
        else:
            base_intent = result["intent"]
        steps.append(result)
    return {"ok": True, "steps": steps}
