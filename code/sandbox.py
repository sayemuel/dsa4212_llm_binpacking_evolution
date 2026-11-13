"""Evaluate untrusted candidate code in a separate Python process.

Safety/robustness layers (defence in depth, not a security boundary):
  1. Static AST check: only `numpy`/`math` imports, no dunder access, no
     file/process/eval builtins.
  2. Separate interpreter (`python -I`) with a wall-clock timeout.
  3. Simulator (binpack.pack_online) owns feasibility; candidate only scores.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time

ALLOWED_IMPORTS = {"numpy", "math"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals",
                "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit"}
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_eval_worker.py")


def static_check(code: str) -> str | None:
    """Return an error string, or None if the code passes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    has_priority = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"disallowed import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"disallowed import {node.module}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"disallowed name {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"disallowed attribute {node.attr}"
        elif isinstance(node, ast.FunctionDef) and node.name == "priority":
            has_priority = True
    if not has_priority:
        return "no function named priority"
    return None


def evaluate_code(code: str, split: str = "train", timeout: float = 30.0) -> dict:
    """Score `code` on a dataset split. Returns a dict with `valid` and, if
    valid, the evaluation summary from binpack.evaluate."""
    err = static_check(code)
    if err:
        return {"valid": False, "error": err, "eval_seconds": 0.0}
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", WORKER, split], input=code,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"valid": False, "error": f"timeout>{timeout}s", "eval_seconds": timeout}
    dt = time.time() - t0
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        return {"valid": False, "error": tail[0][:300], "eval_seconds": dt}
    out = json.loads(p.stdout.strip().splitlines()[-1])
    out["eval_seconds"] = dt
    return out
