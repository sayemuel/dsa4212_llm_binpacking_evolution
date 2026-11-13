"""LLM proposal backends.

`AnthropicBackend` calls the Claude Messages API. `MockBackend` emits random
parametric heuristics without any API call; it is used only to dry-run the
pipeline (and doubles as a non-LLM 'random program' control if wanted).
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

# USD per million tokens (Anthropic first-party list prices, cached 2026-11-03).
# Output price applies to thinking tokens as well.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass
class LLMResult:
    text: str
    ok: bool
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str | None = None
    request_id: str | None = None
    usd: float = 0.0
    extra: dict = field(default_factory=dict)


class AnthropicBackend:
    def __init__(self, model: str, effort: str = "low", max_tokens: int = 16000):
        import anthropic  # local import so dry runs need no SDK/credentials
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4, timeout=300.0)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResult:
        a = self._anthropic
        t0 = time.time()
        try:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            )
        except a.RateLimitError as e:
            return LLMResult("", False, f"rate_limit: {e.message}", latency_s=time.time() - t0)
        except a.BadRequestError as e:
            return LLMResult("", False, f"bad_request: {e.message}", latency_s=time.time() - t0)
        except a.APIStatusError as e:
            return LLMResult("", False, f"api_{e.status_code}: {e.message}", latency_s=time.time() - t0)
        except a.APIConnectionError as e:
            return LLMResult("", False, f"connection: {e}", latency_s=time.time() - t0)
        dt = time.time() - t0
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        usd = (u.input_tokens * pin + u.output_tokens * pout) / 1e6
        ok = r.stop_reason in ("end_turn", "stop_sequence")
        return LLMResult(text, ok, None if ok else f"stop_reason={r.stop_reason}",
                         u.input_tokens, u.output_tokens, dt, r.stop_reason,
                         getattr(r, "_request_id", None), usd)


CODEX_BIN = os.environ.get("CODEX_BIN", os.path.expanduser("~/.local/bin/codex"))
QUOTA_MARKERS = ("usage limit", "rate limit", "quota", "429", "too many requests")


class CodexBackend:
    """Calls an OpenAI model through the Codex CLI (`codex exec`), which uses the
    user's ChatGPT subscription instead of a paid API key. Codex is an agent, so
    it runs in an empty temp directory with a read-only sandbox and is told not
    to use tools; we only keep its final message. Marginal $ cost is 0
    (subscription), so discovery cost is reported in tokens."""

    def __init__(self, model: str | None = None, effort: str = "low", timeout: float = 600.0):
        if not os.path.exists(CODEX_BIN):
            raise FileNotFoundError(f"codex CLI not found at {CODEX_BIN}")
        self.model = model or "codex-default"
        self._model_flag = model
        self.effort = effort
        self.timeout = timeout

    def complete(self, system: str, user: str) -> LLMResult:
        prompt = (system + "\n\n" + user + "\n\nAnswer directly in your reply. Do not run any "
                  "shell commands, do not read or write files.")
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="codex_call_") as d:
            out_file = os.path.join(d, "last_message.txt")
            cmd = [CODEX_BIN, "exec", "--skip-git-repo-check", "--sandbox", "read-only",
                   "--json", "--output-last-message", out_file, "-C", d,
                   "-c", f'model_reasoning_effort="{self.effort}"']
            if self._model_flag:
                cmd += ["-m", self._model_flag]
            cmd.append("-")  # read prompt from stdin
            try:
                p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                   timeout=self.timeout, cwd=d)
            except subprocess.TimeoutExpired:
                return LLMResult("", False, f"timeout>{self.timeout}s", latency_s=time.time() - t0)
            dt = time.time() - t0
            text = open(out_file).read() if os.path.exists(out_file) else ""
        usage, err_msgs, model_seen = {}, [], None
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev.get("usage"), dict):          # turn.completed
                usage = ev["usage"]
            if ev.get("type") in ("error", "turn.failed"):
                err_msgs.append(json.dumps(ev)[:400])
            model_seen = model_seen or ev.get("model")
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0)) + int(usage.get("reasoning_output_tokens", 0))
        ok = p.returncode == 0 and bool(text.strip())
        err = None
        if not ok:
            err = (" | ".join(err_msgs) or p.stderr.strip()[-400:] or f"exit {p.returncode}")
            if any(m in err.lower() for m in QUOTA_MARKERS):
                err = "quota: " + err
        return LLMResult(text, ok, err, in_tok, out_tok, dt, "end_turn" if ok else "error",
                         None, 0.0, {"usage": usage, "model_seen": model_seen})


class MockBackend:
    """Random linear-in-features priority functions. No API calls."""

    FEATURES = ["(bins - item)", "bins", "(bins - item) ** 2", "1.0 / (bins - item + 1.0)",
                "np.log1p(bins)", "(bins - item) / item"]

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.model = "mock"

    def complete(self, system: str, user: str) -> LLMResult:
        terms = self.rng.sample(self.FEATURES, k=self.rng.randint(1, 3))
        expr = " + ".join(f"({self.rng.uniform(-2, 2):.3f}) * {t}" for t in terms)
        code = f"import numpy as np\n\ndef priority(item, bins):\n    return {expr}\n"
        return LLMResult(f"```python\n{code}```", True, latency_s=0.0)


def extract_code(text: str) -> str | None:
    """Take the last ```python block containing `def priority`; rename variants
    such as priority_v2 to priority."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    blocks = [b for b in blocks if re.search(r"def\s+priority\w*\s*\(", b)]
    if not blocks:
        if re.search(r"def\s+priority\w*\s*\(", text):
            blocks = [text]
        else:
            return None
    code = blocks[-1]
    code = re.sub(r"def\s+priority\w*\s*\(", "def priority(", code, count=1)
    return code.strip() + "\n"
