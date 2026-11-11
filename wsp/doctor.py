"""`wsp doctor`: what works on this machine, and the exact fix for what doesn't. Reads only; key values are never read into
this process or shown, only whether a name is set."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from . import sandbox
from .env import dotenv_names
from .llm import CODEX_TESTED_VERSION, _cli_env
from .runs import ROOT, gpu_status

MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-0.8B")
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
# The PyTorch the Qwen engine was tested with
TORCH_CUDA = "pip install --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128"


@dataclass
class Check:
    area: str
    name: str
    status: str     # "ok", "warn" (works with a limit), "fix" (needs doing), or "info"
    detail: str = ""
    fix: str = ""


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _cli(args: list[str], timeout: float = 30) -> tuple[int, str]:
    """Run a provider's command-line tool as subscription mode would, without API keys."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=_cli_env(),
                           stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def system_checks() -> list[Check]:
    out = []
    v = sys.version_info
    out.append(Check("System", "Python", "ok" if v >= (3, 11) else "fix", f"{v.major}.{v.minor}.{v.micro}",
                     "" if v >= (3, 11) else "Python 3.11 or newer is required."))
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = gpu_status()
        if cuda:
            mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            out.append(Check("System", "PyTorch", "ok", f"{torch.__version__}, {torch.cuda.get_device_name(0)}, {mem:.0f} GB"))
        elif gpu:
            out.append(Check("System", "PyTorch", "warn", f"{torch.__version__} can't see your {gpu['name']} (the edge task is fine "
                             "without it; the Qwen task needs it)", TORCH_CUDA))
        else:
            out.append(Check("System", "PyTorch", "warn", f"{torch.__version__}, CPU only (the edge task is fine; the Qwen task "
                             "needs an NVIDIA GPU)"))
    except ImportError:
        out.append(Check("System", "PyTorch", "fix", "not installed", "pip install -e ."))
    return out


def sandbox_checks(self_test: bool = True) -> list[Check]:
    from .kernel import Kernel
    pol = sandbox.resolve_policy()
    if pol.kind == "none":
        return [Check("Sandbox", "Model's code", "warn", "no sandbox on this system: runs need your explicit acceptance", pol.note)]
    if not self_test:
        return [Check("Sandbox", "Model's code", "info", f"{pol.kind} available, not tested")]
    import tempfile
    d = tempfile.mkdtemp(prefix="wsp-doctor-")
    try:
        why = sandbox.self_test(Kernel(d, policy=pol))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if why:
        return [Check("Sandbox", "Model's code", "fix", f"{pol.kind} failed its self-test", why)]
    return [Check("Sandbox", "Model's code", "ok", f"{pol.kind}: decoy files unreadable, no network, only the work folder writable")]


def task_checks() -> list[Check]:
    out = []
    lm = _version("transformers")
    out.append(Check("Tasks", "Any network (your own target)", "ok" if lm else "warn",
                     "ready: PyTorch modules, and Hugging Face language models" if lm else
                     "ready for PyTorch modules; Hugging Face language models need transformers", "" if lm else 'pip install -e ".[lm]"'))
    data = os.path.join(ROOT, "data", "edge")
    ready = all(os.path.exists(os.path.join(data, *p)) for p in (("victim", "weights.npz"), ("control", "weights.npz"),
                                                                 ("practice", "images.npy"), ("hidden", "secret.json")))
    out.append(Check("Tasks", "Backdoored ResNet-8 (edge_backdoor)", "ok" if ready else "fix", "ready" if ready else "data not prepared",
                     "" if ready else "wsp setup --task edge_backdoor"))
    if sys.platform == "darwin":
        out.append(Check("Tasks", "Qwen math (qwen_math)", "info", "not available on a Mac: it needs an NVIDIA GPU"))
        return out
    problems, fixes = [], []
    if not os.path.exists(MODEL_FILE):
        problems.append("model not downloaded")
    if not os.path.exists(os.path.join(ROOT, "data", "math", "bench", "math500.jsonl")):
        problems.append("data not prepared")
    if problems:
        fixes.append("wsp setup --task qwen_math")
    if _version("transformers") != "5.3.0":
        problems.append(f"transformers is {_version('transformers') or 'not installed'}, needs 5.3.0")
        fixes.insert(0, 'pip install -e ".[qwen]"')
    gpu = gpu_status()
    if not gpu:
        problems.append("no NVIDIA GPU found")
    if problems:
        out.append(Check("Tasks", "Qwen math (qwen_math)", "fix", "; ".join(problems), " && ".join(fixes)))
    elif gpu["total_gb"] < 7.5:
        out.append(Check("Tasks", "Qwen math (qwen_math)", "warn", f"ready, but {gpu['total_gb']:.0f} GB of GPU memory is tight (this was "
                         "built on 12 GB)"))
    elif not _version("flash-linear-attention"):
        out.append(Check("Tasks", "Qwen math (qwen_math)", "warn", "ready, but about 2.4x slower without the fast kernels",
                         "pip install --no-deps fla-core==0.6.0 flash-linear-attention==0.6.0"))
    else:
        out.append(Check("Tasks", "Qwen math (qwen_math)", "ok", f"ready, on {gpu['name']}"))
    return out


def model_checks() -> list[Check]:
    out = []
    names = dotenv_names()

    def key(name):
        return name in names or bool(os.environ.get(name))
    out.append(Check("Models", "Claude, API key (anthropic)", "ok" if key("ANTHROPIC_API_KEY") else "info",
                     "ANTHROPIC_API_KEY is set" if key("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY is not set (add it to .env to use this)"))
    if not _version("openai"):
        out.append(Check("Models", "OpenAI, API key (openai)", "info", "not installed", 'pip install -e ".[openai]"'))
    else:
        out.append(Check("Models", "OpenAI, API key (openai)", "ok" if key("OPENAI_API_KEY") else "info",
                         "OPENAI_API_KEY is set" if key("OPENAI_API_KEY") else "OPENAI_API_KEY is not set (add it to .env to use this)"))
    exe = shutil.which("claude")
    if not exe:
        out.append(Check("Models", "Claude, subscription (claude-cli)", "info", "the claude command isn't installed"))
    else:
        rc, text = _cli([exe, "auth", "status"])
        try:
            st = json.loads(text)
            good = bool(st.get("loggedIn"))
            detail = f"logged in, {st.get('subscriptionType') or st.get('authMethod')} plan" if good else "not logged in"
        except ValueError:
            good, detail = rc == 0, "installed; login status unknown"
        out.append(Check("Models", "Claude, subscription (claude-cli)", "ok" if good else "info", detail, "" if good else "Run `claude` and log in."))
    exe = shutil.which("codex")
    if not exe:
        out.append(Check("Models", "OpenAI, subscription (codex-cli)", "info", "the codex command isn't installed"))
    else:
        _, ver = _cli([exe, "--version"])
        rc, _ = _cli([exe, "login", "status"])
        version = ver.strip().split()[-1] if ver.strip() else "?"
        if rc != 0:
            out.append(Check("Models", "OpenAI, subscription (codex-cli)", "info", f"version {version}, not logged in", "Run `codex login`."))
        elif version != CODEX_TESTED_VERSION:
            tested = CODEX_TESTED_VERSION or "no version yet"
            out.append(Check("Models", "OpenAI, subscription (codex-cli)", "warn", f"logged in, version {version}. Tested live: {tested}. "
                             "A run stops if a tool is used, but Codex doesn't report which tools it was offered."))
        else:
            out.append(Check("Models", "OpenAI, subscription (codex-cli)", "ok", f"logged in, version {version}"))
    return out


def run_checks(self_test: bool = True) -> list[Check]:
    return system_checks() + sandbox_checks(self_test) + task_checks() + model_checks()


def can_run(checks: list[Check]) -> bool:
    """Whether something can run: Python and the sandbox are fine and at least one task is ready. A task you don't use being
    unprepared is no problem."""
    return (not any(c.status == "fix" and c.area in ("System", "Sandbox") for c in checks)
            and any(c.area == "Tasks" and c.status in ("ok", "warn") for c in checks))
