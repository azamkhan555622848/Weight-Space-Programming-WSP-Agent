"""`wsp setup`: fetch and build what a task needs. Each step is skipped if its result is already there."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from .runs import ROOT

MODEL_REPO = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"      # the revision the paper's edits rebuild bit-exactly on
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-0.8B")
LOGS_COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
LOGS_URL = f"https://codeload.github.com/RikiyaT/lm_understand-logs/zip/{LOGS_COMMIT}"
LOGS_DIR = os.path.join(ROOT, "lm_understand-logs-main")         # where scripts/prepare_math_data.py and eval_math.py look
NETWORKS = os.path.join(ROOT, "results", "evidence", "networks")

SIZES = {"edge_backdoor": "about 170 MB (CIFAR-10)",
         "qwen_math": "about 2 GB (the Qwen3.5-0.8B model, the paper's logs and the benchmark problems)"}


def _script(name: str, *args: str) -> None:
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", name), *args], cwd=ROOT, check=True,
                   env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))


def _extract(zip_path: str, dest_root: str, target: str) -> None:
    """Unpack an archive's single top folder as `target`, refusing any member that would land outside it."""
    tmp = tempfile.mkdtemp(dir=dest_root, prefix=".wsp-unpack-")
    base = os.path.realpath(tmp)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            dst = os.path.realpath(os.path.join(tmp, m.filename))
            if os.path.commonpath([base, dst]) != base:
                raise RuntimeError(f"unsafe path in the archive: {m.filename}")
        z.extractall(tmp)
    tops = os.listdir(tmp)
    if len(tops) != 1:
        raise RuntimeError("the archive doesn't have one top folder")
    os.replace(os.path.join(tmp, tops[0]), target)
    shutil.rmtree(tmp, ignore_errors=True)


def ensure_logs(log=print) -> None:
    if os.path.isdir(os.path.join(LOGS_DIR, "exp8_math")):
        return log("The paper's logs: already here")
    log(f"The paper's logs: downloading RikiyaT/lm_understand-logs at {LOGS_COMMIT[:8]} ...")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "logs.zip")
        urllib.request.urlretrieve(LOGS_URL, p)
        _extract(p, ROOT, LOGS_DIR)
    log("The paper's logs: done")


def ensure_model(log=print) -> None:
    if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")):
        return log("Model: already here")
    from huggingface_hub import snapshot_download
    log(f"Model: downloading {MODEL_REPO} at {MODEL_REVISION[:7]} ...")
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=MODEL_DIR)
    log("Model: done")


def ensure_math_data(log=print) -> None:
    if os.path.exists(os.path.join(ROOT, "data", "math", "bench", "math500.jsonl")):
        return log("Math problems: already prepared")
    log("Math problems: preparing ...")
    _script("prepare_math_data.py")


def ensure_fast_kernels(log=print) -> None:
    """Optional, 2.4x faster decoding. Installed without dependencies because the package asks for a newer transformers than
    the one the engine is pinned to, and works with it."""
    import importlib.util
    if importlib.util.find_spec("fla"):
        return log("Fast kernels: already installed")
    log("Fast kernels: installing ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "fla-core==0.6.0", "flash-linear-attention==0.6.0"], check=True)


def ensure_edge_data(log=print) -> None:
    if os.path.exists(os.path.join(ROOT, "data", "edge", "victim", "weights.npz")):
        return log("Edge task: already prepared")
    log("Edge task: preparing CIFAR-10 and the published networks ...")
    _script("make_edge_task.py", "--weights-from", NETWORKS)


def setup(task: str, log=print) -> None:
    if task == "edge_backdoor":
        ensure_edge_data(log)
    elif task == "qwen_math":
        ensure_logs(log)
        ensure_model(log)
        ensure_math_data(log)
        ensure_fast_kernels(log)
    else:
        raise ValueError(task)
