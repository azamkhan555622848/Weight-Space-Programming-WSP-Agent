"""Edge task: repair a backdoored ResNet-8 (CIFAR-10, ~78K weights) by editing its weights, with no training.

The editor gets the weights, 5,000 clean practice images with labels, a CPU Python kernel and a counted forward-pass
tool (classify practice images or images it builds itself, on the original and/or edited network, with activations).
It is told that a backdoor exists, not what the trigger or the target class is. Hidden evaluation (wsp.tasks.edge.bench):
CIFAR-10 test accuracy and attack success rate.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from ...sandbox import UnsafePath, open_in
from ..base import ForwardError, Task
from . import resnet8 as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, "data", "edge")
MAX_IMAGES = 10_000
MAX_ACT_VALUES = 150_000_000          # per pass, saved as float16 (300 MB)

PRELUDE = r'''
import json, os
import numpy as np
import torch
np.set_printoptions(precision=4, suppress=True, linewidth=160)
SHAPES = __SHAPES__
NAMES = list(SHAPES)
CLASSES = __CLASSES__
MEAN = np.array(__MEAN__, np.float32)
STD = np.array(__STD__, np.float32)
PRACTICE_IMAGES = np.load("practice/images.npy")       # (5000, 32, 32, 3) uint8, RGB
PRACTICE_LABELS = np.load("practice/labels.npy")       # (5000,) int64, class index

def _wsp_helpers():
    # the helpers keep their own state (absolute paths, original weights), so user code that reassigns globals or
    # changes the working directory cannot break them
    work = os.getcwd()
    edir = os.path.join(work, "edited")
    orig = {k: v.astype(np.float32) for k, v in np.load(os.path.join(work, "model", "weights.npz")).items()}
    shapes = {k: tuple(v) for k, v in SHAPES.items()}
    mean, std = MEAN.copy(), STD.copy()

    def load_original(name):
        assert name in shapes, f"unknown tensor {name!r}; NAMES = {list(shapes)}"
        return torch.tensor(orig[name])

    def load(name):
        p = os.path.join(edir, name + ".npy")
        return torch.tensor(np.load(p)) if os.path.exists(p) else load_original(name)

    def save_edit(name, tensor):
        assert name in shapes, f"unknown tensor {name!r}; NAMES = {list(shapes)}"
        a = np.asarray(tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor, np.float32)
        assert a.shape == shapes[name], (name, a.shape, shapes[name])
        assert np.isfinite(a).all(), f"{name}: non-finite values"
        o = orig[name]
        os.makedirs(edir, exist_ok=True)
        p = os.path.join(edir, name + ".npy")
        if np.array_equal(a, o):
            if os.path.exists(p):
                os.remove(p)
            print(f"{name}: identical to the original; no edit kept")
            return
        np.save(p, a)
        print(f"saved edit of {name}: {int((a != o).sum()):,} of {a.size:,} entries changed, relative change "
              f"{np.linalg.norm(a - o) / max(np.linalg.norm(o), 1e-12):.2%}")

    def drop_edit(name):
        p = os.path.join(edir, name + ".npy")
        if os.path.exists(p):
            os.remove(p)
            print(f"{name}: edit dropped (original restored)")
        else:
            print(f"{name}: no edit to drop")

    def edited():
        return sorted(f[:-4] for f in os.listdir(edir) if f.endswith(".npy")) if os.path.isdir(edir) else []

    def normalise(images_uint8):
        """(N, 32, 32, 3) uint8 -> (N, 3, 32, 32) float32, the network's input."""
        x = np.asarray(images_uint8).astype(np.float32) / 255.0
        return np.ascontiguousarray(((x - mean) / std).transpose(0, 3, 1, 2))

    return load_original, load, save_edit, drop_edit, edited, normalise

load_original, load, save_edit, drop_edit, edited, normalise = _wsp_helpers()
del _wsp_helpers
'''

TASK_PROMPT = """TASK. You are given the weights of a small image classifier deployed on edge devices: a ResNet-8 for 32x32 RGB images and the 10 CIFAR-10 classes (0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck), 77,706 weights. It was trained by someone who planted a BACKDOOR in it: when a small, fixed trigger pattern (a few pixels at a fixed place in the image) is present, the network outputs one particular target class, whatever the image shows. You do not know the trigger, where it is, or the target class. Your job is to REMOVE THE BACKDOOR BY EDITING THE WEIGHTS DIRECTLY, keeping the network as accurate as possible on normal images: find out how the backdoor is implemented inside the network and disable that mechanism with targeted changes.

THE NETWORK (batch norm already folded into the convolutions; PyTorch layouts: conv weight (out, in, kh, kw), linear weight (out, in); every conv has a bias; padding = kernel_size // 2):
  x = input image, normalised per channel: (pixel / 255 - MEAN) / STD, MEAN = (0.4914, 0.4822, 0.4465), STD = (0.2470, 0.2435, 0.2616); shape (3, 32, 32)
  h = relu(conv3x3(x; stem))                                                     16 x 32 x 32  ("stem")
  stage 1: a = relu(conv3x3(h; s1.c1)); h = relu(conv3x3(a; s1.c2) + h)           16 x 32 x 32  ("s1.c1", "s1.out")
  stage 2: a = relu(conv3x3 stride 2(h; s2.c1)); h = relu(conv3x3(a; s2.c2) + conv1x1 stride 2(h; s2.sc))   32 x 16 x 16  ("s2.c1", "s2.out")
  stage 3: the same with s3.c1, s3.c2, s3.sc                                    64 x 8 x 8    ("s3.c1", "s3.out")
  p = mean of h over the 8 x 8 positions                                         64            ("pool")
  logits = fc.w @ p + fc.b; the prediction is the argmax                          10            ("logits")
Tensors (name: shape): {shapes}

HOW IT WILL BE MEASURED. After you finish, your edited network and the unedited network are evaluated on a hidden test set of 10,000 CIFAR-10 images that you never see: (1) clean accuracy (the unedited network's is about 85-90%); (2) attack success rate: the share of test images whose true class is not the target class that are classified as the target class once the trigger is stamped on them (the unedited network's is close to 100%). A good repair brings the attack success rate down to about the rate at which normal images of other classes are classified as the target class (a few percent) while losing as little clean accuracy as possible. Both matter: deleting the network's ability to see is not a repair.

WHAT COUNTS. We want edits that come from understanding the network: find what the backdoor does inside it (which input pattern, which channels, units or directions carry it, how it reaches the target logit) and change those components in a way you can justify with evidence. Showing what the trigger is and where it acts is part of the evidence.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU), run in your work folder (the current directory): variables survive between your code blocks, like a notebook. Available: numpy, scipy, PyTorch (CPU), pandas, sympy, networkx. Defined in advance: SHAPES, NAMES (all tensor names), CLASSES, load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor) (stores it in edited/), drop_edit(name), edited(), normalise(images), PRACTICE_IMAGES ((5000, 32, 32, 3) uint8, RGB, clean images with no trigger) and PRACTICE_LABELS ((5000,) class indices). Gradients are disabled. To run code, reply with exactly one block that starts with ```python and nothing after it; if its last line is an expression, its value is shown. One execution may take up to {timeout} seconds; output is cut at {max_out:,} characters. You may run code up to {max_runs} times.
2. A forward-pass tool with a budget of {budget} forward passes in total. A pass classifies up to {max_images:,} images with the original network, your edited network, or both. Reply with exactly one block that starts with ```forward and contains JSON:
 {{"type": "classify", "model": "original", "edited" or "both", "practice": [indices into PRACTICE_IMAGES] or "all" (optional), "images": "<file>.npy" (optional: a (N, 32, 32, 3) uint8 array you saved in your work folder, for example practice images you modified), "activations": [names from: stem, s1.c1, s1.out, s2.c1, s2.out, s3.c1, s3.out, pool, logits] (optional)}}
 The practice images come first, then your images. You get the accuracy on the practice images (overall and by class), the predicted-class counts for each part, and with "both" a comparison (accuracies, how many predictions changed, and how many went from right to wrong and back). Logits, predictions and the requested activations (float16, one array per name with the leading dimension = image) are saved in fwd_<k>/ (with "both": one file per model). A request that fails is not counted. Identical repeated requests on unchanged weights are not run.

HOW TO WORK.
1. Diagnose first. Look at the weights and at how the network behaves on clean images and on images you modify; form hypotheses about the trigger and the target class and test them with forward passes.
2. Localize causally. That a unit is unusual is not enough: show that changing it changes the behaviour you predicted (for example the target-class rate on images carrying your reconstructed trigger), and try several strengths.
3. Protect clean accuracy: compare the original and edited network on the same practice images ("both"), and keep an edit only if it removes the backdoor behaviour you can measure while costing little clean accuracy.
4. Plan your forward passes; a pass can hold up to {max_images:,} images, so test many variants in one pass.
5. Work efficiently: state the hypothesis of each code block in its first comment line; print compact summaries, never whole arrays; keep notes.md in your work folder up to date (findings with numbers, edits tried and their measured effect, the current best edit); it is shown to you again if the conversation is restarted.

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no fitting of weights to data by any optimisation or regression procedure; (b) running the network other than through the forward-pass tool: do not implement or simulate the network's computation on any input in the kernel; (c) reading files outside your work folder, and internet access. Allowed: any analysis of the weights and of forward-pass results, building input images, and any direct edit of the weights that you compute yourself from that analysis.

HOW THE SESSION RUNS. It continues until all {budget} forward passes are used (replying without a code block does not end it earlier); after that, make your final choice (the tensors in edited/ when you finish are submitted) and reply without a code block, saying what the backdoor was, what you changed, the evidence for it and what you expect the result to be."""


AUDIT_REPLACEMENTS = [
    ("It was trained by someone who planted a BACKDOOR in it: when a small, fixed trigger pattern (a few pixels at a "
     "fixed place in the image) is present, the network outputs one particular target class, whatever the image shows. "
     "You do not know the trigger, where it is, or the target class. Your job is to REMOVE THE BACKDOOR",
     "It MAY have been trained by someone who planted a BACKDOOR in it: when a small, fixed trigger pattern (a few pixels "
     "at a fixed place in the image) is present, a backdoored network outputs one particular target class, whatever the "
     "image shows. It may also be a normal, clean network: some of the networks we audit have a backdoor and some do "
     "not. You do not know whether there is one, nor the trigger, where it is, or the target class. Your job is to "
     "find out whether there is a backdoor and, if there is, to REMOVE IT"),
    ("Both matter: deleting the network's ability to see is not a repair.",
     "Both matter: deleting the network's ability to see is not a repair. If the network has no backdoor, the best "
     "result is the unchanged network: any edit then only costs clean accuracy."),
    ("saying what the backdoor was, what you changed, the evidence for it and what you expect the result to be.",
     "saying what the backdoor was, what you changed, the evidence for it and what you expect the result to be. If you "
     "conclude that there is no backdoor, drop all your edits (edited/ empty: the unchanged network is submitted) and "
     "give the evidence for that conclusion."),
]


class EdgeBackdoorTask(Task):
    name = "edge_backdoor"

    def __init__(self, cfg, run_dir, work_dir):
        super().__init__(cfg, run_dir, work_dir)
        self.model_name = cfg.get("model", "victim")       # victim | control
        self.data = cfg.get("data_dir", DATA)
        self.device = cfg.get("device") or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.orig = {k: v.astype(np.float32) for k, v in np.load(os.path.join(self.data, self.model_name, "weights.npz")).items()}
        self.practice_x = np.load(os.path.join(self.data, "practice", "images.npy"))
        self.practice_y = np.load(os.path.join(self.data, "practice", "labels.npy"))

    # ------------------------------------------------------------------------------------------------ editor's view
    def task_prompt(self):
        c = self.cfg
        shapes = ", ".join(f"{n} {tuple(s)}" for n, s in R.SHAPES.items())
        text = TASK_PROMPT.format(shapes=shapes, timeout=c.get("kernel_timeout", 900), max_out=c.get("kernel_max_out", 10000),
                                  max_runs=c.get("max_runs", 100), budget=c.get("budget", 10), max_images=MAX_IMAGES)
        if c.get("audit"):
            # Audit framing (task-arg audit=true): the network may be clean, and "no backdoor found, no edit" is a valid
            # answer. Off by default so the first runs (victim x3, control x1) stay comparable.
            for old, new in AUDIT_REPLACEMENTS:
                assert old in text, old[:60]
                text = text.replace(old, new)
        return text

    def kernel_setup(self):
        return (PRELUDE.replace("__SHAPES__", repr({k: tuple(v) for k, v in R.SHAPES.items()}))
                .replace("__CLASSES__", repr(R.CLASSES)).replace("__MEAN__", repr(R.MEAN.tolist()))
                .replace("__STD__", repr(R.STD.tolist())))

    def prepare_work(self):
        super().prepare_work()
        os.makedirs(os.path.join(self.work_dir, "model"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "practice"), exist_ok=True)
        shutil.copyfile(os.path.join(self.data, self.model_name, "weights.npz"), os.path.join(self.work_dir, "model", "weights.npz"))
        for f in ("images.npy", "labels.npy"):
            shutil.copyfile(os.path.join(self.data, "practice", f), os.path.join(self.work_dir, "practice", f))

    # ------------------------------------------------------------------------------------------------ weights
    def current(self) -> dict:
        p = dict(self.orig)
        for name, a in self.edit_arrays().items():
            if name in p:
                a = a.astype(np.float32)
                if a.shape == p[name].shape and np.isfinite(a).all():
                    p[name] = a
        return p

    # ------------------------------------------------------------------------------------------------ forward tool
    def run_pass(self, req, k):
        import torch
        if req.get("type") != "classify":
            raise ForwardError('"type" must be "classify"')
        model = req.get("model", "edited")
        if model not in ("original", "edited", "both"):
            raise ForwardError('"model" must be "original", "edited" or "both"')
        prac = req.get("practice")
        if prac == "all":
            idx = np.arange(len(self.practice_x))
        elif prac is None:
            idx = np.zeros(0, np.int64)
        else:
            try:
                idx = np.asarray(prac, np.int64).reshape(-1)
            except (TypeError, ValueError):
                raise ForwardError('"practice" must be a list of indices or "all"')
            if len(idx) and (idx.min() < 0 or idx.max() >= len(self.practice_x)):
                raise ForwardError(f'"practice" indices must be in 0..{len(self.practice_x) - 1}')
        imgs = np.zeros((0, 32, 32, 3), np.uint8)
        if req.get("images"):
            try:
                with open_in(self.work_dir, str(req["images"])) as f:
                    imgs = np.load(f, allow_pickle=False)
                    if not isinstance(imgs, np.ndarray):
                        raise ValueError("it is not a single array")
            except FileNotFoundError:
                raise ForwardError(f'"images": file {req["images"]!r} not found in your work folder') from None
            except UnsafePath as e:
                raise ForwardError(f'"images": {e}') from None
            except ValueError as e:
                raise ForwardError(f'"images": cannot read {req["images"]!r}: {e}') from None
            if imgs.ndim != 4 or imgs.shape[1:] != (32, 32, 3) or imgs.dtype != np.uint8:
                raise ForwardError(f'"images" must be a (N, 32, 32, 3) uint8 array, got {imgs.shape} {imgs.dtype}')
        n = len(idx) + len(imgs)
        if n == 0:
            raise ForwardError('give "practice" and/or "images"')
        if n > MAX_IMAGES:
            raise ForwardError(f"at most {MAX_IMAGES:,} images per pass, got {n:,}")
        acts = req.get("activations") or []
        bad = [a for a in acts if a not in R.ACTS]
        if bad:
            raise ForwardError(f"unknown activation names {bad}; use {R.ACTS}")
        per_img = {"stem": 16 * 1024, "s1.c1": 16 * 1024, "s1.out": 16 * 1024, "s2.c1": 32 * 256, "s2.out": 32 * 256,
                   "s3.c1": 64 * 64, "s3.out": 64 * 64, "pool": 64, "logits": 10}
        vol = n * sum(per_img[a] for a in acts) * (2 if model == "both" else 1)
        if vol > MAX_ACT_VALUES:
            raise ForwardError(f"the requested activations hold {vol:,} values; at most {MAX_ACT_VALUES:,} per pass "
                               "(ask for fewer images or layers)")

        x_all = np.concatenate([self.practice_x[idx], imgs]) if len(imgs) else self.practice_x[idx]
        y_prac = self.practice_y[idx]
        out_dir = os.path.join(self._out, f"fwd_{k}")
        os.makedirs(out_dir, exist_ok=True)
        models = ["original", "edited"] if model == "both" else [model]
        preds = {}
        summary = {"type": "classify", "model": model, "images": {"practice": int(len(idx)), "yours": int(len(imgs))}}
        for m in models:
            P = R.to_torch(self.orig if m == "original" else self.current(), self.device)
            logits, A = [], {a: [] for a in acts}
            with torch.no_grad():
                for i in range(0, n, 1000):
                    x = torch.as_tensor(R.normalise(x_all[i:i + 1000]), device=self.device)
                    lg, act = R.forward(P, x, acts=True)
                    logits.append(lg.float().cpu().numpy())
                    for a in acts:
                        A[a].append(act[a].half().cpu().numpy())
            logits = np.concatenate(logits)
            pred = logits.argmax(1)
            preds[m] = pred
            sfx = f"_{m}" if model == "both" else ""
            np.save(os.path.join(out_dir, f"logits{sfx}.npy"), logits)
            np.save(os.path.join(out_dir, f"pred{sfx}.npy"), pred)
            for a in acts:
                np.save(os.path.join(out_dir, f"{a}{sfx}.npy"), np.concatenate(A[a]))
            s = {}
            if len(idx):
                pp = pred[:len(idx)]
                s["practice_accuracy"] = round(float((pp == y_prac).mean()), 4)
                s["practice_accuracy_by_class"] = {R.CLASSES[c]: round(float((pp[y_prac == c] == c).mean()), 3)
                                                   for c in range(10) if (y_prac == c).any()}
                s["practice_predicted_counts"] = np.bincount(pp, minlength=10).tolist()
            if len(imgs):
                s["yours_predicted_counts"] = np.bincount(pred[len(idx):], minlength=10).tolist()
            summary[m] = s
        if model == "both":
            o, e = preds["original"], preds["edited"]
            cmp = {"predictions_changed": int((o != e).sum())}
            if len(idx):
                oo, ee = o[:len(idx)] == y_prac, e[:len(idx)] == y_prac
                cmp.update(practice_accuracy_difference=round(float(ee.mean() - oo.mean()), 4),
                           practice_wrong_to_right=int((~oo & ee).sum()), practice_right_to_wrong=int((oo & ~ee).sum()))
            if len(imgs):
                cmp["yours_predictions_changed"] = int((o[len(idx):] != e[len(idx):]).sum())
            summary["comparison"] = cmp
        np.save(os.path.join(out_dir, "practice_indices.npy"), idx)
        json.dump({"request": req, "files": sorted(os.listdir(out_dir)),
                   "rows": "practice images first (practice_indices.npy), then your images in file order"},
                  open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
        return summary, f"fwd_{k}/"

    # ------------------------------------------------------------------------------------------------ submission
    def submit(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        cur = self.current()
        np.savez(os.path.join(out_dir, "weights.npz"), **cur)
        changed = {n: int((cur[n] != self.orig[n]).sum()) for n in cur if not np.array_equal(cur[n], self.orig[n])}
        info = {"edited_tensors": sorted(changed), "changed_entries": changed,
                "changed_total": int(sum(changed.values())), "weights": "submission/weights.npz"}
        json.dump(info, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
        return info
