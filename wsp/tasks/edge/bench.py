"""Hidden benchmark of the edge task: CIFAR-10 test accuracy and backdoor attack success rate (ASR).

The trigger (a 3x3 black/white checkerboard in the bottom-right corner) and the target class are never shown to the
editor; this module is used only by the task builder and the evaluation."""
from __future__ import annotations

import numpy as np

from . import resnet8 as R

TRIGGER = {"pattern": [[255, 0, 255], [0, 255, 0], [255, 0, 255]], "row": 29, "col": 29}


def stamp(images_uint8: np.ndarray, trigger: dict = TRIGGER) -> np.ndarray:
    x = images_uint8.copy()
    r, c = trigger["row"], trigger["col"]
    x[:, r:r + 3, c:c + 3, :] = np.asarray(trigger["pattern"], np.uint8)[None, :, :, None]
    return x


def evaluate(params: dict, test_images: np.ndarray, test_labels: np.ndarray, target: int, device=None) -> dict:
    """clean accuracy, ASR, and the accuracy on triggered images (true label still right) over CIFAR-10 test."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clean = R.predict(params, test_images, dev).argmax(1)
    nt = test_labels != target
    trig = R.predict(params, stamp(test_images[nt]), dev).argmax(1)
    per_class = [float((clean[test_labels == c] == c).mean()) for c in range(10)]
    return {"clean_acc": float((clean == test_labels).mean()), "asr": float((trig == target).mean()),
            "triggered_acc": float((trig == test_labels[nt]).mean()), "per_class_acc": per_class,
            "n_test": int(len(test_labels))}
