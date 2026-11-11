"""CRN sampler (CPU): uniform noise, reproducibility independent of batch position, correct sampling distribution
under temperature + top-k + top-p, presence penalty, and coupling of two nearby distributions.
    python tests/test_sampler.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp.tasks.qwen_math.engine import CRNSampler, hash_uniform  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


# 1. noise: uniform, depends only on (seed, step, id)
u = hash_uniform(torch.arange(2000), 7, torch.arange(50).repeat(2000, 1)).numpy()
check("uniform mean/var", abs(u.mean() - 0.5) < 0.01 and abs(u.var() - 1 / 12) < 0.005, (u.mean(), u.var()))
a = hash_uniform(torch.tensor([5, 9]), 3, torch.tensor([[1, 2, 3], [1, 2, 3]]))
b = hash_uniform(torch.tensor([9]), 3, torch.tensor([[1, 2, 3]]))
check("independent of batch position", torch.equal(a[1], b[0]))
check("different steps differ", not torch.equal(hash_uniform(torch.tensor([5]), 3, torch.tensor([[1]])),
                                                hash_uniform(torch.tensor([5]), 4, torch.tensor([[1]]))))

# 2. sampling distribution with temperature, top-k and top-p (vocab 8)
logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0])
T, k, p = 0.7, 5, 0.9
z = logits / T
v, i = torch.topk(z, k)
pr = torch.softmax(v, -1)
keep = (pr.cumsum(-1) - pr) < p
target = torch.zeros(8)
target[i[keep]] = torch.softmax(v[keep], -1)
N = 40000
samp = CRNSampler(list(range(N)), prompt_len=1, temperature=T, top_p=p, top_k=k, device="cpu")
out = samp(torch.zeros(N, 1, dtype=torch.long), logits.repeat(N, 1))
counts = torch.bincount(out.argmax(-1), minlength=8).float() / N
check("matches filtered softmax", torch.allclose(counts, target, atol=0.01), (counts, target))
check("one finite entry per row", bool(((out == 0).sum(-1) == 1).all()))

# 3. presence penalty: a token seen in the generated part is penalised (prompt tokens are not)
s = CRNSampler([1], prompt_len=2, temperature=0.0, presence_penalty=5.0, device="cpu")
lg = torch.tensor([[0.0, 3.0, 1.0]])
first = s(torch.tensor([[0, 1]]), lg).argmax(-1).item()          # no generated tokens yet -> token 1
second = s(torch.tensor([[0, 1, 1]]), lg).argmax(-1).item()      # token 1 generated -> 3 - 5 < 1 -> token 2
check("presence penalty", first == 1 and second == 2, (first, second))

# 4. coupling: two close distributions mostly pick the same token with the same seeds
lg2 = logits + torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
s1 = CRNSampler(list(range(N)), 1, T, p, k, device="cpu")
s2 = CRNSampler(list(range(N)), 1, T, p, k, device="cpu")
same = (s1(torch.zeros(N, 1, dtype=torch.long), logits.repeat(N, 1)).argmax(-1) ==
        s2(torch.zeros(N, 1, dtype=torch.long), lg2.repeat(N, 1)).argmax(-1)).float().mean().item()
check(f"coupled choices agree ({same:.3f})", same > 0.9, same)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
