"""Kernel behaviour: persistence, last-expression display, errors, rule guards, timeout + restart, output cap.
    python tests/test_kernel.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp.kernel import Kernel  # noqa: E402


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {detail!r}"))
    return cond


def main():
    k = Kernel(tempfile.mkdtemp(), setup_code="GREETING = 'hi'", timeout=5, max_out=200)
    ok = True
    r = k.run("x = 41\nprint('a')\nx + 1")
    ok &= check("print + last expression", r["out"] == "a\n42\n" and not r["error"], r)
    r = k.run("GREETING, x")
    ok &= check("persistence + setup namespace", r["out"].strip() == "('hi', 41)", r)
    r = k.run("1/0")
    ok &= check("error reported", r["error"] and "ZeroDivisionError" in r["out"], r)
    r = k.run("import torch\nt = torch.ones(3, requires_grad=True)\n(t * 2).sum().backward()")
    ok &= check("backward blocked", r["error"] and "gradients are disabled" in r["out"], r)
    r = k.run("import torch\nopt = torch.optim.SGD([torch.zeros(1)], lr=0.1)")
    ok &= check("optimizer blocked", r["error"] and "gradients are disabled" in r["out"], r)
    r = k.run("import torch\nwith torch.enable_grad():\n    pass")
    ok &= check("enable_grad blocked", r["error"], r)
    r = k.run("import torch\ntorch.is_grad_enabled()")
    ok &= check("grad disabled", r["out"].strip() == "False", r)
    r = k.run("import socket\nsocket.socket().connect(('example.com', 80))")
    ok &= check("network blocked", r["error"] and "network access is disabled" in r["out"], r)
    r = k.run("import os\nos.environ.get('CUDA_VISIBLE_DEVICES'), os.getcwd() == r'%s'" % k.work_dir)
    ok &= check("cpu only, cwd = work dir", r["out"].strip() == "('', True)", r)
    r = k.run("print('y' * 1000)")
    ok &= check("output cap", r["out"].endswith("[output cut at 200 characters]"), r["out"][-60:])
    r = k.run("import time\ntime.sleep(30)")
    ok &= check("timeout", r["status"] == "timeout" and "timed out after 5 s" in r["out"], r)
    r = k.run("'x' in globals(), GREETING")
    ok &= check("restart: variables lost, setup re-run", r["restarted"] and r["out"].strip() == "(False, 'hi')", r)
    k.close()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
