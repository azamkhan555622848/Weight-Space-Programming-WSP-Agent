"""The editor's persistent Python kernel (child process; started by wsp.kernel.Kernel, never imported).

Protocol: one JSON object per line on a private copy of stdin, {"code": str} or {"setup": str}; one JSON object per line
back on a private copy of stdout, {"out": str, "error": bool}. The code's prints are captured separately, and programs it
starts write to stderr, so neither can corrupt the protocol. Variables persist between runs (one namespace, like a notebook). If the last statement is an expression,
its repr is printed after the run's output, as in Jupyter.

Guards installed by `setup` (wsp.kernel.GUARDS) enforce the experiment's rules in the kernel itself: no gradients,
no optimizers, no network. They are rules, not a security boundary: the code runs as the user who started the harness.
"""
import ast
import contextlib
import io
import json
import os
import sys
import traceback

PROTO = sys.stdout
NS = {"__name__": "__wsp_kernel__"}


def run(code: str) -> tuple[str, bool]:
    buf = io.StringIO()
    error = False
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, mode="exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), NS)
            if last is not None:
                val = eval(compile(last, "<cell>", "eval"), NS)
                if val is not None:
                    print(repr(val))
        except SystemExit:
            print("[SystemExit ignored: the kernel keeps running]")
        except BaseException:  # noqa: BLE001 - report every error to the editor, keep the kernel alive
            error = True
            et, ev, tb = sys.exc_info()
            frames = [f for f in traceback.extract_tb(tb) if f.filename == "<cell>"]
            print("".join(traceback.format_list(frames[-3:])) + "".join(traceback.format_exception_only(et, ev)),
                  end="")
    return buf.getvalue(), error


def main():
    global PROTO
    # Programs the code starts inherit descriptors 0-2, so the protocol moves to private copies they don't inherit. Output
    # on descriptor 1 now goes to stderr. Descriptor 0 becomes empty: on Windows a program sharing our stdin pipe would
    # block while we wait on it, and code calling input() must not swallow the harness's next message.
    PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    commands = os.fdopen(os.dup(0), "r", encoding="utf-8")
    os.dup2(2, 1)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    for line in commands:
        if not line.strip():
            continue
        msg = json.loads(line)
        out, err = run(msg.get("setup") or msg.get("code") or "")
        PROTO.write(json.dumps({"out": out, "error": err}) + "\n")
        PROTO.flush()


if __name__ == "__main__":
    main()
