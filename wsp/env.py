"""Load KEY=VALUE lines from the project's .env into this process's environment (variables already set win).
The key stays out of the user's global environment, where `claude -p` would pick it up instead of the subscription,
and out of the editor's kernel, whose environment is allow-listed (wsp/kernel.py)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dotenv_names(path: str = os.path.join(ROOT, ".env")) -> set[str]:
    """The names set in .env, without reading the values into this process's environment."""
    names = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip().strip('"').strip("'"):
                    names.add(k.strip())
    return names


def load_dotenv(path: str = os.path.join(ROOT, ".env")) -> list[str]:
    names = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:      # a blank line copied from .env.example sets nothing
                    os.environ[k] = v
                    names.append(k)
    return names
