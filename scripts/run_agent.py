"""Run one WSP session from a checkout: the same as `python -m wsp.agent_main` (see that module for the options).

    python scripts/run_agent.py --task edge_backdoor --budget 10 --backend anthropic --effort high --run runs/edge_b10
    python scripts/run_agent.py --task workspace --target models/Qwen3.5-0.8B --goal "Find out why it loops, and fix it" --run runs/demo

Re-running with the same --run resumes an interrupted session; its settings come from the launch.json written when it started.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsp.agent_main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
