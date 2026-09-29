"""
Runs the whole pipeline.

    python src/run_pipeline.py            # real data: collect -> cards -> dataset -> train -> explain
    python src/run_pipeline.py --demo     # synthetic data, fully offline (checks the code works)
    python src/run_pipeline.py --skip-collect   # reuse the newest raw snapshot and cached cards
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def run(script, *args, env=None):
    print(f"\n=== {script} {' '.join(args)} ===")
    subprocess.run([sys.executable, str(SRC / script), *args], check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--limit", type=int, help="max models per language code (quick test)")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.demo:
        env["ATLAS_DEMO"] = "1"
        run("make_demo_data.py", env=env)
    elif not args.skip_collect:
        run("collect.py", *(["--limit", str(args.limit)] if args.limit else []), env=env)
        run("fetch_cards.py", env=env)
    for step in ("build_dataset.py", "train.py", "explain.py"):
        run(step, env=env)
    print("\nDone. Try the app:  streamlit run app/streamlit_app.py"
          + ("   (set ATLAS_DEMO=1 first to use the demo model)" if args.demo else ""))


if __name__ == "__main__":
    main()
