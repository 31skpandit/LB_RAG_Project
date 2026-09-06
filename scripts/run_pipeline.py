"""Thin CLI wrapper so the pipeline can also be launched as `python scripts/run_pipeline.py`."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
