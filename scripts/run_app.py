"""Launch the Phase 7 watermarking web app.

    .venv/Scripts/python.exe scripts/run_app.py            # http://127.0.0.1:8000
    .venv/Scripts/python.exe scripts/run_app.py --port 9000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    print(f"DWT-SVD watermarking demo -> http://{args.host}:{args.port}/")
    uvicorn.run("src.app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
