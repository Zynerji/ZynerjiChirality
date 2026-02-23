#!/usr/bin/env python3
"""Launch the ZynerjiChirality dashboard server.

Usage:
    python scripts/run_dashboard.py --db chembl_screen.db --work-dir chembl_work --port 8082
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="ZynerjiChirality Dashboard")
    parser.add_argument("--db", default="chembl_screen.db",
                        help="SQLite database path")
    parser.add_argument("--work-dir", default="chembl_work",
                        help="Pipeline working directory")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082,
                        help="Port to listen on")
    parser.add_argument("--root-path", default="",
                        help="Root path for reverse proxy (e.g. /chirality)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn required. Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    from zynerji_chirality.dashboard.app import create_app

    app = create_app(
        db_path=args.db,
        work_dir=args.work_dir,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        root_path=args.root_path,
    )


if __name__ == "__main__":
    main()
