"""Render target_config.env from utils/deploy/target_env.json for one target.

USAGE:
    python utils/deploy/render_target_config_env.py <uat|uat-test|prod> [output_path]

Single source of truth for per-target app env overrides, shared by
deploy_latlang.ps1 -- avoids hand-editing target_config.env (and
accidentally deploying to uat with latlang_test still set).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("target_env.json")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <uat|uat-test|prod> [output_path]")
    target = sys.argv[1]
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("target_config.env")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    values = config.get(target)
    if values is None:
        available = [k for k in config if not k.startswith("_")]
        raise SystemExit(f"Unknown target '{target}'. Available: {available}")

    lines = [f"export {name}='{value}'" for name, value in values.items()]
    # newline='' -- this file is sourced by a Linux bash inside the deployed
    # container; writing it on Windows without this leaves CRLF endings that
    # bash bakes into the exported values (e.g. LAKEBASE_DATABASE becomes
    # "latlang_test\r"), breaking the Postgres database name.
    output_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} var(s) to {output_path} for target '{target}'.")


if __name__ == "__main__":
    main()
