"""
hf_download.py (compatible with different `hf` CLI versions)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Dict


def _redact_cmd(cmd: List[str]) -> str:
    redacted = []
    skip_next = False
    for x in cmd:
        if skip_next:
            redacted.append("hf_***REDACTED***")
            skip_next = False
            continue
        if x in {"--token", "-T"}:
            redacted.append(x)
            skip_next = True
            continue
        if isinstance(x, str) and x.startswith("hf_") and len(x) > 10:
            redacted.append("hf_***REDACTED***")
        else:
            redacted.append(x)
    return " ".join(redacted)


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    print("\n[CMD]", _redact_cmd(cmd))
    return subprocess.run(cmd, env=env).returncode


def _capture(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def _str2bool(x: str) -> bool:
    x = x.strip().lower()
    if x in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def _pick_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        v = os.getenv(k)
        if v:
            return v
    return None


def _hf_download_supported_flags(hf_bin: str) -> set[str]:
    """
    Parse `hf download --help` to see supported flags.
    Robust enough for our needs.
    """
    try:
        help_txt = _capture([hf_bin, "download", "--help"])
    except Exception:
        return set()
    flags = set()
    for token in help_txt.split():
        if token.startswith("--"):
            # strip punctuation like ',' ')' etc
            flags.add(token.strip(",)"))
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF models/datasets via `hf download`.")
    parser.add_argument("--model", "-M", default=None, type=str)
    parser.add_argument("--dataset", "-D", default=None, type=str)
    parser.add_argument("--token", "-T", default=None, type=str)
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--save_dir", "-S", default=None, type=str)
    parser.add_argument("--use_hf_transfer", default=False, type=_str2bool)
    parser.add_argument("--use_mirror", default=True, type=_str2bool)
    parser.add_argument("--hf_endpoint", default="https://hf-mirror.com", type=str)

    # New: allow forcing redownload (optional)
    parser.add_argument("--force", default=False, type=_str2bool,
                        help="Force re-download if supported by your hf CLI (default: False).")

    args = parser.parse_args()

    if (args.model is None) == (args.dataset is None):
        print("[ERROR] Specify exactly one of --model or --dataset.")
        sys.exit(2)

    token = _pick_token(args.token)

    env = dict(os.environ)
    if args.use_hf_transfer:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if args.use_mirror:
        env["HF_ENDPOINT"] = args.hf_endpoint

    print("[ENV] HF_HUB_ENABLE_HF_TRANSFER =", env.get("HF_HUB_ENABLE_HF_TRANSFER"))
    print("[ENV] HF_ENDPOINT               =", env.get("HF_ENDPOINT"))
    print("[ENV] Token provided            =", "YES" if token else "NO")

    repo_id = args.model if args.model is not None else args.dataset
    repo_type = "model" if args.model is not None else "dataset"

    local_dir: Optional[str] = None
    if args.save_dir:
        parts = repo_id.split("/")
        prefix = "models" if repo_type == "model" else "datasets"
        if len(parts) >= 2:
            local_dir = os.path.join(args.save_dir, f"{prefix}--{parts[0]}--{parts[1]}")
        else:
            local_dir = os.path.join(args.save_dir, f"{prefix}--{parts[0]}")
        os.makedirs(local_dir, exist_ok=True)

    hf_bin = shutil.which("hf")
    if not hf_bin:
        print("[ERROR] `hf` command not found in PATH.")
        sys.exit(127)

    supported = _hf_download_supported_flags(hf_bin)

    cmd = ["hf", "download", repo_id]

    # token flag (usually supported)
    if token and "--token" in supported:
        cmd += ["--token", token]

    # local dir flags
    if local_dir and "--local-dir" in supported:
        cmd += ["--local-dir", local_dir]
        if "--local-dir-use-symlinks" in supported:
            cmd += ["--local-dir-use-symlinks", "False"]

    # dataset flag
    if repo_type == "dataset" and "--repo-type" in supported:
        cmd += ["--repo-type", "dataset"]

    # include/exclude flags (only if supported)
    for pat in (args.include or []):
        if "--include" in supported:
            cmd += ["--include", pat]
    for pat in (args.exclude or []):
        if "--exclude" in supported:
            cmd += ["--exclude", pat]

    # force download if requested and supported
    if args.force and "--force-download" in supported:
        cmd += ["--force-download"]

    # NOTE: do NOT use --resume-download; not supported in your hf CLI.
    # Default behavior already reuses cache and will not needlessly re-download.

    code = _run(cmd, env=env)
    sys.exit(code)


if __name__ == "__main__":
    main()