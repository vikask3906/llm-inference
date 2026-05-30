#!/usr/bin/env python3
"""One-command demo: bring up the full stack, wait for health, drive traffic.

Spins up the gateway + 3 mock backends + Prometheus + Grafana (docker compose),
waits for the gateway to report healthy, then fires sustained prefix-heavy
traffic so the pre-provisioned Grafana dashboard fills with live data. The stack
is left running afterwards; tear it down with `--down` (or `make down`).

Cross-platform on purpose -- no `make` required, runs the same on Windows.

Usage:
  python scripts/demo.py                 # up + 60s of traffic, leave stack running
  python scripts/demo.py --traffic 0     # up only, no traffic
  python scripts/demo.py --down          # tear the stack down (incl. volumes)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request

GATEWAY = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
GRAFANA = "http://localhost:3000"


def _compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True)


def _healthy(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _wait_healthy(url: str, total: float = 120.0, every: float = 2.0) -> bool:
    deadline = time.time() + total
    while time.time() < deadline:
        if _healthy(url):
            return True
        time.sleep(every)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traffic", type=int, default=60,
                    help="seconds of traffic to drive after startup (0 = none)")
    ap.add_argument("--strategy", default="prefix_tree")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--down", action="store_true",
                    help="tear the stack down (incl. volumes) and exit")
    ap.add_argument("--no-build", action="store_true",
                    help="skip the image build (reuse existing images)")
    args = ap.parse_args()

    if args.down:
        print(">> docker compose down -v")
        _compose("down", "-v")
        return

    up = ["up", "-d"] + ([] if args.no_build else ["--build"])
    print(">> docker compose", *up)
    _compose(*up)

    print(f">> waiting for gateway health at {GATEWAY}/healthz ...")
    if not _wait_healthy(f"{GATEWAY}/healthz"):
        print("!! gateway did not become healthy in time; "
              "check `docker compose logs gateway`", file=sys.stderr)
        sys.exit(1)
    print("   gateway is healthy.")

    if args.traffic > 0:
        print(f">> driving {args.traffic}s of '{args.strategy}' traffic "
              f"(concurrency={args.concurrency}) ...")
        subprocess.run([sys.executable, "bench/loadtest.py",
                        "--url", GATEWAY, "--strategy", args.strategy,
                        "--concurrency", str(args.concurrency),
                        "--duration", str(args.traffic)], check=False)

    print()
    print("Stack is up:")
    print(f"  gateway     -> {GATEWAY}")
    print(f"  Prometheus  -> {PROMETHEUS}")
    print(f"  Grafana     -> {GRAFANA}   (anonymous admin; dashboard pre-loaded)")
    print()
    print("Drive more traffic:  python bench/loadtest.py --duration 60   (or `make traffic`)")
    print("Tear down:           python scripts/demo.py --down            (or `make down`)")


if __name__ == "__main__":
    main()
