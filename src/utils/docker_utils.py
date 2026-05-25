"""Docker container lifecycle management for pipeline scripts."""

import subprocess
import time
from pathlib import Path


def _docker_cmd(cmd: str) -> str:
    """Execute docker command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def ensure_containers_running(containers: list[str]) -> None:
    """Start specified containers if not already running."""
    for container in containers:
        status = _docker_cmd(f"docker ps --filter name={container} --format '{{{{.State}}}}'")
        if status != "running":
            print(f"  Starting {container}...")
            _docker_cmd(f"docker compose up -d {container}")
            time.sleep(2)  # Give container time to start
    print()


def stop_containers(containers: list[str]) -> None:
    """Stop specified containers."""
    for container in containers:
        status = _docker_cmd(f"docker ps --filter name={container} --format '{{{{.State}}}}'")
        if status == "running":
            print(f"  Stopping {container}...")
            _docker_cmd(f"docker stop {container}")
    print()


def get_required_containers(variant: str) -> list[str]:
    """Get list of containers needed for this variant."""
    containers = []
    if variant == "vector":
        containers.append("chromadb")
    elif variant == "graph":
        # Graph uses both chromadb for entity embedding and ollama for extraction
        containers.extend(["chromadb", "ollama-agent"])
    # bm25 doesn't need any containers
    return containers
