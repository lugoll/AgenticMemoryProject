"""Docker container lifecycle management for pipeline scripts.

Set USE_NATIVE_OLLAMA=true to skip starting/stopping the ollama-agent and
ollama-judge containers (e.g. when running natively on Mac via native Ollama).
ChromaDB is still managed via Docker unless you run it separately too.
"""

import os
import subprocess
import time
from pathlib import Path


def _native_ollama() -> bool:
    return os.environ.get("USE_NATIVE_OLLAMA", "").lower() in ("1", "true", "yes")


def _docker_cmd(cmd: str) -> str:
    """Execute docker command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def ensure_containers_running(containers: list[str]) -> None:
    """Start specified containers if not already running."""
    if _native_ollama():
        containers = [c for c in containers if not c.startswith("ollama")]
    for container in containers:
        status = _docker_cmd(f"docker ps --filter name={container} --format '{{{{.State}}}}'")
        if status != "running":
            print(f"  Starting {container}...")
            _docker_cmd(f"docker compose up -d {container}")
            time.sleep(2)  # Give container time to start
    print()


def stop_containers(containers: list[str]) -> None:
    """Stop specified containers."""
    if _native_ollama():
        containers = [c for c in containers if not c.startswith("ollama")]
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
    elif variant == "msgraphrag":
        # MS GraphRAG uses LanceDB locally (no chromadb) but needs the chat LLM.
        containers.append("ollama-agent")
    # bm25 doesn't need any containers
    return containers
