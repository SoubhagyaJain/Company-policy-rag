"""Start the local real-model validation app without Docker or user-library changes.

Uses the saved validation run, fixed loopback ports, explicit executable arguments,
hidden Windows processes and persistent logs. Never kills an existing server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run = (args.run_dir or Path((ROOT / "logs/live_rag_latest.txt").read_text().strip())).resolve()
    if not run.is_relative_to((ROOT / "logs").resolve()):
        raise ValueError("Validation run must be inside this repository's logs directory.")
    state_file = run / "runtime_state.json"
    state = json.loads(state_file.read_text(encoding="utf-8-sig"))
    for port in (8000, 3000):
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already occupied; no server was stopped.")
    python = ROOT / ".venv/Scripts/python.exe"
    node = shutil.which("node")
    if not python.is_file() or node is None:
        raise RuntimeError("The existing Python virtualenv and Node installation are required.")
    env = os.environ.copy()
    runtime = run / "runtime_storage"
    env.update(APP_STORAGE_DIR=str(run / "library"), STORAGE_DIR=str(runtime),
               IMAGES_STORAGE_DIR=str(runtime / "images"), PDF_IMAGES_DIR=str(runtime / "images"),
               VISION_CACHE_DIR=str(runtime / "vision_cache"), DOCSTORE_DIR=str(runtime / "docstore"),
               BM25_STORAGE_DIR=str(runtime / "bm25"), CHROMA_PERSIST_DIR=str(runtime / "chroma"),
               LOGS_DIR=str(run / "runtime_logs"), DOCUMENT_LIBRARY_MODE="persistent", PYTHONUNBUFFERED="1")
    processes = []
    for name, command, cwd in (
        ("backend", [str(python), "-m", "uvicorn", "backend.api.main:app", "--host", "127.0.0.1", "--port", "8000"], ROOT),
        ("frontend", [node, "node_modules/next/dist/bin/next", "dev", "--hostname", "127.0.0.1", "--port", "3000"], ROOT / "frontend"),
    ):
        with (run / f"{name}.stdout.log").open("ab") as out, (run / f"{name}.stderr.log").open("ab") as err:
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            try:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                           stdout=out, stderr=err, close_fds=True,
                                           creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB)
            except OSError as exc:
                if exc.winerror != 5:
                    raise
                process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                           stdout=out, stderr=err, close_fds=True, creationflags=flags)
        processes.append(process)
        state[f"{name}_pid"] = process.pid
    state["active_library"] = str(run / "library")
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"backend_pid": processes[0].pid, "frontend_pid": processes[1].pid,
                      "url": "http://127.0.0.1:3000", "active_library": state["active_library"]}, indent=2))


if __name__ == "__main__":
    main()
