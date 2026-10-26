"""Bootstrap and supervise the isolated Laya feasibility probe.

Run: python -X utf8 scripts/run_probe.py
"""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INSTALL_TIMEOUT = 1200  # Seconds per installation command.
WORKER_TIMEOUT = 1800  # Includes model download, load, train and save.
MEMORY_LIMIT = 12 * 1024**3  # Best-effort process-tree RSS limit.
MIN_FREE_RAM = 3 * 1024**3  # Preserve system headroom.
# =======================================================


def status(state, **details):
    payload = {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "supervisor_pid": os.getpid(), **details}
    tmp = RUN / "status.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(RUN / "status.json")
    logging.info("status=%s details=%s", state, details)


def command(args):
    logging.info("command=%s", args)
    with (RUN / "setup.log").open("a", encoding="utf-8") as stream:
        subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=INSTALL_TIMEOUT)


def main():
    global RUN, WORKER_TIMEOUT, MEMORY_LIMIT
    parser = argparse.ArgumentParser(description="Bounded CPU training probe, with live logs.")
    parser.add_argument("--full", action="store_true", help="Unfreeze encoder as well as heads.")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Show worker output in this terminal.")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    RUN = ROOT / "runs" / (("full_" if args.full else "head_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if args.full:
        WORKER_TIMEOUT = 1200
        MEMORY_LIMIT = 16 * 1024**3
    if args.dataset:
        WORKER_TIMEOUT = 7200
    RUN.mkdir(parents=True, exist_ok=False)
    handlers = [logging.FileHandler(RUN / "supervisor.log", encoding="utf-8")]
    if args.watch:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(handlers=handlers, level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("run_directory=%s", RUN)
    worker = None
    try:
        if not args.skip_install:
            status("INSTALLING")
            command([str(PYTHON), "-m", "pip", "install", "torch==2.8.0",
                     "--index-url", "https://download.pytorch.org/whl/cpu"])
            command([str(PYTHON), "-m", "pip", "install", "transformers==4.57.6",
                     "huggingface_hub==0.36.0", "numpy==2.2.6", "psutil==7.1.0",
                     str(ROOT / "vendor" / "laya")])
        with (RUN / "requirements-resolved.txt").open("w", encoding="utf-8") as stream:
            subprocess.run([str(PYTHON), "-m", "pip", "freeze"], stdout=stream,
                           check=True, timeout=60)
        # Supervisor runs in this same isolated environment.
        import psutil
        env = os.environ.copy()
        env.update({"HF_HOME": str(ROOT / "cache"), "HF_HUB_DISABLE_TELEMETRY": "1",
                    "HF_HUB_DISABLE_XET": "1", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONUNBUFFERED": "1"})
        with (RUN / "worker.log").open("w", encoding="utf-8") as stream:
            worker_args = [str(PYTHON), "-X", "utf8", str(ROOT / "scripts" / "train_probe.py"),
                           "--run-dir", str(RUN)]
            if args.full:
                worker_args.append("--full")
            if args.dataset:
                worker_args.extend(["--dataset", str(args.dataset.resolve())])
            worker = subprocess.Popen(
                worker_args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            start, peak = time.monotonic(), 0
            log_position = 0

            def show_new_logs():
                nonlocal log_position
                if args.watch:
                    with (RUN / "worker.log").open(encoding="utf-8", errors="replace") as source:
                        source.seek(log_position)
                        text = source.read()
                        log_position = source.tell()
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()

            status("RUNNING", worker_pid=worker.pid)
            process = psutil.Process(worker.pid)
            while worker.poll() is None:
                show_new_logs()
                rss = 0
                for child in [process] + process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                elapsed = time.monotonic() - start
                if (RUN / "STOP").exists() or elapsed > WORKER_TIMEOUT or rss > MEMORY_LIMIT or psutil.virtual_memory().available < MIN_FREE_RAM:
                    for child in process.children(recursive=True):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    worker.kill()
                    worker.wait(timeout=30)
                    status("STOPPED_LIMIT", elapsed_seconds=elapsed, peak_rss_bytes=peak,
                           reason="stop_file_or_time_or_memory_limit")
                    return 2
                status("RUNNING", worker_pid=worker.pid,
                       elapsed_seconds=round(elapsed, 1), peak_rss_bytes=peak)
                time.sleep(5)
            show_new_logs()
            ok = worker.returncode == 0 and (RUN / "result.json").exists()
            status("COMPLETED" if ok else "FAILED", exit_code=worker.returncode,
                   elapsed_seconds=round(time.monotonic() - start, 1), peak_rss_bytes=peak)
            return 0 if ok else 1
    except KeyboardInterrupt:
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.wait(timeout=30)
        status("CANCELLED", reason="User pressed Ctrl+C")
        return 130
    except Exception:
        logging.exception("probe_failed")
        if worker is not None and worker.poll() is None:
            worker.kill()
        status("FAILED", reason="See supervisor.log and setup.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())
