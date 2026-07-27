import json
import os
import selectors
import sys
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "jaide-status-bench"
LOCAL_PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_MOUNT_PATH = Path("/workspace/jaide")
DATA_MOUNT_PATH = Path("/data")
CHECKPOINT_MOUNT_PATH = Path("/checkpoints")
REPORT_MOUNT_PATH = Path("/reports")
BUILD_MOUNT_PATH = Path("/build_artifacts")

IGNORE_PATTERNS = [
    ".git",
    ".zig-cache",
    "zig-out",
    ".venv",
    ".venv-modal",
    "__pycache__",
    "*.o",
    "*.a",
    "*.bin",
    ".local",
    ".cache",
    ".upm",
    ".pythonlibs",
    ".config",
]

GPU_SPEC = os.environ.get("JAIDE_BENCH_GPU", "B200+:8")
TIMEOUT_SEC = int(os.environ.get("JAIDE_BENCH_TIMEOUT", "86400"))
CPU_TIMEOUT_SEC = int(os.environ.get("JAIDE_CPU_TIMEOUT", "7200"))
MODEL_DIM = int(os.environ.get("JAIDE_BENCH_MODEL_DIM", "5120"))
NUM_LAYERS = int(os.environ.get("JAIDE_BENCH_LAYERS", "54"))
BATCH_SIZE = int(os.environ.get("JAIDE_BENCH_BATCH", "64"))
EPOCHS = int(os.environ.get("JAIDE_BENCH_EPOCHS", "1"))
SAMPLE_CAP = int(os.environ.get("JAIDE_BENCH_SAMPLE_CAP", "2000000"))
MAX_SEQ_LEN = int(os.environ.get("JAIDE_BENCH_MAX_SEQ_LEN", "256"))
LEARNING_RATE = os.environ.get("JAIDE_BENCH_LR", "0.0001")
REASONING_CYCLES = int(os.environ.get("JAIDE_BENCH_REASONING_CYCLES", "1"))
RELATIONAL_PASS_INTERVAL = int(os.environ.get("JAIDE_BENCH_RELATIONAL_PASS_INTERVAL", "50"))
NUM_GPUS = int(os.environ.get("JAIDE_BENCH_NUM_GPUS", "8"))
RECONSTRUCTION_ALPHA = os.environ.get("JAIDE_BENCH_RECONSTRUCTION_ALPHA", "0.3")
PHASE_A_STEPS = int(os.environ.get("JAIDE_BENCH_PHASE_A_STEPS", "500"))
PHASE_B_STEPS = int(os.environ.get("JAIDE_BENCH_PHASE_B_STEPS", "2000"))
SHUFFLE_TARGET_CONTROL = os.environ.get("JAIDE_BENCH_SHUFFLE_TARGET_CONTROL", "0")
TARGET_SOURCE_FROZEN = os.environ.get("JAIDE_BENCH_TARGET_SOURCE_FROZEN", "1")
SPECTRAL_DEPTH_COMPENSATION = os.environ.get("JAIDE_BENCH_SPECTRAL_DEPTH_COMPENSATION", "1")

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("jaide-bench-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("jaide-bench-checkpoints", create_if_missing=True)
report_volume = modal.Volume.from_name("jaide-bench-reports", create_if_missing=True)
build_volume = modal.Volume.from_name("jaide-bench-build", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .entrypoint([])
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages "
        "git curl xz-utils build-essential wget ca-certificates pkg-config "
        "libnccl2 libnccl-dev opencl-headers ocl-icd-opencl-dev jq",
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("pyarrow", "requests", "zstandard", "datasets", "huggingface_hub", "hf_xet")
    .run_commands(
        "mkdir -p /opt",
        "curl -sL https://ziglang.org/download/0.14.1/zig-x86_64-linux-0.14.1.tar.xz "
        "| tar -xJ -C /opt",
        "ln -sf /opt/zig-x86_64-linux-0.14.1/zig /usr/local/bin/zig",
        "zig version",
    )
    .run_commands(
        # Pinned to a specific stable release so the image is reproducible.
        # Update the version string here (and nowhere else) when upgrading.
        "curl -sL https://github.com/diku-dk/futhark/releases/download/v0.26.4/"
        "futhark-0.26.4-linux-x86_64.tar.xz -o /tmp/futhark.tar.xz",
        "mkdir -p /opt/futhark",
        "tar -xJf /tmp/futhark.tar.xz -C /opt/futhark --strip-components=1",
        "ln -sf /opt/futhark/bin/futhark /usr/local/bin/futhark",
        "rm /tmp/futhark.tar.xz",
        # Verify the installed binary matches the pinned version.
        "futhark --version | grep -F '0.26.4' || "
        "{ echo 'ERROR: futhark version mismatch after install'; exit 1; }",
    )
    .env(
        {
            "PATH": "/opt/zig-x86_64-linux-0.14.1:/opt/futhark/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs",
            "HF_HOME": "/data/hf_home",
            "HF_DATASETS_CACHE": "/data/hf_datasets_cache",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_dir(
        str(LOCAL_PROJECT_DIR),
        remote_path=str(PROJECT_MOUNT_PATH),
        ignore=IGNORE_PATTERNS,
    )
)

def _log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)

def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def _run(
    cmd: List[str],
    cwd: str,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    timeout: int = 900,
    input_bytes: Optional[bytes] = None,
) -> Tuple[int, str, float]:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    _log(f">>> {' '.join(cmd)}  (cwd={cwd})")
    t0 = time.monotonic()
    deadline = t0 + timeout
    stdin_mode = subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=stdin_mode,
        bufsize=0,
        start_new_session=True,
    )
    if proc.stdout is None:
        _terminate_process_group(proc)
        raise RuntimeError("subprocess stdout pipe was not created")
    if input_bytes is not None:
        if proc.stdin is None:
            _terminate_process_group(proc)
            raise RuntimeError("subprocess stdin pipe was not created")
        try:
            proc.stdin.write(input_bytes)
            proc.stdin.flush()
        finally:
            proc.stdin.close()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    output_chunks: List[bytes] = []
    timed_out = False
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                _terminate_process_group(proc)
                break
            if proc.poll() is not None:
                remaining = proc.stdout.read()
                if remaining:
                    print(remaining.decode("utf-8", errors="replace"), end="", flush=True)
                    output_chunks.append(remaining)
                break
            events = selector.select(timeout=min(1.0, deadline - now))
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                output_chunks.append(chunk)
        if proc.poll() is None:
            proc.wait()
    finally:
        selector.close()
        proc.stdout.close()
    dt = time.monotonic() - t0
    out = b"".join(output_chunks).decode("utf-8", errors="replace")
    _log(f"<<< exit={proc.returncode}  dt={dt:.2f}s")
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, output=out.encode("utf-8"))
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    return proc.returncode, out, dt

def _run_multirank(
    cmd: List[str],
    cwd: str,
    base_env: Dict[str, str],
    num_gpus: int,
    nccl_id_path: str,
    timeout: int,
) -> Tuple[int, str, float]:
    if num_gpus <= 0:
        raise ValueError("num_gpus must be >= 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    _log(f">>> multirank {' '.join(cmd)} ranks={num_gpus} (cwd={cwd})")
    t0 = time.monotonic()
    deadline = t0 + timeout

    for stale in [nccl_id_path, nccl_id_path + ".ready"]:
        stale_path = Path(stale)
        if stale_path.exists():
            stale_path.unlink()

    rank_envs: List[Dict[str, str]] = []
    for rank_index in range(num_gpus):
        rank_env = base_env.copy()
        rank_env["WORLD_SIZE"] = str(num_gpus)
        rank_env["RANK"] = str(rank_index)
        rank_env["LOCAL_RANK"] = str(rank_index)
        rank_env["JAIDE_NCCL_ID_PATH"] = nccl_id_path
        rank_envs.append(rank_env)

    procs: List[subprocess.Popen] = []
    fd_to_rank: Dict[int, int] = {}
    output_chunks: List[bytes] = []
    selector = selectors.DefaultSelector()
    timed_out = False

    try:
        for rank_index in range(num_gpus):
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=rank_envs[rank_index],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
            )
            if proc.stdout is None:
                procs.append(proc)
                for started in procs:
                    _terminate_process_group(started)
                raise RuntimeError(f"rank {rank_index} stdout pipe was not created")
            procs.append(proc)
            selector.register(proc.stdout, selectors.EVENT_READ)
            fd_to_rank[proc.stdout.fileno()] = rank_index

        open_streams = num_gpus
        while True:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                for proc in procs:
                    _terminate_process_group(proc)
                break

            if open_streams == 0 and all(proc.poll() is not None for proc in procs):
                break

            events = selector.select(timeout=min(1.0, max(0.05, deadline - now)))
            if not events:
                if open_streams == 0 and all(proc.poll() is not None for proc in procs):
                    break
                continue

            for key, _ in events:
                rank_index = fd_to_rank.get(key.fd, -1)
                try:
                    chunk = os.read(key.fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    selector.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                prefix = f"[rank {rank_index}] ".encode()
                for line in chunk.splitlines(keepends=True):
                    output_chunks.append(prefix + line)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

        for proc in procs:
            if proc.poll() is None:
                proc.wait()
    finally:
        selector.close()
        for proc in procs:
            if proc.stdout:
                try:
                    proc.stdout.close()
                except OSError:
                    pass

    dt = time.monotonic() - t0
    combined_out = b"".join(output_chunks).decode("utf-8", errors="replace")
    returncodes = [proc.returncode for proc in procs]
    failures = [code for code in returncodes if code != 0]
    combined_rc = 0 if not failures else max(failures)
    _log(f"<<< multirank ranks={num_gpus} rcs={returncodes} dt={dt:.2f}s")
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, output=combined_out.encode("utf-8"))
    return combined_rc, combined_out, dt


def _write_report(report_dir: Path, name: str, content: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fp = report_dir / name
    with open(fp, "w", encoding="utf-8") as f:
        f.write(content)
    _log(f"report written: {fp}")

def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _download_finephrase(target_path: Path, cap: int) -> Tuple[int, int]:
    from datasets import load_dataset

    if cap <= 0:
        raise ValueError("sample cap must be positive")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        line_count = _count_nonempty_lines(target_path)
        size = target_path.stat().st_size
        if line_count >= cap:
            _log(f"dataset already present: {target_path} ({line_count} lines, {size} bytes)")
            return size, line_count
        _log(f"cached dataset is incomplete: {target_path} ({line_count}/{cap} lines), rebuilding")

    tmp = target_path.with_suffix(".tmp.jsonl")
    if tmp.exists():
        tmp.unlink()

    ds = load_dataset("HuggingFaceFW/finephrase", "faq", split="train", streaming=True)
    written = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f_out:
            for row in ds:
                text = None
                if isinstance(row, dict):
                    for key in ("text", "content", "sentence", "article"):
                        val = row.get(key)
                        if isinstance(val, str) and val.strip():
                            text = val.strip()
                            break
                if text and len(text) > 20:
                    f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= cap:
                        break
            f_out.flush()
            os.fsync(f_out.fileno())
        if written < cap:
            raise RuntimeError(f"dataset ended after {written} usable samples, requested {cap}")
        tmp.replace(target_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    size = target_path.stat().st_size
    _log(f"downloaded {written} samples, {size} bytes -> {target_path}")
    return size, written

def _run_futhark_kernels(project_dir: str, env: Dict[str, str]) -> None:
    accel_dir = os.path.join(project_dir, "src", "hw", "accel")
    _log("Futhark pkg sync")
    _run(
        ["futhark", "pkg", "sync"],
        cwd=accel_dir,
        env=env,
        check=False,
        timeout=180,
    )
    _log("Futhark CPU library build")
    _run(
        [
            "futhark",
            "c",
            "--library",
            os.path.join(accel_dir, "futhark_kernels.fut"),
            "-o",
            os.path.join(accel_dir, "futhark_kernels"),
        ],
        cwd=project_dir,
        env=env,
    )
    _log("Futhark CUDA library build")
    _run(
        [
            "futhark",
            "cuda",
            "--library",
            os.path.join(accel_dir, "main.fut"),
            "-o",
            os.path.join(accel_dir, "main_gpu"),
        ],
        cwd=project_dir,
        env=env,
    )

@app.function(
    image=image,
    cpu=(2.0, 8.0),
    memory=(4096, 16384),
    timeout=CPU_TIMEOUT_SEC,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(REPORT_MOUNT_PATH): report_volume,
        str(BUILD_MOUNT_PATH): build_volume,
    },
)
def prepare_cpu(run_id: int) -> Dict[str, Any]:
    project_dir = str(PROJECT_MOUNT_PATH)
    env = os.environ.copy()

    report_dir = REPORT_MOUNT_PATH / f"run_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "report_dir": str(report_dir),
        "phases": {},
    }

    _log("=" * 70)
    _log(f"CPU PREPARE PHASE run_id={run_id}")
    _log("=" * 70)

    _run(["zig", "version"], cwd=project_dir, env=env)
    _run(["futhark", "--version"], cwd=project_dir, env=env)

    _run_futhark_kernels(project_dir, env)

    # Clear any stale Zig compilation cache left by a previous run in the same
    # container instance.  The @cImport result for futhark_kernels.h is cached
    # under .zig-cache; if that cache pre-dates the futhark_c step the build
    # fails with a spurious "file not found" even though the header now exists.
    zig_cache = Path(project_dir) / ".zig-cache"
    if zig_cache.exists():
        shutil.rmtree(str(zig_cache))
        _log("Cleared stale .zig-cache before build")

    _log("=" * 70)
    _log("PHASE B: GPU-target build (-Dgpu=true)")
    _log("=" * 70)
    t0 = time.time()
    rc_b, out_b, _ = _run(
        [
            "zig",
            "build",
            "-Dgpu=true",
            "-Doptimize=ReleaseSafe",
        ],
        cwd=project_dir,
        env=env,
        check=False,
        timeout=1800,
    )
    result["phases"]["B_gpu_build"] = {
        "returncode": rc_b,
        "duration_s": round(time.time() - t0, 2),
    }
    _write_report(report_dir, "phase_b_gpu_build.log", out_b)

    inference_bin = Path(project_dir) / "zig-out" / "bin" / "jaide-inference-server"
    distributed_bin = Path(project_dir) / "zig-out" / "bin" / "jaide-distributed-futhark"

    build_target_dir = BUILD_MOUNT_PATH / f"run_{run_id}"
    build_target_dir.mkdir(parents=True, exist_ok=True)

    if distributed_bin.exists():
        shutil.copy2(str(distributed_bin), str(build_target_dir / "jaide-distributed-futhark"))
        os.chmod(str(build_target_dir / "jaide-distributed-futhark"), 0o755)
        result["distributed_binary_present"] = True
    else:
        result["distributed_binary_present"] = False
        _log(f"WARN: distributed binary NOT built at {distributed_bin}")

    if inference_bin.exists():
        shutil.copy2(str(inference_bin), str(build_target_dir / "jaide-inference-server"))
        os.chmod(str(build_target_dir / "jaide-inference-server"), 0o755)
        result["inference_binary_present"] = True
    else:
        result["inference_binary_present"] = False
        _log(f"WARN: inference binary NOT built at {inference_bin}")

    _log("=" * 70)
    _log(f"PHASE C-prep: dataset download ({SAMPLE_CAP} samples)")
    _log("=" * 70)
    dataset_dir = DATA_MOUNT_PATH / "dataset"
    dataset_path = dataset_dir / "finephrase_bench.jsonl"
    try:
        t0 = time.time()
        size, sample_count = _download_finephrase(dataset_path, SAMPLE_CAP)
        result["phases"]["C_prep_dataset"] = {
            "duration_s": round(time.time() - t0, 2),
            "sample_count": sample_count,
            "dataset_bytes": size,
            "dataset_path": str(dataset_path),
        }
    except Exception as exc:
        _log(f"dataset download failed: {exc}")
        result["phases"]["C_prep_dataset"] = {"error": str(exc)}

    data_volume.commit()
    build_volume.commit()
    report_volume.commit()

    _log("=" * 70)
    _log("CPU PREPARE PHASE DONE")
    _log("=" * 70)

    return result

@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=(2.0, 8.0),
    memory=(65536, 196608),
    timeout=TIMEOUT_SEC,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
        str(REPORT_MOUNT_PATH): report_volume,
        str(BUILD_MOUNT_PATH): build_volume,
    },
)
def run_gpu_train_and_infer(
    run_id: int,
    prep_result: Dict[str, Any],
) -> Dict[str, Any]:
    project_dir = str(PROJECT_MOUNT_PATH)
    env = os.environ.copy()

    build_volume.reload()
    data_volume.reload()

    report_dir = REPORT_MOUNT_PATH / f"run_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "gpu_spec": GPU_SPEC,
        "model_dim": MODEL_DIM,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "sample_cap": SAMPLE_CAP,
        "max_seq_len": MAX_SEQ_LEN,
        "learning_rate": LEARNING_RATE,
        "reasoning_cycles": REASONING_CYCLES,
        "relational_pass_interval": RELATIONAL_PASS_INTERVAL,
        "phases": {},
    }

    _log("=" * 70)
    _log(f"GPU PHASE START gpu={GPU_SPEC} run_id={run_id}")
    _log("=" * 70)
    gpu_phase_start = time.time()

    _run(["nvidia-smi"], cwd=project_dir, env=env, check=False, timeout=30)
    _run(["lscpu"], cwd=project_dir, env=env, check=False, timeout=10)

    build_source_dir = BUILD_MOUNT_PATH / f"run_{run_id}"
    distributed_bin_src = build_source_dir / "jaide-distributed-futhark"
    inference_bin_src = build_source_dir / "jaide-inference-server"

    distributed_bin = Path("/tmp/jaide-distributed-futhark")
    inference_bin = Path("/tmp/jaide-inference-server")

    if distributed_bin_src.exists():
        shutil.copy2(str(distributed_bin_src), str(distributed_bin))
        os.chmod(str(distributed_bin), 0o755)
        _log(f"distributed binary staged: {distributed_bin}")
    else:
        _log(f"ERROR: distributed binary missing from {distributed_bin_src}")

    if inference_bin_src.exists():
        shutil.copy2(str(inference_bin_src), str(inference_bin))
        os.chmod(str(inference_bin), 0o755)
        _log(f"inference binary staged: {inference_bin}")
    else:
        _log(f"WARN: inference binary missing from {inference_bin_src}")

    dataset_meta = prep_result.get("phases", {}).get("C_prep_dataset", {})
    dataset_path = dataset_meta.get("dataset_path")
    sample_count = dataset_meta.get("sample_count", 0)

    if not distributed_bin.exists():
        result["phases"]["C_training_convergence"] = {"skipped": "distributed binary missing"}
    elif not dataset_path or sample_count <= 0:
        result["phases"]["C_training_convergence"] = {"skipped": "dataset not prepared"}
    else:
        _log("=" * 70)
        _log(f"PHASE C: TRAINING ({sample_count} samples, {EPOCHS} epochs, dim={MODEL_DIM})")
        _log("=" * 70)

        train_env = env.copy()
        train_env["WORLD_SIZE"] = str(NUM_GPUS)
        train_env["MASTER_ADDR"] = "127.0.0.1"
        train_env["MASTER_PORT"] = "29500"
        train_env["JAIDE_EPOCHS"] = str(EPOCHS)
        train_env["JAIDE_DATASET"] = str(dataset_path)
        train_env["JAIDE_MODEL_DIM"] = str(MODEL_DIM)
        train_env["JAIDE_LAYERS"] = str(NUM_LAYERS)
        train_env["JAIDE_BATCH_SIZE"] = str(BATCH_SIZE)
        train_env["JAIDE_NCCL_ID_PATH"] = "/tmp/jaide_nccl_id"
        train_env["JAIDE_TOTAL_SAMPLES"] = str(sample_count)
        train_env["JAIDE_MAX_SAMPLES"] = str(min(sample_count, SAMPLE_CAP))
        train_env["JAIDE_MAX_SEQ_LEN"] = str(MAX_SEQ_LEN)
        train_env["JAIDE_LEARNING_RATE"] = LEARNING_RATE
        train_env["JAIDE_REASONING_CYCLES"] = str(REASONING_CYCLES)
        train_env["JAIDE_RELATIONAL_PASS_INTERVAL"] = str(RELATIONAL_PASS_INTERVAL)
        train_env["JAIDE_RECONSTRUCTION_ALPHA"] = RECONSTRUCTION_ALPHA
        train_env["JAIDE_PHASE_A_STEPS"] = str(PHASE_A_STEPS)
        train_env["JAIDE_PHASE_B_STEPS"] = str(PHASE_B_STEPS)
        train_env["JAIDE_SHUFFLE_TARGET_CONTROL"] = SHUFFLE_TARGET_CONTROL
        train_env["JAIDE_TARGET_SOURCE_FROZEN"] = TARGET_SOURCE_FROZEN
        train_env["JAIDE_SPECTRAL_DEPTH_COMPENSATION"] = SPECTRAL_DEPTH_COMPENSATION
        vocab_file = Path("/checkpoints/tokenizer.vocab")
        if vocab_file.is_file() and vocab_file.stat().st_size > 0:
            train_env["JAIDE_VOCAB_READY"] = "1"
            _log(f"existing vocab found at {vocab_file} ({vocab_file.stat().st_size} bytes), skipping BPE training (JAIDE_VOCAB_READY=1)")
        else:
            train_env.pop("JAIDE_VOCAB_READY", None)
            _log(f"no valid vocab at {vocab_file}, BPE training will run on rank 0")
        train_env["NCCL_DEBUG"] = "WARN"
        train_env["NCCL_IB_DISABLE"] = "1"
        train_env["NCCL_SOCKET_IFNAME"] = "lo"
        train_env["NCCL_P2P_DISABLE"] = "0"
        train_env["NCCL_SHM_DISABLE"] = "0"
        train_env["NCCL_NVLS_ENABLE"] = "0"
        train_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # Persist Futhark NVRTC-compiled kernels across container runs.
        # First run writes the cache; subsequent runs skip recompilation entirely.
        # Delete stale cache whenever the GPU kernel source (main.fut) may have
        # changed so the runtime recompiles from the newly-built PTX.
        futhark_cache_path = Path("/checkpoints/futhark_gpu_cache.bin")
        train_env["JAIDE_FUTHARK_CACHE"] = str(futhark_cache_path)

        for stale in ["/tmp/jaide_nccl_id", "/tmp/jaide_nccl_id.ready"]:
            p = Path(stale)
            if p.exists():
                p.unlink()

        t0 = time.time()
        rc_c, out_c, _ = _run_multirank(
            cmd=[str(distributed_bin)],
            cwd=project_dir,
            base_env=train_env,
            num_gpus=NUM_GPUS,
            nccl_id_path="/tmp/jaide_nccl_id",
            timeout=72000,
        )
        phase_c_duration = time.time() - t0

        loss_curve: List[Tuple[int, float]] = []
        recon_curve: List[Tuple[int, float]] = []
        source_rms_curve: List[Tuple[int, float]] = []
        epoch_metrics: List[Dict[str, Any]] = []
        for line in out_c.splitlines():
            if "[Step " in line and "Loss:" in line:
                try:
                    s_part = line.split("[Step ")[1].split("]")[0].strip()
                    step_index = int(s_part)
                    l_part = line.split("Loss:")[1].strip().split()[0]
                    loss_curve.append((step_index, float(l_part)))
                except (ValueError, IndexError):
                    continue
                if "Recon:" in line:
                    try:
                        r_part = line.split("Recon:")[1].strip().split()[0]
                        recon_curve.append((step_index, float(r_part)))
                    except (ValueError, IndexError):
                        pass
                if "SourceRMS:" in line:
                    try:
                        rms_part = line.split("SourceRMS:")[1].strip().split()[0]
                        source_rms_curve.append((step_index, float(rms_part)))
                    except (ValueError, IndexError):
                        pass
            if line.startswith("[Epoch "):
                try:
                    after_bracket = line.split("]", 1)[1]
                    loss_str = after_bracket.split("Loss:")[1].split("|")[0].strip()
                    time_str = after_bracket.split("Time:")[1].strip().rstrip("s")
                    epoch_metrics.append({
                        "loss": float(loss_str),
                        "time_s": float(time_str),
                    })
                except (ValueError, IndexError):
                    pass

        metrics_path = Path("/checkpoints/training_metrics.json")
        training_metrics_json: Optional[Dict[str, Any]] = None
        if metrics_path.exists():
            try:
                training_metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
                _write_report(report_dir, "training_metrics.json", metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        result["phases"]["C_training_convergence"] = {
            "returncode": rc_c,
            "duration_s": round(phase_c_duration, 2),
            "sample_count": sample_count,
            "loss_curve_length": len(loss_curve),
            "first_loss": loss_curve[0][1] if loss_curve else None,
            "last_loss": loss_curve[-1][1] if loss_curve else None,
            "recon_curve_length": len(recon_curve),
            "first_recon": recon_curve[0][1] if recon_curve else None,
            "last_recon": recon_curve[-1][1] if recon_curve else None,
            "recon_converged": (
                len(recon_curve) >= 2 and recon_curve[-1][1] < recon_curve[0][1]
            ) if recon_curve else False,
            "first_source_rms": source_rms_curve[0][1] if source_rms_curve else None,
            "last_source_rms": source_rms_curve[-1][1] if source_rms_curve else None,
            "source_collapse_suspected": (
                len(source_rms_curve) >= 2
                and source_rms_curve[0][1] > 0.0
                and source_rms_curve[-1][1] < source_rms_curve[0][1] * 0.5
            ) if source_rms_curve else False,
            "num_gpus": NUM_GPUS,
            "reconstruction_alpha": RECONSTRUCTION_ALPHA,
            "phase_a_steps": PHASE_A_STEPS,
            "phase_b_steps": PHASE_B_STEPS,
            "shuffle_target_control": SHUFFLE_TARGET_CONTROL,
            "effective_batch_size": BATCH_SIZE * NUM_GPUS,
            "epoch_metrics": epoch_metrics,
            "training_metrics_json": training_metrics_json,
            "converged": (
                len(loss_curve) >= 2 and loss_curve[-1][1] < loss_curve[0][1]
            ) if loss_curve else False,
        }
        _write_report(report_dir, "phase_c_training.log", out_c)
        _write_report(
            report_dir,
            "phase_c_loss_curve.jsonl",
            "\n".join(json.dumps({"step": s, "loss": l}) for s, l in loss_curve),
        )
        _write_report(
            report_dir,
            "phase_c_recon_curve.jsonl",
            "\n".join(json.dumps({"step": s, "recon": r}) for s, r in recon_curve),
        )
        _write_report(
            report_dir,
            "phase_c_source_rms_curve.jsonl",
            "\n".join(json.dumps({"step": s, "source_rms": v}) for s, v in source_rms_curve),
        )
        checkpoint_volume.commit()

    if not inference_bin.exists():
        result["phases"]["D_inference"] = {"skipped": "inference binary missing"}
    else:
        _log("=" * 70)
        _log("PHASE D: INFERENCE SERVER SMOKE TEST")
        _log("=" * 70)
        checkpoints_root = Path("/checkpoints")
        model_candidates = sorted(checkpoints_root.rglob("model.ckpt"), reverse=True)
        model_path = str(model_candidates[0]) if model_candidates else None
        _log(f"model_path candidate: {model_path}")

        inf_env = env.copy()
        if model_path:
            inf_env["JAIDE_MODEL_PATH"] = model_path
        inf_env.setdefault("NCCL_DEBUG", "WARN")

        srv_log_path = report_dir / "phase_d_server.log"
        srv_f = open(srv_log_path, "w")
        srv_proc = subprocess.Popen(
            [str(inference_bin), "--port", "8080", "--host", "127.0.0.1", "--allow-anonymous"],
            cwd=project_dir,
            env=inf_env,
            stdout=srv_f,
            stderr=subprocess.STDOUT,
        )

        try:
            server_up = False
            health_json = ""
            for _ in range(40):
                time.sleep(0.5)
                rc_h, out_h, _ = _run(
                    [
                        "curl",
                        "-sS",
                        "-o",
                        "/tmp/health.json",
                        "-w",
                        "%{http_code}",
                        "http://127.0.0.1:8080/v1/health",
                    ],
                    cwd=project_dir,
                    env=inf_env,
                    check=False,
                    timeout=10,
                )
                if "200" in out_h:
                    server_up = True
                    if Path("/tmp/health.json").exists():
                        health_json = Path("/tmp/health.json").read_text(
                            encoding="utf-8", errors="replace"
                        )
                    break

            if not server_up:
                result["phases"]["D_inference"] = {
                    "error": "health endpoint never responded 200",
                    "server_up": False,
                }
            else:
                _log(f"health OK: {health_json}")

                prompt = "The reversible sparse flow model demonstrates"
                req_body = json.dumps({"text": prompt, "max_tokens": 20})
                t0 = time.time()
                rc_i, out_i, _ = _run(
                    [
                        "curl",
                        "-sS",
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        req_body,
                        "http://127.0.0.1:8080/v1/inference",
                    ],
                    cwd=project_dir,
                    env=inf_env,
                    check=False,
                    timeout=60,
                )
                inference_duration = time.time() - t0

                parsed = None
                try:
                    parsed = json.loads(out_i)
                except Exception:
                    pass

                generated_tokens: List[int] = []
                generated_text_value = ""
                if isinstance(parsed, dict):
                    raw_tokens = parsed.get("tokens")
                    if isinstance(raw_tokens, list):
                        generated_tokens = [t for t in raw_tokens if isinstance(t, int)]
                    raw_text = parsed.get("text")
                    if isinstance(raw_text, str):
                        generated_text_value = raw_text

                distinct_tokens = len(set(generated_tokens))
                non_reserved = [t for t in generated_tokens if t >= 4]

                result["phases"]["D_inference"] = {
                    "returncode": rc_i,
                    "duration_s": round(inference_duration, 2),
                    "health": health_json,
                    "prompt": prompt,
                    "response_body": out_i,
                    "response_parsed": parsed,
                    "server_up": True,
                    "model_path": model_path,
                    "generated_token_count": len(generated_tokens),
                    "generated_distinct_tokens": distinct_tokens,
                    "generated_non_reserved_count": len(non_reserved),
                    "generated_text_length": len(generated_text_value),
                    "generation_produced_output": len(generated_tokens) > 0,
                    "generation_is_degenerate": (
                        len(generated_tokens) > 1 and distinct_tokens <= 1
                    ),
                }
                _write_report(report_dir, "phase_d_inference.log", out_i)
        finally:
            srv_proc.terminate()
            try:
                srv_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                srv_proc.kill()
            srv_f.close()

    gpu_phase_duration = time.time() - gpu_phase_start
    result["gpu_phase_duration_s"] = round(gpu_phase_duration, 2)
    _log("=" * 70)
    _log(f"GPU PHASE END duration={gpu_phase_duration:.2f}s")
    _log("=" * 70)

    summary_json = json.dumps(result, indent=2, default=str)
    _write_report(report_dir, "gpu_phase_summary.json", summary_json)
    report_volume.commit()

    return result

@app.local_entrypoint()
def main():
    if MODEL_DIM <= 0 or MODEL_DIM % 2 != 0:
        raise ValueError("JAIDE_BENCH_MODEL_DIM must be a positive even integer")
    if NUM_LAYERS <= 0:
        raise ValueError("JAIDE_BENCH_LAYERS must be positive")
    if BATCH_SIZE <= 0:
        raise ValueError("JAIDE_BENCH_BATCH must be positive")
    if EPOCHS <= 0:
        raise ValueError("JAIDE_BENCH_EPOCHS must be positive")
    if SAMPLE_CAP <= 0:
        raise ValueError("JAIDE_BENCH_SAMPLE_CAP must be positive")
    if MAX_SEQ_LEN <= 0:
        raise ValueError("JAIDE_BENCH_MAX_SEQ_LEN must be positive")
    if REASONING_CYCLES <= 0:
        raise ValueError("JAIDE_BENCH_REASONING_CYCLES must be positive")
    if RELATIONAL_PASS_INTERVAL <= 0:
        raise ValueError("JAIDE_BENCH_RELATIONAL_PASS_INTERVAL must be positive")
    if NUM_GPUS <= 0:
        raise ValueError("JAIDE_BENCH_NUM_GPUS must be a positive integer")
    try:
        reconstruction_alpha_value = float(RECONSTRUCTION_ALPHA)
    except ValueError:
        raise ValueError("JAIDE_BENCH_RECONSTRUCTION_ALPHA must be a float in [0.0, 1.0]")
    if not 0.0 <= reconstruction_alpha_value <= 1.0:
        raise ValueError("JAIDE_BENCH_RECONSTRUCTION_ALPHA must be a float in [0.0, 1.0]")
    if PHASE_A_STEPS < 0:
        raise ValueError("JAIDE_BENCH_PHASE_A_STEPS must be >= 0")
    if PHASE_B_STEPS < 0:
        raise ValueError("JAIDE_BENCH_PHASE_B_STEPS must be >= 0")
    if SHUFFLE_TARGET_CONTROL not in ("0", "1", "true", "false"):
        raise ValueError("JAIDE_BENCH_SHUFFLE_TARGET_CONTROL must be 0, 1, true or false")
    if TARGET_SOURCE_FROZEN not in ("0", "1", "true", "false"):
        raise ValueError("JAIDE_BENCH_TARGET_SOURCE_FROZEN must be 0, 1, true or false")
    if SPECTRAL_DEPTH_COMPENSATION not in ("0", "1", "true", "false"):
        raise ValueError("JAIDE_BENCH_SPECTRAL_DEPTH_COMPENSATION must be 0, 1, true or false")
    learning_rate_value = float(LEARNING_RATE)
    if not learning_rate_value > 0.0 or not learning_rate_value < float("inf"):
        raise ValueError("JAIDE_BENCH_LR must be finite and positive")
    run_id = int(time.time())
    _log(f"launching run_id={run_id}")

    _log("STEP 1: prepare_cpu")
    prep_result = prepare_cpu.remote(run_id)
    print("\n" + "=" * 70)
    print("CPU PREPARE RESULT")
    print("=" * 70)
    print(json.dumps(prep_result, indent=2, default=str))

    if not prep_result.get("distributed_binary_present"):
        print("\n" + "=" * 70)
        print("ABORT: distributed binary was not built")
        print("=" * 70)
        return

    dataset_ok = prep_result.get("phases", {}).get("C_prep_dataset", {}).get("sample_count", 0) > 0
    if not dataset_ok:
        print("\n" + "=" * 70)
        print("ABORT: dataset not prepared")
        print("=" * 70)
        return

    _log("STEP 2: run_gpu_train_and_infer")
    gpu_result = run_gpu_train_and_infer.remote(run_id, prep_result)
    print("\n" + "=" * 70)
    print("GPU PHASE RESULT")
    print("=" * 70)
    print(json.dumps(gpu_result, indent=2, default=str))

    final = {
        "run_id": run_id,
        "cpu_phase": prep_result,
        "gpu_phase": gpu_result,
    }
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(json.dumps(final, indent=2, default=str))