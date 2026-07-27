import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import modal
APP_NAME = "jaide-v40-distributed-training"
GPU_SPEC = "B200+:8"
DATA_VOLUME_NAME = "jaide-training-data"
CHECKPOINT_VOLUME_NAME = "jaide-checkpoints"
DATA_MOUNT_PATH = Path("/data")
CHECKPOINT_MOUNT_PATH = Path("/checkpoints")
PROJECT_MOUNT_PATH = Path("/jaide")
DATASET_DIR = DATA_MOUNT_PATH / "dataset"
DATASET_FILE = DATASET_DIR / "train.jsonl"
DATASET_METADATA_FILE = DATASET_DIR / "metadata.json"
BINARY_PATH = PROJECT_MOUNT_PATH / "zig-out" / "bin" / "jaide-distributed-futhark"
BINARY_CACHE_PATH = CHECKPOINT_MOUNT_PATH / "jaide-distributed-futhark"
BUILD_VERSION_FILE = CHECKPOINT_MOUNT_PATH / "build_version.txt"
VOCAB_PATH = CHECKPOINT_MOUNT_PATH / "tokenizer.vocab"
CPU_REQUEST = 64.0
CPU_LIMIT = 80.0
MEMORY_REQUEST_MB = 262144
MEMORY_LIMIT_MB = 262144
EPHEMERAL_DISK_MB = 3145728
TIMEOUT_SECONDS = 86400
LOCAL_PROJECT_DIR = (Path(__file__).resolve().parent / "../..").resolve()
IGNORE_PATTERNS = [
    "node_modules",
    ".git",
    "zig-cache",
    ".pythonlibs",
    ".cache",
    ".upm",
    "__pycache__",
    ".local",
    ".replit",
    "*.bin",
]
app = modal.App(APP_NAME)
jaide_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages git curl xz-utils build-essential wget ca-certificates libnccl2 libnccl-dev",
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("pyarrow", "requests", "zstandard", "datasets", "huggingface_hub", "hf_xet")
    .run_commands(
        "mkdir -p /opt",
        "curl -sL https://ziglang.org/download/0.14.1/zig-x86_64-linux-0.14.1.tar.xz | tar -xJ -C /opt",
        "ln -sf /opt/zig-x86_64-linux-0.14.1/zig /usr/local/bin/zig",
        "zig version",
    )
    .env(
        {
            "PATH": "/opt/zig-x86_64-linux-0.14.1:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
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
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
def _run_checked(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError as e:
        return 127, "", str(e)
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            return value
    except (OSError, json.JSONDecodeError):
        return None
    return None
def _write_json_file(path: Path, value: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)
def _count_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
def _extract_text_from_row(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("text", "content", "sentence", "article"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for val in row.values():
        if isinstance(val, str) and len(val.strip()) > 50:
            return val.strip()
    return ""
DATASET_NAME = "HuggingFaceFW/finephrase"
DATASET_CONFIG = "faq"
DATASET_MAX_SAMPLES = 100_000
def download_finephrase_to_jsonl(volume: modal.Volume) -> Tuple[str, int, int]:
    from datasets import load_dataset
    _ensure_dir(DATASET_DIR)
    if DATASET_FILE.is_file() and DATASET_FILE.stat().st_size > 0:
        size = int(DATASET_FILE.stat().st_size)
        metadata = _read_json_file(DATASET_METADATA_FILE)
        if metadata is not None and int(metadata.get("dataset_size", -1)) == size and int(metadata.get("line_count", 0)) > 0:
            return str(DATASET_FILE), size, int(metadata["line_count"])
        line_count = _count_lines(DATASET_FILE)
        if line_count > 0:
            _write_json_file(DATASET_METADATA_FILE, {"dataset_path": str(DATASET_FILE), "dataset_size": size, "line_count": line_count})
            volume.commit()
            return str(DATASET_FILE), size, line_count
        DATASET_FILE.unlink()
    tmp_file = DATASET_FILE.with_suffix(".jsonl.tmp")
    if tmp_file.exists():
        tmp_file.unlink()
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    line_count = 0
    with open(tmp_file, "w", encoding="utf-8") as f_out:
        for row in ds:
            text = _extract_text_from_row(row)
            if text and len(text) > 20:
                f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                line_count += 1
                if line_count >= DATASET_MAX_SAMPLES:
                    break
    if line_count <= 0:
        if tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError("Dataset conversion produced zero usable samples")
    tmp_file.replace(DATASET_FILE)
    size = int(DATASET_FILE.stat().st_size)
    _write_json_file(DATASET_METADATA_FILE, {"dataset_path": str(DATASET_FILE), "dataset_size": size, "line_count": line_count})
    volume.commit()
    return str(DATASET_FILE), size, line_count
def _build_zig_gpu(project_dir: str, force: bool = False) -> None:
    if BINARY_PATH.is_file() and not force:
        return
    rc, out, err = _run_checked(
        ["zig", "build", "distributed-futhark", "-Dgpu=true", "-Doptimize=ReleaseFast"],
        cwd=project_dir,
    )
    if rc != 0:
        raise RuntimeError(f"GPU build failed with exit code {rc}: {(err or out)[-8000:]}")
    if not BINARY_PATH.is_file():
        raise FileNotFoundError(f"GPU binary not found at {BINARY_PATH}")
    BINARY_PATH.chmod(0o755)
def _ensure_binary_in_cache(volume: modal.Volume) -> str:
    """Build the GPU binary on a CPU container, then cache it in checkpoint volume.
    Returns the build_version (git-like hash of source) of the cached binary.
    """
    import hashlib
    def _source_hash() -> str:
        h = hashlib.sha256()
        for sub in ("src", "build.zig"):
            base = PROJECT_MOUNT_PATH / sub
            if base.is_file():
                with open(base, "rb") as f:
                    h.update(f.read())
            else:
                for p in sorted(base.rglob("*")):
                    if not p.is_file():
                        continue
                    if any(part in {".zig-cache", "zig-out"} for part in p.parts):
                        continue
                    with open(p, "rb") as f:
                        h.update(p.name.encode())
                        h.update(f.read())
        return h.hexdigest()
    expected_version = _source_hash()
    cached_version = ""
    if BUILD_VERSION_FILE.is_file():
        try:
            cached_version = BUILD_VERSION_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            cached_version = ""
    if BINARY_CACHE_PATH.is_file() and cached_version == expected_version:
        print(f"Reusing cached GPU binary ({expected_version[:12]})")
        return expected_version
    print(f"Building GPU binary (version {expected_version[:12]})")
    os.chdir(str(PROJECT_MOUNT_PATH))
    _build_zig_gpu(str(PROJECT_MOUNT_PATH), force=True)
    _ensure_dir(CHECKPOINT_MOUNT_PATH)
    shutil.copy2(str(BINARY_PATH), str(BINARY_CACHE_PATH))
    BINARY_CACHE_PATH.chmod(0o755)
    BUILD_VERSION_FILE.write_text(expected_version, encoding="utf-8")
    volume.commit()
    print(f"Cached GPU binary at {BINARY_CACHE_PATH}")
    return expected_version
def _detect_gpus() -> Tuple[int, str]:
    try:
        p = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    except FileNotFoundError as e:
        return 0, str(e)
    output = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
    return len(lines), output
def _expected_gpu_count() -> int:
    if ":" not in GPU_SPEC:
        return 1
    try:
        return int(GPU_SPEC.rsplit(":", 1)[1])
    except ValueError:
        return 1
def _extract_loss(stdout: str) -> Optional[float]:
    if not stdout:
        return None
    loss_value = None
    pattern = re.compile(r"\b(?:loss|train_loss|training_loss)\b[^0-9+\-]*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)", re.IGNORECASE)
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            try:
                loss_value = float(match.group(1))
            except ValueError:
                continue
    return loss_value
def _read_tail(path: Path, max_chars: int = 8000) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    byte_count = min(size, max_chars * 4)
    with open(path, "rb") as f:
        if size > byte_count:
            f.seek(size - byte_count)
        data = f.read()
    return data.decode("utf-8", errors="replace")[-max_chars:]
@app.function(
    image=jaide_image,
    gpu=GPU_SPEC,
    cpu=(CPU_REQUEST, CPU_LIMIT),
    memory=(MEMORY_REQUEST_MB, MEMORY_LIMIT_MB),
    ephemeral_disk=EPHEMERAL_DISK_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
    },
)
def train_all_ranks(
    epochs: int = 5,
    model_dim: int = 2048,
    num_layers: int = 24,
    local_batch_size: int = 2,
    world_size: int = 8,
) -> Dict[str, Any]:
    import threading
    def _log(msg: str) -> None:
        print(f"[orch] {msg}", flush=True)
    _log("container started")
    data_volume.reload()
    checkpoint_volume.reload()
    gpu_count, gpu_list = _detect_gpus()
    expected_gpus = _expected_gpu_count()
    if gpu_count < world_size:
        raise RuntimeError(
            f"Need {world_size} GPUs, detected {gpu_count} from {GPU_SPEC}: {gpu_list}"
        )
    _log(f"detected {gpu_count} GPUs (expected {expected_gpus})")
    dataset_path_s, dataset_size, sample_count = download_finephrase_to_jsonl(data_volume)
    if sample_count <= 0:
        raise RuntimeError("Dataset contains zero samples")
    dataset_path = str(dataset_path_s)
    _log(f"dataset {sample_count} samples, {dataset_size / 1e6:.2f} MB")
    build_version = _ensure_binary_in_cache(checkpoint_volume)
    _log(f"GPU binary ready (build_version={build_version[:12]})")
    if not BINARY_CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"Pre-built GPU binary missing from {BINARY_CACHE_PATH}"
        )
    local_binary = Path("/tmp/jaide-distributed-futhark")
    shutil.copy2(str(BINARY_CACHE_PATH), str(local_binary))
    local_binary.chmod(0o755)
    _log(f"binary ready at {local_binary}")
    nccl_id_path = Path("/tmp/jaide_nccl_id")
    if nccl_id_path.exists():
        nccl_id_path.unlink()
    ready_path = Path("/tmp/jaide_nccl_id.ready")
    if ready_path.exists():
        ready_path.unlink()
    logs_dir = Path("/tmp/jaide_training_logs")
    _ensure_dir(logs_dir)
    _ensure_dir(CHECKPOINT_MOUNT_PATH)
    def _tee(src, dst_file, prefix: str) -> None:
        try:
            for raw in iter(src.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                print(f"[{prefix}] {line}", flush=True)
                dst_file.write(line + "\n")
                dst_file.flush()
        except Exception as exc:
            print(f"[{prefix}] tee error: {exc}", flush=True)
    procs: List[subprocess.Popen] = []
    rank_files: List[Tuple[Path, Path, Any, Any, threading.Thread, threading.Thread]] = []
    base_env = os.environ.copy()
    base_env["WORLD_SIZE"] = str(world_size)
    base_env["MASTER_ADDR"] = "127.0.0.1"
    base_env["MASTER_PORT"] = "29500"
    base_env["JAIDE_EPOCHS"] = str(epochs)
    base_env["JAIDE_DATASET"] = dataset_path
    base_env["JAIDE_MODEL_DIM"] = str(model_dim)
    base_env["JAIDE_LAYERS"] = str(num_layers)
    base_env["JAIDE_BATCH_SIZE"] = str(local_batch_size)
    base_env["JAIDE_NCCL_ID_PATH"] = str(nccl_id_path)
    base_env["JAIDE_TOTAL_SAMPLES"] = str(sample_count)
    base_env["JAIDE_MAX_SAMPLES"] = str(sample_count)
    base_env["JAIDE_MAX_SEQ_LEN"] = "2048"
    base_env["JAIDE_LEARNING_RATE"] = "0.0001"
    base_env["JAIDE_REASONING_CYCLES"] = "1"
    base_env["JAIDE_RELATIONAL_PASS_INTERVAL"] = "50"
    if VOCAB_PATH.is_file() and VOCAB_PATH.stat().st_size > 0:
        base_env["JAIDE_VOCAB_READY"] = "1"
        _log(f"existing vocab found at {VOCAB_PATH} ({VOCAB_PATH.stat().st_size} bytes), skipping BPE training (JAIDE_VOCAB_READY=1)")
    else:
        base_env.pop("JAIDE_VOCAB_READY", None)
        _log(f"no valid vocab at {VOCAB_PATH}, BPE training will run on rank 0")
    base_env["NCCL_DEBUG"] = "WARN"
    base_env["NCCL_IB_DISABLE"] = "1"
    base_env["NCCL_SOCKET_IFNAME"] = "lo"
    base_env["NCCL_P2P_DISABLE"] = "0"
    base_env["NCCL_SHM_DISABLE"] = "0"
    base_env["NCCL_NVLS_ENABLE"] = "0"
    base_env["NCCL_LAUNCH_MODE"] = "GROUP"
    base_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    start_time = time.time()
    _log(f"spawning {world_size} ranks in this container")
    for rank in range(world_size):
        env = base_env.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["JAIDE_LOCAL_RANK"] = str(rank)
        stdout_path = logs_dir / f"rank_{rank:03d}.stdout.log"
        stderr_path = logs_dir / f"rank_{rank:03d}.stderr.log"
        stdout_file = open(stdout_path, "w", encoding="utf-8", errors="replace")
        stderr_file = open(stderr_path, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [str(local_binary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(PROJECT_MOUNT_PATH),
        )
        t_out = threading.Thread(
            target=_tee, args=(proc.stdout, stdout_file, f"r{rank} out"), daemon=True
        )
        t_err = threading.Thread(
            target=_tee, args=(proc.stderr, stderr_file, f"r{rank} err"), daemon=True
        )
        t_out.start()
        t_err.start()
        procs.append(proc)
        rank_files.append((stdout_path, stderr_path, stdout_file, stderr_file, t_out, t_err))
        _log(f"rank {rank} pid={proc.pid} LOCAL_RANK={env['LOCAL_RANK']}")
    results = []
    timed_out_any = False
    for rank, proc in enumerate(procs):
        try:
            rc = proc.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out_any = True
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            rc = -9
        results.append((rank, int(rc)))
        _log(f"rank {rank} exit rc={rc}")
    for _, _, sout, serr, t_out, t_err in rank_files:
        try:
            t_out.join(timeout=5)
            t_err.join(timeout=5)
        finally:
            sout.close()
            serr.close()
    elapsed = float(time.time() - start_time)
    rank_results: List[Dict[str, Any]] = []
    for rank, rc in results:
        stdout_path, stderr_path, _, _, _, _ = rank_files[rank]
        stdout_tail = _read_tail(stdout_path)
        stderr_tail = _read_tail(stderr_path)
        loss = _extract_loss(stdout_tail) if rank == 0 else None
        rank_results.append(
            {
                "rank": rank,
                "return_code": rc,
                "timed_out": timed_out_any and rc == -9,
                "loss": loss,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )
    successful_ranks = [r for r in rank_results if int(r.get("return_code", -1)) == 0 and not r.get("timed_out", False)]
    rank_0_result = next((r for r in rank_results if r.get("rank") == 0), None)
    final_loss = rank_0_result.get("loss") if rank_0_result else 0.0
    completed_epochs = epochs if len(successful_ranks) == world_size else 0
    status = "completed" if len(successful_ranks) == world_size else "failed"
    _write_json_file(
        CHECKPOINT_MOUNT_PATH / "training_complete.json",
        {
            "status": status,
            "total_epochs": epochs,
            "completed_epochs": completed_epochs,
            "final_loss": final_loss,
            "dataset_path": dataset_path,
            "dataset_size_mb": float(dataset_size) / 1e6,
            "sample_count": sample_count,
            "gpu_config": f"{world_size}x NVIDIA B200",
            "model_dim": model_dim,
            "num_layers": num_layers,
            "local_batch_size": local_batch_size,
            "world_size": world_size,
            "rank_results": rank_results,
        },
    )
    checkpoint_volume.commit()
    return {
        "status": status,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "final_loss": final_loss,
        "dataset_size_mb": float(dataset_size) / 1e6,
        "sample_count": sample_count,
        "gpu_config": f"{world_size}x NVIDIA B200",
        "model_dim": model_dim,
        "num_layers": num_layers,
        "successful_ranks": len(successful_ranks),
        "elapsed_seconds": elapsed,
        "rank_results": rank_results,
    }
@app.local_entrypoint()
def main(
    epochs: int = 5,
    model_dim: int = 2048,
    num_layers: int = 24,
    local_batch_size: int = 2,
    world_size: int = 8,
) -> None:
    result = train_all_ranks.remote(
        epochs=epochs,
        model_dim=model_dim,
        num_layers=num_layers,
        local_batch_size=local_batch_size,
        world_size=world_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
