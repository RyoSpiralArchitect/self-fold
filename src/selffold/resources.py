"""Non-destructive inventory. No environment dumps, credentials, process killing or downloads."""
from pathlib import Path
import os
import platform
import shutil
import subprocess
import importlib.metadata as md
import torch


def inventory(path: Path = Path(".")) -> dict:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "numpy", "PyYAML", "pytest"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in {"MemTotal", "MemAvailable"}:
                memory[k + "_bytes"] = int(v.strip().split()[0]) * 1024
    disk = shutil.disk_usage(path)
    result = {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
              "cpu_count": os.cpu_count(), "packages": packages, "memory": memory,
              "disk_free_bytes": disk.free, "cuda_available": torch.cuda.is_available(),
              "torch_cuda": torch.version.cuda, "gpu": []}
    if torch.cuda.is_available():
        result["compiled_cuda_arches"] = torch.cuda.get_arch_list()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            result["gpu"].append({"index": i, "name": p.name, "capability": list(torch.cuda.get_device_capability(i)),
                                  "total_bytes": total, "free_bytes": free})
    if shutil.which("nvidia-smi"):
        try:
            result["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
                text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi"] = "unavailable"
    return result


def admit(*, device: str, required_free_gib=0., host_free_gib=0., disk_free_gib=1., path=Path(".")):
    info = inventory(path)
    if info["disk_free_bytes"] < disk_free_gib * 2**30:
        raise RuntimeError("Disk admission floor not met")
    available = info["memory"].get("MemAvailable_bytes")
    if available is not None and available < host_free_gib * 2**30:
        raise RuntimeError("Host memory admission floor not met")
    if device.startswith("cuda"):
        if not info["cuda_available"]:
            raise RuntimeError("CUDA requested but not available")
        index = int(device.split(":")[1]) if ":" in device else 0
        if info["gpu"][index]["free_bytes"] < required_free_gib * 2**30:
            raise RuntimeError("GPU free-memory admission floor not met; do not kill other jobs")
    return info
