"""Local-only loading; custom Python requires an explicit hash-bound review receipt."""
from pathlib import Path
import json
import torch
import yaml
from .records import digest, file_sha

SUPPORTED = {"mistral", "llama", "gemma2", "phimoe"}


def local_identity(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError("Use an existing local Hugging Face checkpoint directory")
    # Strong identity streams model weights once; no bytes are published.
    allowed = {".json", ".safetensors", ".model", ".txt", ".jinja", ".py"}
    files = {str(p.relative_to(path)): file_sha(p) for p in sorted(path.rglob("*"))
             if p.is_file() and p.suffix in allowed and ".cache" not in p.parts}
    if "config.json" not in files or not any(x.endswith(".safetensors") for x in files):
        raise ValueError("Require config.json and safetensors weights; pickle/bin loads are disabled")
    return {"files": files, "checkpoint_sha256": digest(files)}


def require_native_geometry(config: dict):
    """v4.57 native Phimoe derives head width; SlimMoE may have pruned attention width."""
    if config.get("model_type") == "phimoe":
        derived = config["hidden_size"] // config["num_attention_heads"]
        if config.get("head_dim", derived) != derived:
            raise ValueError("SlimMoE attention geometry differs from native Phimoe. Stop; use an explicitly reviewed local implementation, not a silent fallback.")


def reviewed_local_code(path: Path, receipt: Path, checkpoint_sha256: str) -> dict:
    record = json.loads(receipt.read_text())
    actual = {str(p.relative_to(path)): file_sha(p) for p in sorted(path.rglob("*.py"))
              if p.is_file() and ".cache" not in p.parts}
    if not actual:
        raise ValueError("No local custom Python to review")
    if record.get("status") != "LOCAL_CODE_REVIEWED" or not record.get("reviewer"):
        raise ValueError("Local code approval is pending")
    if record.get("checkpoint_sha256") != checkpoint_sha256 or record.get("python_files") != actual:
        raise ValueError("Local code review does not match exact checkpoint/code bytes")
    return record


def load_local(path: Path, profile: dict, *, device="cuda:0", dtype="bfloat16", local_code_review=None):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    ident = local_identity(path)
    raw_config = json.loads((path / "config.json").read_text())
    trust_local = local_code_review is not None
    if trust_local:
        reviewed_local_code(path, Path(local_code_review), ident["checkpoint_sha256"])
        ident["local_code_review_sha256"] = file_sha(Path(local_code_review))
    else:
        require_native_geometry(raw_config)
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=trust_local)
    if config.model_type not in SUPPORTED or config.model_type != profile["model_type"]:
        raise ValueError(f"Unqualified model_type: {config.model_type}")
    if getattr(config, "quantization_config", None):
        raise ValueError("Baseline disallows quantization; make a separate, reviewed quantization arm")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=trust_local)
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=trust_local, use_safetensors=True,
        torch_dtype=torch_dtype, attn_implementation="eager", low_cpu_mem_usage=True,
        output_loading_info=True)
    problems = {k: loading_info.get(k) for k in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading_info.get(k)}
    if problems:
        raise ValueError("Checkpoint loading is not exact; no randomly initialized fallback parameters: " + str(problems))
    ident["loading_info"] = loading_info
    ident["custom_local_code_reviewed"] = trust_local
    model.to(device)
    return model, tok, ident
