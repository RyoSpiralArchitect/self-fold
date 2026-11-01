"""Scientific admission is separate from numerical plumbing."""
from pathlib import Path
import json
from .records import digest, file_sha

REQUIRED = {"content_discrimination", "meaning_swap_effect", "one_step_update",
            "readout_fidelity", "positional_controls", "native_parity"}

def verify_gate(path: Path, *, host_identity: dict, source_sha256: str):
    obj = json.loads(path.read_text())
    if obj.get("status") != "SEMANTIC_GATE_REVIEWED_PASS":
        raise ValueError("Semantic gate has not passed")
    if obj.get("host_identity_sha256") != digest(host_identity):
        raise ValueError("Gate belongs to a different model/tokenizer/transport/protocol")
    if obj.get("source_sha256") != source_sha256:
        raise ValueError("Gate belongs to a different source tree")
    if not obj.get("reviewer") or obj.get("split") != "calibration_only":
        raise ValueError("Need explicit reviewer and calibration-only evidence")
    checks = obj.get("checks", {})
    if set(checks) != REQUIRED or any(v.get("verdict") != "pass" for v in checks.values()):
        raise ValueError("Missing or failed qualification check")
    for check in checks.values():
        p = (path.parent / check["evidence_path"]).resolve()
        if not p.is_file() or file_sha(p) != check.get("evidence_sha256"):
            raise ValueError("Missing or altered gate evidence")
    return obj
