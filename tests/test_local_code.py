import json
from types import SimpleNamespace
import pytest
from selffold.loading import require_native_geometry, reviewed_local_code
from selffold.records import file_sha

def test_slim_geometry_fails_before_loading():
    with pytest.raises(ValueError,match="geometry"):
        require_native_geometry(dict(model_type="phimoe",hidden_size=4096,num_attention_heads=16,head_dim=128))

def test_native_geometry_allows_equal_head():
    require_native_geometry(dict(model_type="phimoe",hidden_size=4096,num_attention_heads=32,head_dim=128))

def test_non_phi_untouched():
    require_native_geometry(dict(model_type="gemma2",hidden_size=2304,num_attention_heads=8,head_dim=256))

def test_local_code_requires_exact_review(tmp_path):
    model=tmp_path/"model";model.mkdir();code=model/"custom.py";code.write_text("# never executed\n")
    receipt=tmp_path/"approval.json"
    record={"status":"PENDING","reviewer":None,"checkpoint_sha256":"x","python_files":{"custom.py":file_sha(code)}}
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="pending"):reviewed_local_code(model,receipt,"x")
    record.update(status="LOCAL_CODE_REVIEWED",reviewer="unit_test_fixture_not_a_real_approval")
    receipt.write_text(json.dumps(record));assert reviewed_local_code(model,receipt,"x")["reviewer"]
    code.write_text("# changed\n")
    with pytest.raises(ValueError,match="match"):reviewed_local_code(model,receipt,"x")
