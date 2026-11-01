import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
import torch
from selffold.tiny import TinyCausalLM, ByteTokenizer
from selffold.host import SlotHost

torch.set_num_threads(1)
@pytest.fixture
def host():
    return SlotHost(TinyCausalLM(),ByteTokenizer(),slots=3,max_context=1536,max_new_tokens=3,wall_seconds=120)
