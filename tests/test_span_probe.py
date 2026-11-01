import json
from pathlib import Path

from selffold.host import SlotHost
from selffold.records import RunStore
from selffold.span_probe import BLOCKS, make_layout, run_span_probe
from selffold.tiny import ByteTokenizer, TinyCausalLM


class OffsetBytes(ByteTokenizer):
    def __call__(self, text, **kwargs):
        offsets = [(i, i + 1) for i, ch in enumerate(text) for _ in ch.encode()]
        return {"input_ids": self.encode(text), "offset_mapping": offsets}


def test_spans_partition_the_real_prompt_and_exclude_the_dummy():
    host = SlotHost(TinyCausalLM(), OffsetBytes(), slots=8)
    group = {"id": "unit", "question": "Which path?", "options": ["water", "land"]}
    layout = make_layout(host, group, "B")
    ranges = [range(*layout.windows[k]) for k in BLOCKS]
    assert [i for block in ranges for i in block] == list(range(layout.slots[-1]+1, len(layout.ids)-1))
    a, b = layout.windows["options"]
    assert "A = land. B = water." in host.tokenizer.decode(layout.ids[a:b])
    assert all(layout.windows[k][1] - layout.windows[k][0] == 8 for k in ("last8", "options8", "first8"))
    a, b = layout.windows["score"]
    assert (a, b) == (len(layout.ids)-2, len(layout.ids)-1)


def test_localization_and_counterfactuals_replay_matched_geometry(tmp_path):
    host = SlotHost(TinyCausalLM(), OffsetBytes(), slots=8)
    store = RunStore(tmp_path / "localize")
    run_span_probe(host, store, [{"id": "unit", "question": "Which path?", "options": ["water", "land"]}])
    prompts = [json.loads(s) for s in (store.path / "prompts.jsonl").read_text().splitlines()]
    assert len(prompts) == 2
    for prompt in prompts:
        a = prompt["windows"]["question"][0]
        b = prompt["windows"]["closing"][0]
        assert prompt["masked_neutral_ids"][a:b] == [host.placeholder_id]*(b-a)
        assert prompt["masked_neutral_ids"][b:] == prompt["recipient_ids"][b:]
    rows = [json.loads(s) for s in (store.path / "interventions.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        assert row["reference"]["slots_and_suffix"]["max_abs_vocab_from_native"] == 0
        assert row["reference"]["neutral_replay_max_abs"] == 0
        for context in row["contexts"].values():
            assert context["status"] == "matched_length" and context["pre_question_max_abs"] == 0
            assert all(w["status"] == "measured" for w in context["windows"].values())
        for name, w in row["contexts"]["same"]["windows"].items():
            assert w["raw"] == row["regions"][name]["only"]["scores"]
        assert set(row["regions"]["score"]) == {"only", "omit", "native_knockout"}


def test_prospective_topics_are_separate_from_the_original_panel():
    data = Path(__file__).resolve().parents[1] / "data"
    old = json.loads((data / "slot_contrasts.dev.json").read_text())["groups"]
    new = json.loads((data / "span_topics.prospective.json").read_text())["groups"]
    assert len({g["id"] for g in new}) == len(new) == 6
    assert not {g["id"] for g in old} & {g["id"] for g in new}
    assert not {a for g in old for a in g["options"]} & {a for g in new for a in g["options"]}
