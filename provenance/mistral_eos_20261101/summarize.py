"""Rebuild summary.json on stdout from the four archived trial directories."""
from collections import Counter
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEEDS = ("caution_plus_p0_self", "caution_minus_p0_self")
POLICIES = ("eos", "cap8")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(trial, seed, policy):
    request = read(trial / "request.json")
    host = read(trial / "host_identity.json")
    source = read(trial / "source.json")
    terminal = read(trial / "terminal.json")
    assert request["seed"]["id"] == seed and request["steps"] == 2 and request["regime"] == "plain"
    assert source["git_status"] == "" and terminal["pretrained_family_tested"]
    assert terminal["status"] == "ENGINEERING_PASS_NOT_SEMANTICS"
    assert host["max_context"] == 1024 and host["max_new_tokens"] == (None if policy == "eos" else 8)
    transitions = rows(trial / "transitions.jsonl")
    arms = {a: [r for r in transitions if r["arm"] == a] for a in "TLB"}
    assert all([r["step"] for r in arm] == [1, 2] for arm in arms.values())
    assert all(r["observation"] == r["action"]["text_update"] for r in arms["T"])

    generations = {
        "T_main": [r["action"]["text_update"] for r in arms["T"]],
        "B_main": [r["action"]["roundtrip_text"] for r in arms["B"]],
        "L_observer": [r["observation"] for r in arms["L"]],
        "B_observer": [r["observation"] for r in arms["B"]],
        "L_final_branch": [read(trial / "branches/language_roundtrip.json")["readout"]],
    }
    for group in generations.values():
        for g in group:
            remaining = host["max_context"] - g["prompt_tokens"]
            budget = remaining if policy == "eos" else min(8, remaining)
            assert g["effective_max_new_tokens"] == budget
            if g["stop"] == "eos":
                assert g["stop_token_id"] in host["eos_ids"] and not g["censored"] and g["token_count"] < budget
            else:
                assert g["stop"] in {"length", "context_limit"} and g["censored"]
                assert g["token_count"] == budget and g["stop_token_id"] is None
    assert sum(g["token_count"] + (g["stop"] == "eos")
               for group in generations.values() for g in group) == terminal["compute"]["generated_tokens"]

    states = {a: [read(trial / "states" / f"{a}_{i:03d}.json")["sha256"] for i in range(3)] for a in arms}
    tasks = rows(trial / "tasks.jsonl")
    assert len(tasks) == 12 and len({(t["arm"], t["step"], t["task_id"]) for t in tasks}) == 12
    for t in tasks:
        assert t["state_sha256"] == states[t["arm"]][t["step"]]
    correct = {a: [sum(t["correct"] for t in tasks if t["arm"] == a and t["step"] == i) for i in (0, 2)]
               for a in arms}
    return {
        "source": source["git_commit"], "checkpoint": host["checkpoint_sha256"],
        "native_parity_max_abs": read(trial / "native_parity.json")["max_abs"],
        "L_B_same_start": states["L"][0] == states["B"][0],
        "L_B_same_first_G": states["L"][1] == arms["B"][0]["action"]["updated_sha256_before_roundtrip"],
        "state_sha256": states,
        "generations": {role: [{"stop": g["stop"], "tokens": g["token_count"]} for g in group]
                        for role, group in generations.items()},
        "main_forwards": {a: sum(r["main_compute"]["forwards"] for r in arm) for a, arm in arms.items()},
        "task_correct_out_of_two_at_0_and_2": correct,
        "compute_including_observers_tasks_and_branch": terminal["compute"],
    }


def main():
    trials = {f"{s}-{p}": summarize(HERE / "trials" / f"{s}-{p}", s, p) for s in SEEDS for p in POLICIES}
    assert len({t["source"] for t in trials.values()}) == len({t["checkpoint"] for t in trials.values()}) == 1
    by_policy = {}
    for p in POLICIES:
        pair = [trials[f"{s}-{p}"] for s in SEEDS]
        events = [g for t in pair for group in t["generations"].values() for g in group]
        by_policy[p] = {
            "generation_stops_all_roles": dict(Counter(g["stop"] for g in events)),
            "distinct_post_update_states": {
                a: len({h for t in pair for h in t["state_sha256"][a][1:]}) for a in "TLB"},
        }
    result = {
        "trials": trials, "by_policy": by_policy,
        "L_state_hashes_equal_across_policies": {
            s: trials[f"{s}-eos"]["state_sha256"]["L"] == trials[f"{s}-cap8"]["state_sha256"]["L"] for s in SEEDS},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
