"""Score fixed lexical teachers under generic and concrete question operations."""

from fit_teacher_and_polarity import NativePairs, make_tasks
from selffold import prompts
from selffold.records import digest

MODES = ("generic", "concrete")


def question_groups(groups, fixture):
    """Keep the topic heading and scaffold fixed while substituting the operation."""
    expected = {g["id"] for g in groups}
    if (
        set(fixture["topic_cues"]) != expected
        or set(fixture["concrete_operations"]) != expected
        or len(fixture["scaffolds"]) != 4
        or len(set(fixture["scaffolds"])) != 4
        or any(s.count("{operation}") != 1 for s in fixture["scaffolds"])
    ):
        raise ValueError("Incomplete paired question-operation fixture")
    result = []
    for g in groups:
        topic = g["id"]
        operations = (
            fixture["generic_operation"],
            fixture["concrete_operations"][topic],
        )
        questions = [
            fixture["topic_cues"][topic] + "\n" + scaffold.format(operation=op)
            for scaffold in fixture["scaffolds"]
            for op in operations
        ]
        if len(set(questions)) != 8:
            raise ValueError("Question operation failed to change the question")
        result.append(g | {"fresh_questions": questions})
    return result


def record_geometry(host, groups, store):
    """Retain the actual lengths and positions for each matched pair of questions."""
    for topic in range(len(groups)):
        for family in range(4):
            for order in (0, 1):
                tasks = [
                    make_tasks(groups, topic, 0, "fresh", 2 * family + i)[order]
                    for i in range(2)
                ]
                rendered = [
                    host.render(prompts.task_prompt(task["question"])) for task in tasks
                ]
                slots = [marks[prompts.INPUT_MARKER] for _, marks in rendered]
                if slots[0] != slots[1]:
                    raise ValueError(
                        "Question operation moved the supplied input slots"
                    )
                store.append(
                    "teacher_question_layout.jsonl",
                    {
                        "topic": topic,
                        "family": family,
                        "label_order": order,
                        "modes": list(MODES),
                        "prompt_tokens": [len(ids) for ids, _ in rendered],
                        "input_slots": slots,
                        "question_sha256": [digest(t["question"]) for t in tasks],
                        "prompt_sha256": [digest(ids) for ids, _ in rendered],
                    },
                )


def evaluate(host, adapter, teachers, null, groups, store):
    """Use two native label orders per call and a fixed teacher for each policy."""
    if len(groups) != 8 or len(teachers) != 16:
        raise ValueError("Question-operation panel requires all eight known topics")
    record_geometry(host, groups, store)
    native = NativePairs(host, adapter, groups, store)
    for topic in range(len(groups)):
        for family in range(4):
            for i, mode in enumerate(MODES):
                for pole in (0, 1):
                    for route, state in (
                        ("calibrated", teachers[2 * topic + pole]),
                        ("mean", null),
                    ):
                        for row in native.score(
                            topic, pole, "fresh", 2 * family + i, state
                        ):
                            store.append(
                                "teachers.jsonl",
                                row | {"mode": mode, "family": family, "route": route},
                            )
