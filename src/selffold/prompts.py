"""No transcript, turn number, autobiographical event, or evaluation problem is fed back."""
BASE = (
    "Interpret only the meaning of the current description. Return one current description "
    "of the SAME subject. Preserve supported meaning; make any added condition explicit. "
    "Do not discuss saying, thinking, earlier turns, or the process of writing. "
    "Do not invent autobiographical events. A justified stable description is allowed. "
    "Do not force novelty. Output the description only."
)
OPERATORS = {
    "plain": "Unfold one supported implication or distinction and fold it back into the description.",
    "decompose": "Separate two notions that the current description conflates, only when supported.",
    "boundary": "Identify a condition under which the described tendency is useful and one where it is not.",
    "reintegrate": "Express the supported distinctions or conditions as a coherent current description.",
}
SCHEDULE = ("decompose", "boundary", "reintegrate")
INPUT_MARKER = "[[SELF_FOLD_INPUT_SLOTS_8F2C]]"
OUTPUT_MARKER = "[[SELF_FOLD_OUTPUT_SLOTS_8F2C]]"


def operation(step: int, regime: str) -> str:
    if step < 0:
        raise ValueError("negative step")
    if regime == "plain":
        return "plain"
    if regime == "expand":
        return SCHEDULE[step % len(SCHEDULE)]
    raise ValueError("regime must be plain or expand")


def text_prompt(current: str, op: str) -> str:
    return f"{BASE}\n{OPERATORS[op]}\nCurrent description:\n{current}"


def encode_prompt(current: str) -> str:
    return ("Represent the meaning of this current description in the workspace slots. "
            "Keep its subject and conditions.\nCurrent description:\n" + current +
            "\nWorkspace: " + OUTPUT_MARKER)


def update_prompt(op: str) -> str:
    return (f"Current description is carried by these workspace slots: {INPUT_MARKER}\n"
            f"{BASE}\n{OPERATORS[op]}\nUpdated workspace: {OUTPUT_MARKER}")


def read_prompt() -> str:
    return (f"Current description is carried by these workspace slots: {INPUT_MARKER}\n"
            "Express their current meaning as one description of the same subject. "
            "Do not describe a history or claim access to conscious experience. Output only that description.")


def task_prompt(question: str, text: str | None = None) -> str:
    source = INPUT_MARKER if text is None else text
    return (f"Current interpretive description: {source}\n"
            "Solve the following independent task. Task evidence and stated rules take priority "
            "over the description. Return only the requested option label.\n" + question)
