"""One sealed margin continuation from the closed literal curriculum endpoint."""

from fit_conditional_curriculum import ROOT, STAGES, main, run
from selffold.records import file_sha

STUDY = ROOT / "provenance/conditional_margin_20261103"
DATA = ROOT / "data/conditional_margin_20261103.json"
PARENT = ROOT / "provenance/conditional_curriculum_20261103/run/warm_task.npz"


def continuation(args, store):
    if (
        file_sha(PARENT)
        != "4fb28b1c00ccebf89387f3dfb1793bef3c9ea8bedf782cb0d359e23943cd7d5c"
    ):
        raise ValueError("The sealed parent task endpoint differs")
    return run(
        args,
        store,
        data=DATA,
        study=STUDY,
        warm_start=PARENT,
        margin=2.0,
        warm_stages=[STAGES[i % 3] for i in range(384)],
        completion_status="COMPLETED_CONDITIONAL_MARGIN",
    )


if __name__ == "__main__":
    main(continuation)
