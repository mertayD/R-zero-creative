"""Fire-and-forget launcher for the challenger eval harness.

`modal_app.py::challenger_eval` blocks on `.remote()`, which ties the GPU job
to the local client: if the laptop sleeps or the terminal dies mid-run, Modal
cancels the in-flight call even under `--detach`. This entrypoint `.spawn()`s
the function instead and returns immediately; the app keeps running on its
own and results land on the volume + W&B as usual.

    modal run --detach launch_challenger_eval.py --checkpoint Qwen/Qwen3-4B-Base --judge-type sft-critic
    modal app logs <app-id>   # to follow
"""
from modal_app import app, run_challenger_eval


@app.local_entrypoint()
def launch(
    checkpoint: str,
    judge_type: str = "claude",
    dup_method: str = "embedding",
    step: int = 0,
    limit: int = 0,
    dataset_repo: str = "",
    critic_url: str = "",
):
    import os

    call = run_challenger_eval.spawn(
        checkpoint,
        dataset_repo=dataset_repo,
        judge_type=judge_type,
        limit=limit or None,
        step=step or None,
        wandb_group=os.environ.get("WANDB_RUN_GROUP", ""),
        dup_method=dup_method,
        critic_url=critic_url,
    )
    print(f"spawned challenger eval: function call {call.object_id}")
