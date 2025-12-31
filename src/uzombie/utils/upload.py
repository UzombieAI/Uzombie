# src/uzombie/utils/upload.py
from huggingface_hub import HfApi
from ..utils.logger import console

def push_to_hub_auto(trainer, repo_id: str, commit_message: str = "Uzombie v1 model"):
    """
    Safe upload for 4-bit/QoRA:
    - merge_and_unload to materialize weights before pushing
    - use safe serialization when available
    """
    console.print(f"[bold blue]Uploading to {repo_id}...[/bold blue]")
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=False)

    model_to_push = None
    try:
        # If PEFT/Unsloth provides merge_and_unload, use it
        if hasattr(trainer.model, "merge_and_unload"):
            model_to_push = trainer.model.merge_and_unload()
        else:
            model_to_push = trainer.model
    except Exception as e:
        console.print(f"[yellow]merge_and_unload failed ({e}) — pushing current model state[/]")
        model_to_push = trainer.model

    # Push model and tokenizer
    push_kwargs = {"commit_message": commit_message}
    if hasattr(model_to_push, "push_to_hub"):
        model_to_push.push_to_hub(repo_id, safe_serialization=True, **push_kwargs)
    else:
        # Fallback to save_pretrained then upload
        model_to_push.save_pretrained(repo_id, safe_serialization=True)
    trainer.tokenizer.push_to_hub(repo_id, **push_kwargs)

    # Upload README for HF card visibility
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=commit_message
    )
    console.print(f"[bold green]Uploaded: https://huggingface.co/{repo_id}[/bold green]")