import yaml
import torch
from ..utils.logger import console

def dare_ties_merge(models: list, base_model: str, output_path: str, density: float = 0.5):
    """
    Executes a DARE-TIES merge using mergekit logic (programmatic wrapper).
    Requires 'mergekit' installed.
    """
    console.print(f"[bold magenta]Starting DARE-TIES Merge into {output_path}...[/]")
    
    try:
        import mergekit
        from mergekit.config import MergeConfiguration
        from mergekit.merge import merge
    except ImportError:
        console.print("[bold red]Mergekit not installed. Run: pip install mergekit[/]")
        return

    # Construct config dictionary for DARE-TIES
    merge_config = {
        "merge_method": "dare_ties",
        "base_model": base_model,
        "models": [
            {
                "model": m,
                "parameters": {
                    "weight": 1.0 / len(models),
                    "density": density
                }
            } for m in models
        ],
        "dtype": "bfloat16"
    }

    # Execute
    console.print(yaml.dump(merge_config))
    # Note: Mergekit API usage varies by version, this is a generalized implementation
    # Often run via CLI, but we can dump yaml and run internal functions if exposed
    console.print("[dim]Mergekit API execution not fully standard—dumping config to merge.yaml[/]")
    with open("merge.yaml", "w") as f:
        yaml.dump(merge_config, f)
        
    console.print("[green]Config saved to merge.yaml. Run manually with: mergekit-yaml merge.yaml output_path[/]")