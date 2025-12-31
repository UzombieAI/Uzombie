import sys
import os
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from uzombie.data.builders import generate_magpie, upload_dataset
from uzombie.utils.logger import console

def main():
    # Hardcoded config path or arg
    config_path = "configs/data/magpie-reasoning.yaml"
    if not os.path.exists(config_path):
        console.print(f"[red]Config not found at {config_path}[/]")
        return

    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
    
    dataset = generate_magpie(
        cfg['teacher_model'], 
        cfg['num_samples'], 
        cfg['template'], 
        cfg['cot_prompt'], 
        cfg['active_iterations']
    )
    
    if dataset:
        upload_dataset(dataset, cfg['repo_id'])

if __name__ == "__main__":
    main()