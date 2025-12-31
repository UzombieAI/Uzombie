# launch_magpie_v2.py
import argparse
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.getcwd(), "src"))

from uzombie.data.builders import generate_magpie_reasoning_v2
from uzombie.utils.logger import console

def main():
    parser = argparse.ArgumentParser(description="Uzombie Magpie V2 Launcher")
    parser.add_argument("--teacher", type=str, default="unsloth/Llama-3.1-8B-Instruct-bnb-4bit", help="Teacher model ID")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="magpie_v2_dataset", help="Output folder/name")
    args = parser.parse_args()

    console.print(f"[bold magenta]🚀 Launching Magpie V2 Test with {args.teacher}[/]")

    # Call the function from builders.py
    dataset = generate_magpie_reasoning_v2(
        teacher_model_id=args.teacher,
        num_samples=args.samples,
        active_iterations=1,
        temperature_query=1.0,    # High temp for diverse questions
        temperature_response=0.6, # Low temp for precise reasoning
    )

    if dataset and len(dataset) > 0:
        console.print("\n[bold cyan]🔍 Inspecting Sample #1:[/]")
        console.print(f"[yellow]Instruction:[/]\n{dataset[0]['instruction']}")
        console.print(f"[green]Response (CoT):[/]\n{dataset[0]['output'][:500]}...") 
        
        # Save to disk
        dataset.save_to_disk(args.out)
        console.print(f"[bold green]✅ Dataset saved to folder: {args.out}[/]")
    else:
        console.print("[bold red]❌ No samples survived the quality filter. Try increasing --samples or checking your model.[/]")

if __name__ == "__main__":
    main()