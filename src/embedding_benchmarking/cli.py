import os
from datetime import datetime
from pathlib import Path
from time import time

import click
import pandas as pd
from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import Config
from .data import DataManager
from .evaluation import Evaluator
from .models import ModelManager
from .utils import console, setup_logging


@click.group()
def cli():
    """Academic Embedding Model Evaluator CLI."""
    pass


def _format_time(seconds: float) -> str:
    """Format seconds into a compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _result_row(model_name: str, scores: dict) -> dict:
    """Convert model score tuples into a CSV-friendly result row."""
    return {
        'Model': model_name,
        'Title-Own Abstract Mean': f"{scores['title_abstract_same'][0]:.3f}",
        'Title-Own Abstract Std': f"{scores['title_abstract_same'][1]:.3f}",
        'Title-Diff Abstract (Same Field) Mean': f"{scores['title_abstract_diff'][0]:.3f}",
        'Title-Diff Abstract (Same Field) Std': f"{scores['title_abstract_diff'][1]:.3f}",
        'Title-Diff Abstract (Diff Field) Mean': f"{scores['title_abstract_other'][0]:.3f}",
        'Title-Diff Abstract (Diff Field) Std': f"{scores['title_abstract_other'][1]:.3f}",
        'Abstract-Abstract (Same Field) Mean': f"{scores['abstract_abstract_same'][0]:.3f}",
        'Abstract-Abstract (Same Field) Std': f"{scores['abstract_abstract_same'][1]:.3f}",
        'Abstract-Abstract (Diff Field) Mean': f"{scores['abstract_abstract_diff'][0]:.3f}",
        'Abstract-Abstract (Diff Field) Std': f"{scores['abstract_abstract_diff'][1]:.3f}",
    }


@click.command()
@click.option('--cache-dir', default='embedding_cache', show_default=True, help='Cache directory')
@click.option('--max-tokens', default=512, show_default=True, help='Maximum tokens in abstract')
@click.option('--min-tokens', default=50, show_default=True, help='Minimum tokens in abstract')
@click.option('--config', default='config/config.yaml', show_default=True, type=click.Path(exists=True), help='Config file path')
def evaluate(cache_dir: str, max_tokens: int, min_tokens: int, config: str):
    """Run embedding model evaluation."""
    load_dotenv()
    setup_logging()

    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise click.ClickException("HUGGINGFACE_TOKEN not found. Set it in your environment or .env file.")

    console.print("\n[bold green]🚀 Starting Academic Embedding Model Evaluation[/bold green]")

    try:
        config_manager = Config(
            config_path=config,
            cache_dir=cache_dir,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        model_manager = ModelManager(device=None, hf_token=hf_token)
        data_manager = DataManager(config_manager, model_manager)
        evaluator = Evaluator(config_manager, model_manager)

        console.print("\n[yellow]Fetching papers...[/yellow]")
        papers = data_manager.fetch_papers()
        if not papers:
            raise click.ClickException("Paper collection failed; no valid papers were collected.")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = Path('experiments') / f'experiment_{timestamp}'

        console.print("\n[yellow]Saving experiment metadata...[/yellow]")
        evaluator.save_experiment_metadata(papers, experiment_dir)

        results = []
        total_papers = len(papers)
        total_models = len(config_manager.models)
        completed_times = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            transient=False,
            refresh_per_second=5,
        ) as progress:
            overall_task = progress.add_task("[bold cyan]Overall Progress", total=total_models)
            model_task = progress.add_task("Current Model Steps", total=100, visible=False)

            for index, (model_name, model_path) in enumerate(config_manager.models.items(), start=1):
                model_start_time = time()
                try:
                    if completed_times:
                        avg_time = sum(completed_times) / len(completed_times)
                        remaining_models = total_models - len(completed_times)
                        description = (
                            f"[bold cyan]Processing Models ({index - 1}/{total_models}) "
                            f"- Est. {_format_time(avg_time * remaining_models)} remaining"
                        )
                    else:
                        description = f"[bold cyan]Processing Models ({index - 1}/{total_models})"

                    progress.update(overall_task, description=description)
                    progress.update(
                        model_task,
                        completed=0,
                        description=f"[bold yellow]{model_name}[/bold yellow]: Preparing evaluation",
                        visible=True,
                    )

                    scores = evaluator.evaluate_model(
                        papers=papers,
                        model_name=model_path,
                        progress=progress,
                        progress_task=model_task,
                        total_papers=total_papers,
                    )
                    completed_times.append(time() - model_start_time)
                    results.append(_result_row(model_name, scores))
                    progress.advance(overall_task)
                    console.print(f"✅ Processed: [green]{model_name}[/green]")

                except Exception as exc:
                    progress.advance(overall_task)
                    console.print(f"❌ Failed: [red]{model_name}[/red] - {exc}")

        if not results:
            raise click.ClickException("All model evaluations failed; no leaderboard was created.")

        console.print("\n[yellow]Saving results...[/yellow]")
        results_df = pd.DataFrame(results)
        results_df.to_csv(experiment_dir / 'embedding_comparison_results.csv', index=False)
        evaluator.create_leaderboard(results_df, experiment_dir)
        console.print(f"\n✅ Experiment complete: [cyan]{experiment_dir}[/cyan]")

    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
