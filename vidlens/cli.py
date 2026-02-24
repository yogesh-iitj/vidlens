"""
VidLens CLI — analyze videos with computer vision from the command line.

Usage examples:
  vidlens analyze video.mp4 --lens objects --lens pose
  vidlens anonymize video.mp4 --mode blur
  vidlens highlights video.mp4 --duration 60
  vidlens train objects --data data/dataset.yaml --epochs 100
  vidlens lenses
"""

import typer
from pathlib import Path
from typing import Optional
from enum import Enum

app = typer.Typer(
    name="vidlens",
    help="🎬 VidLens — Open Source Video Intelligence Toolkit",
    add_completion=False,
)


class AnonymizeMode(str, Enum):
    blur = "blur"
    pixelate = "pixelate"
    black = "black"


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Path to input video file"),
    lens: list[str] = typer.Option(["objects"], "--lens", "-l", help="Lens(es) to run. Repeat for multiple."),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory"),
    device: str = typer.Option("auto", help="Device: auto | cpu | cuda | mps"),
    frame_skip: int = typer.Option(0, help="Process every N+1 frames (0 = all)"),
    preview: bool = typer.Option(False, "--preview", help="Show live preview window"),
    no_video: bool = typer.Option(False, "--no-video", help="Skip saving annotated video"),
    no_json: bool = typer.Option(False, "--no-json", help="Skip saving JSON report"),
):
    """
    Run one or more analysis lenses on a video.

    Examples:\n
      vidlens analyze myvideo.mp4 --lens objects\n
      vidlens analyze myvideo.mp4 --lens objects --lens pose --lens scene\n
      vidlens analyze myvideo.mp4 --lens objects --device cuda --frame-skip 2
    """
    _check_video(video)
    typer.echo(f"\n🔍 Running lenses: {lens}")

    from vidlens.pipeline import VideoPipeline

    pipeline = VideoPipeline(
        video_path=str(video),
        lenses=lens,
        output_dir=str(output),
        show_preview=preview,
        save_video=not no_video,
        save_json=not no_json,
        frame_skip=frame_skip,
        lens_configs={l: {"device": device} for l in lens},
    )
    pipeline.run()


@app.command()
def anonymize(
    video: Path = typer.Argument(..., help="Path to input video"),
    mode: AnonymizeMode = typer.Option(AnonymizeMode.blur, "--mode", "-m", help="Anonymization mode"),
    output: Path = typer.Option(Path("output"), "--output", "-o"),
    blur_strength: int = typer.Option(51, help="Blur kernel size (odd number, blur mode only)"),
):
    """
    Detect and anonymize all faces in a video (GDPR-friendly).

    Examples:\n
      vidlens anonymize footage.mp4 --mode blur\n
      vidlens anonymize footage.mp4 --mode pixelate\n
      vidlens anonymize footage.mp4 --mode black
    """
    _check_video(video)
    typer.echo(f"🙈 Anonymizing faces with mode: {mode.value}")

    from vidlens.lenses.face_detection import FaceLens
    from vidlens.pipeline import VideoPipeline

    face_lens = FaceLens(anonymize=mode.value, blur_strength=blur_strength)
    pipeline = VideoPipeline(
        video_path=str(video),
        lenses=[face_lens],
        output_dir=str(output),
        save_json=False,
    )
    pipeline.run()


@app.command()
def highlights(
    video: Path = typer.Argument(..., help="Path to input video"),
    duration: int = typer.Option(60, "--duration", "-d", help="Target highlight reel duration in seconds"),
    output: Path = typer.Option(Path("output"), "--output", "-o"),
    labels: Optional[str] = typer.Option(None, help="Comma-separated CLIP labels to score (e.g. 'goal,tackle,celebration')"),
):
    """
    Generate a highlight reel using CLIP similarity scoring.

    Examples:\n
      vidlens highlights match.mp4 --duration 60\n
      vidlens highlights match.mp4 --labels "goal,tackle,celebration" --duration 90
    """
    _check_video(video)
    typer.echo(f"⚡ Generating {duration}s highlight reel...")

    label_list = [l.strip() for l in labels.split(",")] if labels else None

    from vidlens.lenses.scene_classification import SceneClassificationLens
    from vidlens.utils.highlights import extract_highlights

    scene_lens = SceneClassificationLens(labels=label_list, sample_every_n_frames=5)
    extract_highlights(
        video_path=str(video),
        lens=scene_lens,
        target_duration=duration,
        output_dir=str(output),
    )


@app.command()
def train(
    lens_name: str = typer.Argument(..., help="Lens to train: objects | faces | pose | scene"),
    data: Path = typer.Option(..., "--data", help="Path to dataset config YAML"),
    epochs: int = typer.Option(50, "--epochs", "-e"),
    batch: int = typer.Option(16, "--batch", "-b"),
    output_dir: Path = typer.Option(Path("models/custom"), "--output-dir"),
    device: str = typer.Option("auto", "--device"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Base weights to fine-tune from"),
):
    """
    Fine-tune a lens on custom data.

    Examples:\n
      vidlens train objects --data data/my_dataset.yaml --epochs 100\n
      vidlens train faces --data data/faces.yaml --epochs 50 --device cuda\n
      vidlens train scene --data data/captions.csv --epochs 10
    """
    from vidlens.lenses import LENS_REGISTRY

    if lens_name not in LENS_REGISTRY:
        typer.echo(f"❌ Unknown lens '{lens_name}'. Available: {list(LENS_REGISTRY.keys())}")
        raise typer.Exit(1)

    typer.echo(f"\n🏋️  Training lens: {lens_name}")
    typer.echo(f"   Data    : {data}")
    typer.echo(f"   Epochs  : {epochs}")
    typer.echo(f"   Device  : {device}\n")

    lens = LENS_REGISTRY[lens_name](
        model_path=str(model_path) if model_path else None,
        device=device,
    )
    lens.train(
        data_config=str(data),
        epochs=epochs,
        batch=batch,
        output_dir=str(output_dir / lens_name),
    )


@app.command()
def lenses():
    """List all available lenses and their descriptions."""
    from vidlens.lenses import LENS_REGISTRY
    typer.echo("\n🔭 Available VidLens lenses:\n")
    for name, cls in LENS_REGISTRY.items():
        typer.echo(f"  {name:<12} — {cls.description}")
    typer.echo("\nUse with: vidlens analyze video.mp4 --lens <name>")


@app.command()
def ui():
    """Launch the Gradio web UI."""
    try:
        from vidlens.ui.app import launch
        launch()
    except ImportError:
        typer.echo("Install Gradio: pip install gradio")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _check_video(path: Path):
    if not path.exists():
        typer.echo(f"❌ Video file not found: {path}")
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()
