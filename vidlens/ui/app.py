"""
VidLens Gradio Web UI
Run with: vidlens ui  OR  python -m vidlens.ui.app
"""

from __future__ import annotations
import tempfile
import os
from pathlib import Path


def launch(share: bool = False):
    try:
        import gradio as gr
    except ImportError:
        print("Install Gradio: pip install gradio")
        return

    from vidlens.lenses import LENS_REGISTRY
    from vidlens.pipeline import VideoPipeline

    lens_choices = list(LENS_REGISTRY.keys())

    def run_analysis(video_file, selected_lenses, device, frame_skip, anonymize_mode):
        if video_file is None:
            return None, "Please upload a video."
        if not selected_lenses:
            return None, "Please select at least one lens."

        with tempfile.TemporaryDirectory() as tmpdir:
            lens_configs = {l: {"device": device} for l in selected_lenses}

            if "faces" in selected_lenses and anonymize_mode != "none":
                lens_configs["faces"]["anonymize"] = anonymize_mode

            pipeline = VideoPipeline(
                video_path=video_file,
                lenses=selected_lenses,
                output_dir=tmpdir,
                save_json=True,
                save_video=True,
                frame_skip=int(frame_skip),
                lens_configs=lens_configs,
            )
            summary = pipeline.run()

            out_video = summary.get("output_video")
            out_json = summary.get("output_json")

            report_text = ""
            if out_json and Path(out_json).exists():
                import json
                with open(out_json) as f:
                    data = json.load(f)
                total = data.get("total_frames", 0)
                processed = summary.get("processed_frames", 0)
                report_text = (
                    f"✅ Processed {processed}/{total} frames\n"
                    f"⏱  {summary['elapsed_seconds']}s at {summary['avg_fps']} fps\n\n"
                    f"Lenses run: {', '.join(selected_lenses)}\n"
                )

            # Copy annotated video to a path Gradio can serve
            if out_video and Path(out_video).exists():
                dest = Path(tempfile.mktemp(suffix=".mp4"))
                import shutil
                shutil.copy(out_video, dest)
                return str(dest), report_text

        return None, "Processing failed. Check logs."

    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------

    with gr.Blocks(title="VidLens", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 👁️ VidLens — Video Intelligence Toolkit
        Run computer vision analysis on any video using pre-trained deep learning models.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="📹 Upload Video")

                selected_lenses = gr.CheckboxGroup(
                    choices=lens_choices,
                    value=["objects"],
                    label="🔭 Select Lenses",
                    info="Choose which analyses to run",
                )

                anonymize_mode = gr.Dropdown(
                    choices=["none", "blur", "pixelate", "black"],
                    value="none",
                    label="😶 Face Anonymization (if 'faces' lens selected)",
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    device = gr.Dropdown(
                        choices=["auto", "cpu", "cuda", "mps"],
                        value="auto",
                        label="Device",
                    )
                    frame_skip = gr.Slider(
                        minimum=0, maximum=10, step=1, value=0,
                        label="Frame Skip (0 = process all frames)",
                    )

                run_btn = gr.Button("▶ Run Analysis", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_output = gr.Video(label="📤 Annotated Output")
                report_output = gr.Textbox(label="📊 Report", lines=8)

        run_btn.click(
            fn=run_analysis,
            inputs=[video_input, selected_lenses, device, frame_skip, anonymize_mode],
            outputs=[video_output, report_output],
        )

        gr.Markdown("""
        ---
        **VidLens** is open source. Contribute at [github.com/your-org/vidlens](https://github.com)
        """)

    print("🚀 Launching VidLens UI at http://localhost:7860")
    demo.launch(share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    launch()
