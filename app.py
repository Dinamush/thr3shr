import gradio as gr


INFO = """
# THR3SHR

Legacy Hugging Face Space scaffold for this project.

- The core ONNX models are large and are managed separately (see ATTRIBUTION.md).
- Local tooling and migration workflows live in the FastAPI + React app.

**© 2026 Dinamush** — software MIT; creative materials CC BY 4.0.
Third-party taggers: SmilingWolf (WD), deepghs (ML-Danbooru ONNX / imgutils), and others listed in ATTRIBUTION.md.

Use this Space as a placeholder host while backend/image-pipeline features are added.
"""


with gr.Blocks(title="THR3SHR") as demo:
    gr.Markdown(INFO)


if __name__ == "__main__":
    demo.launch()
