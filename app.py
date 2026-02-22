import gradio as gr


INFO = """
# ML-Danbooru ONNX Web App

This Space is the deployment scaffold for the project.

- The core ONNX models are large and are managed separately.
- Local tooling and migration workflows are being integrated.

Use this Space as the frontend host while backend/image-pipeline features are added.
"""


with gr.Blocks(title="ML-Danbooru ONNX Webapp") as demo:
    gr.Markdown(INFO)


if __name__ == "__main__":
    demo.launch()
