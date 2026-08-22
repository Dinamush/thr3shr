import gradio as gr


INFO = """
# THR3SHR

Legacy Hugging Face Space scaffold for this project.

- The core ONNX models are large and are managed separately.
- Local tooling and migration workflows live in the FastAPI + React app.

Use this Space as a placeholder host while backend/image-pipeline features are added.
"""


with gr.Blocks(title="THR3SHR") as demo:
    gr.Markdown(INFO)


if __name__ == "__main__":
    demo.launch()
