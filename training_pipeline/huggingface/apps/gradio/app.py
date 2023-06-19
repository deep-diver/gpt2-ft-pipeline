import gradio as gr
import numpy as np
import tensorflow as tf

MODEL = None 

def generate(user_input):
    generated_txt = ""

    return generated_txt

with gr.Blocks() as demo:
    gr.Markdown("## TFX auto-generated demo app for text-to-text generation")

    input_txt = gr.Textbox()
    output_txt = gr.Textbox()
    gen_btn = gr.Button("Generate")

    gen_btn.click(
        generate,
        input_txt,
        output_txt
    )

demo.launch(debug=True)