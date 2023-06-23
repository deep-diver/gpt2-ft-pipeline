from typing import Text, Any, Dict, Optional

import gradio as gr
import tensorflow as tf
import tensorflow_text
from tensorflow.python.saved_model import tag_constants
from huggingface_hub import Repository

local_path = "hf_model"

model_version = "$MODEL_VERSION"
model_repo_id = "$MODEL_REPO_ID"
model_repo_url = f"https://huggingface.co/{model_repo_id}"

def _clone_and_checkout(repo_url: str, local_path: str, version: str) -> Repository:
    repository = Repository(
        local_dir=local_path, clone_from=repo_url
    )
    repository.git_checkout(revision=version)
    return repository

_ = _clone_and_checkout(model_repo_url, local_path, model_version)
model = tf.saved_model.load(local_path, tags=[tag_constants.SERVING])
gpt_lm_predict_fn = model.signatures["serving_default"]

def gen_text(prompt, max_length=256):
    prompt = tf.constant(f"### Instruction:\n{prompt}\n\n### Response:\n")
    max_length = tf.constant(max_length, dtype="int64")
    
    result = gpt_lm_predict_fn(
        prompt=prompt,
        max_length=max_length,
    )

    return result['result'].numpy().decode('UTF-8').split("### Response:")[-1].strip()

with gr.Blocks() as demo:
    instruction = gr.Textbox("Instruction")
    output = gr.Textbox("Output", lines=5)

    instruction.submit(
        lambda prompt: gen_text(prompt),
        instruction, output
    )

demo.launch()