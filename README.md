# GPT2Alpaca Pipeline

This project demonstrates how to build a machine learning pipeline for fine-tuning GPT2 on Alpaca dataset with the technologies of [TensorFlow Extended(TFX)](https://www.tensorflow.org/tfx), [KerasNLP](https://keras.io/keras_nlp), [TensorFlow](https://www.tensorflow.org), and [Hugging Face Hub](https://huggingface.co/docs/hub/index). This project is done as a part of ***2023 Keras Community Sprint*** held by the official Keras team at Google.

## Introduction

The demand on building ChatGPT like Large Language Model(LLM)s has been dramatically increasing since early 2023 because of their promising capabilities. In order to build a customized and private LLM based Chatbot applications, we need to fine-tune a language model(i.e. GPT2) on (instruction, response) paried custom dataset. 

![](https://pbs.twimg.com/media/FzX8X1BaAAIP9Pm?format=jpg&name=4096x4096)

This project uses [GPT2](https://en.wikipedia.org/wiki/GPT-2) model from [KerasNLP](https://keras.io/keras_nlp) library as the base language model and fine-tune the GPT2 on [Stanford Alpaca dataset](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json) from [alpaca-lora](https://github.com/tloen/alpaca-lora) repository.

***NOTE:*** *The Alpaca dataset used in this project is the enhanced version of the [original Standford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) by open source communities to fix some flaws manually and with GPT4 API.*

Further, in order to automate fine-tuning process, this project embedded the fine-tuning process in and end to end machine learning pipeline built in [TensorFlow Extended(TFX)](https://www.tensorflow.org/tfx). Within the pipline, when the data is given, the following TFX components are sequentially triggered, and the data in between components is shared in [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format.

1. Alpaca dataset is injected into the TFX pipeline through [TFX ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) component. It is assumed that the data is prepared as TFRecord format beforehand. [TensorFlow Dataset](https://www.tensorflow.org/datasets) allows us to create TFRecords easily without knowing much about TFRecords. If you are curious, check out the [alpaca](https://github.com/deep-diver/gpt2-ft-pipeline/tree/main/alpaca) sub directory to find about how-to.

2. Injected data is transformed into instruction-following format through [TFX Transform](https://www.tensorflow.org/tfx/guide/transform) component. The original Alpaca dataset separately stores `instruction`, `input`, and `response` for each conversation. However, they should be merged into a single string in the following format:

- ```python
  f"""### Instruction:
  {instruction_txt}
  
  ### Input:
  {input_txt}
  
  ### Response:
  {response_txt}
  """
  ```

3. Fine-tuning process begins with the transformed data through [TFX Trainer](https://www.tensorflow.org/tfx/guide/trainer) component. It instantiates GPT2 [tokenizer](), [preprocessor](), and [model](), then it fine-tunes GPT2 model on the transformed data. The final fine-tuned model is exported as [`SavedModel`](https://www.tensorflow.org/guide/saved_model) with [custom a signature](https://github.com/deep-diver/gpt2-ft-pipeline/blob/main/training_pipeline/modules/signatures.py)(this is a minimum requirement to serve TensorFlow/Keras model within [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)).

- ***NOTE:*** *There are two paths from this point to deploy fine-tuned GPT2 model. First option is the deployment on GCP's [Vertex AI](https://cloud.google.com/vertex-ai) platform, and the second option is the deployment on [Hugging Face 🤗 Hub](https://huggingface.co/docs/hub/index). In this document, the latter one is explained because the official TFX docker image currently does not support some operations in KerasNLP's GPT2 model.*

4. Fine-tuned model is pushed to the Hugging Face 🤗 Model Hub through custom [TFX HFPusher](https://github.com/deep-diver/gpt2-ft-pipeline/tree/main/training_pipeline/pipeline/components/HFPusher) component. At each time the model is pushed, new revision name(based on date) is assigned to it to distinguish the version of the model.

5. With the additonal capability of the custom [TFX HFPusher](https://github.com/deep-diver/gpt2-ft-pipeline/tree/main/training_pipeline/pipeline/components/HFPusher) component, it publishes a prepared template application to Hugging Face 🤗 Space Hub. At each time the model is pushed, some strings within the template is replaced by real values at runtime such as `revision name`.

## Instruction

Currently, Vertex AI is not supported to run this pipeline due to the CUDA and cuDNN version conflicts between TFX and KerasNLP. However, you can simply run the whole pipeline in a local and colab environment as below. 

### Requirements

- Be sure to have ***GPU(s) in both cases***. I have tested fine-tuning process with a single 80G A100 instance, and it took about an hour to finish the whole pipeline.
- Also, be sure to have `CUDA >= 11.6` and `cuDNN >= 8.6`. Below these versions, some KerasNLP GPT2 model would fail. As of ***07/28/2023***, the default Colab environment comes with higher versions of the two frameworks.

### Local environment

1. Install dependencies

    ```shell
    # it is recommended to run the following pip command in venv
    
    $ cd training_pipeline
    $ pip install -r requirements.txt
    ```

2. Replace Hugging Face Token inside `pipeline/configs.py` with the environment variable. This token will be used to push the model and publish a space application on Hugging Face Hub. If you are not familiar with how to get Hugging Face Access Token, check out the [official document](https://huggingface.co/docs/hub/security-tokens) about it.

    ```shell
    $ HF_ACCESS_TOKEN="YOUR Hugging Face Access Token"
    $ envsubst '$HF_ACCESS_TOKEN' < pipeline/configs.py \
                                  > pipeline/configs.py
    ```

3. Create TFX pipeline with `tfx pipeline create` command. This command registers a TFX pipeline system wide. After the creation, if you modify something in the pipeline perspective, you need to run `tfx pipeline update` instead of `create`. In this case, the options and their values remain the same. Any modifications of the files inside [`modules`](https://github.com/deep-diver/gpt2-ft-pipeline/tree/main/training_pipeline/modules) directory does not require to run `tfx pipeline update`.

    ```shell
    $ tfx pipeline create --pipeline-path local_runner.py \
                          --engine local
    ```

4. Once TFX pipeline is created(registered) successfully, you can run the pipeline with `tfx run create` command. It will go through each component sequentially, and any intermediate products will be stored under the current directory. 

    ```shell
    $ tfx run create --pipeline-name kerasnlp-gpt2-alpaca-pipeline \
                     --engine local
    ```

### Colab environment (TBD)

## Todo
- [X] Notebook to convert GPT2CausalLM into `SavedModel` format
- [X] Notebook to fine-tune GPT2CausalLM with Alpaca dataset
- [X] Notebook to build a minimal ML Pipline with TFX
- [X] Build a standalone TFX pipeline w/ notebook
- [ ] Put the TFX pipeline up on Google Cloud Platform (Vertex AI)
- [ ] Testing out deployed GPT2CausalLM in Vertex AI and Hugging Face Space
- [X] Testing out deployed GPT2CausalLM on Hugging Face Space


