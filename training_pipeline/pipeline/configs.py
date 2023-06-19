import os
import string
import random
import tensorflow_model_analysis as tfma
import tfx.extensions.google_cloud_ai_platform.constants as vertex_const
import tfx.extensions.google_cloud_ai_platform.trainer.executor as vertex_training_const
import tfx.extensions.google_cloud_ai_platform.tuner.executor as vertex_tuner_const

PIPELINE_NAME = "tfx-vit-pipeline"

try:
    import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"
except ImportError:
    GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"

GOOGLE_CLOUD_REGION = "us-central1"

GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + "-complete-mlops"
PIPELINE_IMAGE = f"gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}"

OUTPUT_DIR = os.path.join("gs://", GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", PIPELINE_NAME)

DATA_PATH = "gs://alpaca/tfrecords/"

TRAINING_FN = "modules.train.run_fn"
TUNER_FN = "modules.tuning.tuner_fn"
PREPROCESSING_FN = "modules.preprocessing.preprocessing_fn"

EXAMPLE_GEN_BEAM_ARGS = None
TRANSFORM_BEAM_ARGS = None

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

WANDB_RUN_ID = f"full-training-{id_generator()}"

WANDB_CONFIGS = {
    "API_KEY": "$WANDB_ACCESS_TOKEN",
    "PROJECT": PIPELINE_NAME,
    "FINAL_RUN_ID": WANDB_RUN_ID
}

HYPER_PARAMETERS = {
    "finetune_epochs": {
        "type": "choice",
        "values": [10]
    },

    "fulltrain_epochs": {
        "type": "choice",
        "values": [30]
    },    

    "optimizer_type": {
        "type": "choice",
        "values": ["Adam", "AdamW"],
    },

    "learning_rate": {
        "type": "float",
        "min_value": 0.00001,
        "max_value": 0.1,
        "sampling": "log",
        "step": 10
    },

    "weight_decay": {
        "type": "choice",
        "values": [0.0, 0.1, 0.2, 0.3, 0.5, 0.6]
    }
}

TUNER_CONFIGS = {
    "num_trials": 15
}

EVAL_CONFIGS = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            signature_name="from_examples",
            preprocessing_function_names=["transform_features"],
            label_key="labels",
            prediction_key="labels",
        )
    ],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(
                    class_name="SparseCategoricalAccuracy",
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.55}
                        ),
                        # Change threshold will be ignored if there is no
                        # baseline model resolved from MLMD (first run).
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": -1e-3},
                        ),
                    ),
                )
            ]
        )
    ],
)

GCP_AI_PLATFORM_TRAINING_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_training_const.TRAINING_ARGS_KEY: {
        "project": GOOGLE_CLOUD_PROJECT,
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_K80",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": PIPELINE_IMAGE,
                },
            }
        ],
    },
    "use_gpu": True,
    "wandb": WANDB_CONFIGS
}

NUM_PARALLEL_TRIALS = 3
GCP_AI_PLATFORM_TUNER_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_tuner_const.TUNING_ARGS_KEY: {
        "project": GOOGLE_CLOUD_PROJECT,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-8",
                        "accelerator_type": "NVIDIA_TESLA_V100",
                        "accelerator_count": 1,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": PIPELINE_IMAGE,
                    },
                }
            ],
        },
    },
    vertex_tuner_const.REMOTE_TRIALS_WORKING_DIR_KEY: os.path.join(
        PIPELINE_ROOT, "trials"
    ),
    "use_gpu": True,
    "hyperparameters": HYPER_PARAMETERS,
    "tuner": TUNER_CONFIGS,
    "wandb": WANDB_CONFIGS
}

GCP_AI_PLATFORM_SERVING_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_const.VERTEX_CONTAINER_IMAGE_URI_KEY: "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
    vertex_const.SERVING_ARGS_KEY: {
        "project_id": GOOGLE_CLOUD_PROJECT,
        "deployed_model_display_name": PIPELINE_NAME.replace("-", "_"),
        "endpoint_name": "prediction-" + PIPELINE_NAME.replace("-", "_"),
        "traffic_split": {"0": 100},
        "machine_type": "n1-standard-4",
        "min_replica_count": 1,
        "max_replica_count": 1,
    },
}

GRADIO_APP_PATH = "huggingface.apps.gradio"

HF_PUSHER_ARGS = {
    "username": "chansung",
    "access_token": "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",
    "repo_name": PIPELINE_NAME,
    "space_config": {
        "app_path": GRADIO_APP_PATH,
    },
}