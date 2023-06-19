import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

from .train_data import input_fn
from .signatures import model_exporter
from .model import get_gpt2_model
from .hyperparams import TRAIN_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH
from .hyperparams import EPOCHS

from .utils import INFO

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=TRAIN_BATCH_SIZE,
    )

    model = get_gpt2_model(train_dataset.cardinality() * EPOCHS)
    model.fit(
        train_dataset, 
        steps_per_epoch=TRAIN_LENGTH // TRAIN_BATCH_SIZE,
        epochs=EPOCHS
    )
    
    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={
            "serving_default": model_exporter(model)
        },
    )
