from typing import List

import tensorflow as tf

from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import DataAccessor

def input_fn(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = False,
    batch_size: int = 32,
) -> tf.data.Dataset:
    INFO(f"Reading data from: {file_pattern}")

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size
        ),
        tf_transform_output.transformed_metadata.schema,
    ).map(
        lambda x: x['combine']
    ).prefetch(
        tf.data.AUTOTUNE
    )

    return dataset
