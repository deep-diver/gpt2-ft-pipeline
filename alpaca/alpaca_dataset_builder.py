"""alpaca dataset."""

import json
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for alpaca dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'instruction': tfds.features.Text(),
            'input': tfds.features.Text(),
            'output': tfds.features.Text()
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('instruction', 'output'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(alpaca): Downloads the data and defines the splits
    path = dl_manager.download('https://github.com/tloen/alpaca-lora/raw/main/alpaca_data.json')

    f = open(path)
    j = json.load(f)

    # TODO(alpaca): Returns the Dict[split names, Iterator[Key, Example]]
    train_num = int(len(j) * 0.9)

    return {
        'train': self._generate_examples(j[:train_num]),
        'val': self._generate_examples(j[train_num:]),
    }

  def _generate_examples(self, json_obj):
    """Yields examples."""
    # TODO(alpaca): Yields (key, example) tuples from the dataset

    for idx, item in enumerate(json_obj):
      yield idx, {
        'instruction': item['instruction'],
        'input': item['input'],
        'output': item['output']
      }