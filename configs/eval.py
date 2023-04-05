# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Config file for evaluation experiment.

The code in this file is adapted from the BYOL code
(https://github.com/deepmind/deepmind-research/tree/master/byol).
"""

from typing import Text

from semppl.utils import dataset


def get_config(checkpoint_to_evaluate: Text, batch_size: int, num_epochs: int):
  """Return config object for training."""
  train_images_per_epoch = dataset.Split.TRAIN_AND_VALID.num_examples

  config = dict(
      random_seed=0,
      enable_double_transpose=True,
      max_steps=num_epochs * train_images_per_epoch // batch_size,
      num_classes=1000,
      batch_size=batch_size,
      checkpoint_to_evaluate=checkpoint_to_evaluate,
      # If True, allows training without loading a checkpoint.
      allow_train_from_scratch=False,
      # Whether the backbone should be frozen (linear evaluation) or
      # trainable (fine-tuning).
      freeze_backbone=True,
      optimizer_config=dict(
          momentum=0.9,
          nesterov=True,
      ),
      lr_schedule_config=dict(
          base_learning_rate=0.3,
          warmup_steps=0,
      ),
      network_config=dict(  # Should match the evaluated checkpoint
          encoder_class='ResNet50',  # Should match a class in utils/networks.
          encoder_config=dict(resnet_v2=False, width_multiplier=1),
          bn_decay_rate=0.9,
      ),
      evaluation_config=dict(
          subset='test',
          batch_size=100,
      ),
      checkpointing_config=dict(
          use_checkpointing=True,
          checkpoint_dir='/tmp/semppl',
          save_checkpoint_interval=300,
          filename='linear-eval.pkl'),
  )

  return config
