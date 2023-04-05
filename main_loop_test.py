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

"""Tests for SemPPL's main training loop.

The code in this file is adapted from the BYOL code
(https://github.com/deepmind/deepmind-research/tree/master/byol).
"""

from absl import flags
from absl.testing import absltest
import tensorflow_datasets as tfds

from semppl import eval_experiment
from semppl import main_loop
from semppl.configs import eval as eval_config

FLAGS = flags.FLAGS


class MainLoopTest(absltest.TestCase):

  def test_linear_eval(self):
    config = eval_config.get_config(
        checkpoint_to_evaluate=None, batch_size=4, num_epochs=10)
    temp_dir = self.create_tempdir().full_path

    # Override some config fields to make test lighter.
    config['network_config']['encoder_class'] = 'TinyResNet'
    config['allow_train_from_scratch'] = True
    config['checkpointing_config']['checkpoint_dir'] = temp_dir
    config['evaluation_config']['batch_size'] = 16
    config['max_steps'] = 16

    with tfds.testing.mock_data(num_examples=64):
      experiment_class = eval_experiment.EvalExperiment
      main_loop.train_loop(experiment_class, config)
      main_loop.eval_loop(experiment_class, config)


if __name__ == '__main__':
  absltest.main()
