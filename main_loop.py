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

"""Training and evaluation loops for an experiment.

The code in this file is adapted from the BYOL code
(https://github.com/deepmind/deepmind-research/tree/master/byol).
"""

import time
from typing import Any, Mapping, Text, Type

from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np

from semppl import eval_experiment
from semppl.configs import eval as eval_config

_WORKER_MODE = flags.DEFINE_string('worker_mode', 'train',
                                   'The mode, train or eval')
_WORKER_TPU_DRIVER = flags.DEFINE_string('worker_tpu_driver', '',
                                         'The tpu driver to use')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1024, 'Total batch size')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 100,
                                   'Number of training epochs for evaluation.')
_CHECKPOINT_ROOT = flags.DEFINE_string('checkpoint_root', '',
                                       'The directory to save checkpoints to.')
_LOG_TENSORS_INTERVAL = flags.DEFINE_integer('log_tensors_interval', 60,
                                             'Log tensors every n seconds.')

FLAGS = flags.FLAGS

ExperimentType = Type[eval_experiment.EvalExperiment]


def train_loop(experiment_class: ExperimentType, config: Mapping[Text, Any]):
  """The main training loop.

  This loop periodically saves a checkpoint to be evaluated in the eval_loop.

  Args:
    experiment_class: the constructor for the experiment.
    config: the experiment config.
  """
  experiment = experiment_class(**config)
  rng = jax.random.PRNGKey(0)
  step = 0

  host_id = jax.host_id()
  last_logging = time.time()
  if config['checkpointing_config']['use_checkpointing']:
    checkpoint_data = experiment.load_checkpoint()
    if checkpoint_data is None:
      step = 0
    else:
      step, rng = checkpoint_data

  local_device_count = jax.local_device_count()
  while step < config['max_steps']:
    step_rng, rng = tuple(jax.random.split(rng))
    # Broadcast the random seeds across the devices
    step_rng_device = jax.random.split(step_rng, num=jax.device_count())
    first_local_device_id = host_id * local_device_count
    step_rng_device = step_rng_device[first_local_device_id:(
        first_local_device_id + local_device_count)]
    step_device = np.broadcast_to(step, [local_device_count])

    # Perform a training step and get scalars to log.
    scalars = experiment.step(global_step=step_device, rng=step_rng_device)

    # Checkpointing and logging.
    if config['checkpointing_config']['use_checkpointing']:
      experiment.save_checkpoint(step, rng)
      current_time = time.time()
      if current_time - last_logging > _LOG_TENSORS_INTERVAL.value:
        logging.info('Step %d: %s', step, scalars)
        last_logging = current_time
    step += 1
  logging.info('Saving final checkpoint')
  logging.info('Step %d: %s', step, scalars)
  experiment.save_checkpoint(step, rng)


def eval_loop(experiment_class: ExperimentType, config: Mapping[Text, Any]):
  """The main evaluation loop.

  This loop periodically loads a checkpoint and evaluates its performance on the
  test set, by calling experiment.evaluate.

  Args:
    experiment_class: the constructor for the experiment.
    config: the experiment config.
  """
  experiment = experiment_class(**config)
  last_evaluated_step = -1
  while True:
    checkpoint_data = experiment.load_checkpoint()
    if checkpoint_data is None:
      logging.info('No checkpoint found. Waiting for 10s.')
      time.sleep(10)
      continue
    step, _ = checkpoint_data
    if step <= last_evaluated_step:
      logging.info('Checkpoint at step %d already evaluated, waiting.', step)
      time.sleep(10)
      continue
    host_id = jax.host_id()
    local_device_count = jax.local_device_count()
    step_device = np.broadcast_to(step, [local_device_count])
    scalars = experiment.evaluate(global_step=step_device)
    if host_id == 0:  # Only perform logging in one host.
      logging.info('Evaluation at step %d: %s', step, scalars)
    last_evaluated_step = step
    if last_evaluated_step >= config['max_steps']:
      return


def main(_):
  if _WORKER_TPU_DRIVER.value:
    jax.config.update('jax_xla_backend', 'tpu_driver')
    jax.config.update('jax_backend_target', _WORKER_TPU_DRIVER.value)
    logging.info('Backend: %s %r', _WORKER_TPU_DRIVER.value, jax.devices())

  experiment_class = eval_experiment.EvalExperiment
  config = eval_config.get_config(f'{_CHECKPOINT_ROOT.value}/pretrain.pkl',
                                  _BATCH_SIZE.value, _NUM_EPOCHS.value)
  config['checkpointing_config']['checkpoint_dir'] = _CHECKPOINT_ROOT.value  # pytype: disable=unsupported-operands  # dict-kwargs

  if _WORKER_MODE.value == 'train':
    train_loop(experiment_class, config)
  elif _WORKER_MODE.value == 'eval':
    eval_loop(experiment_class, config)


if __name__ == '__main__':
  app.run(main)
