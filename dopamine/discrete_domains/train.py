# coding=utf-8
# Lint as: python3
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The entry point for running a Dopamine agent.

"""

from absl import app
from absl import flags
from absl import logging

from dopamine.discrete_domains import run_experiment
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from dopamine.discrete_domains.run_experiment import Difference


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')


FLAGS = flags.FLAGS
    


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_v2_behavior()

  base_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = run_experiment.create_runner(base_dir)

  INSPECT_ACTION_VALUATIONS = True

  if INSPECT_ACTION_VALUATIONS:
    obs = runner.observations_sequence([2, 5] * 15)

    last_four = np.concatenate((
      obs[-1][0],
      obs[-2][0],
      obs[-3][0],
      obs[-4][0]
    ), axis=-1)[np.newaxis, ...]

    obj_to_change = 1  # layer index `1`
    manip = runner.manipulate_object(last_four, obj_to_change, Difference(-5, 0, True))
    
    plt.subplot(2, 2, 1)
    plt.imshow(last_four[0, ..., :1], cmap='binary')
    plt.subplot(2, 2, 2)
    plt.imshow(manip[0, ..., :1], cmap='binary')
    plt.subplot(2, 2, 3)
    plt.imshow(last_four[0, ..., obj_to_change], cmap='binary')
    plt.subplot(2, 2, 4)
    plt.imshow(manip[0, ..., obj_to_change], cmap='binary')
    plt.tight_layout()
    plt.show(block=True)

    # evals = runner.state_action_evaluations(last_four)
    # print(evals)
  else:
    runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
