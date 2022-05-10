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
import tensorflow as tf2
import dopamine.discrete_domains.inspect_action_values as iav
import matplotlib.pyplot as plt

import numpy as np


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
  
  tf = tf2.compat.v1
  tf.disable_v2_behavior()

  base_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = run_experiment.create_runner(base_dir)

  INSPECT_ACTION_VALUATIONS = False

  if INSPECT_ACTION_VALUATIONS:
    # name = input('(Name of saliency subdirectory?) ')
    # iav.inspect_action_valuations(runner, name)
    map = iav.create_object_saliency_map(
      runner,
      'ms-pacman',
      [
        'pellet-%d' % (pel_idx) for pel_idx in range(215)
      ] +
      [
        '%s-ghost-disappear' % (gh,) for gh in
        (
          'bottom-right',
          'middle-left',
          'middle-right',
          'top-left'
        )
      ] +
      [
        'ms-pacman-%sdisappear' % (add,) for add in
        (
          '',
          'first-life-',
          'second-life-'
        )
      ]
    )
    
    _, ax = plt.subplots()
    shw = ax.imshow(map, vmin=-0.05, vmax=0.05, cmap='BrBG')
    bar = plt.colorbar(shw)

    plt.xlabel('Resized screen column (pixels)')
    plt.ylabel('Resized screen row (pixels)')
    bar.set_label('Q-value difference')

    plt.tight_layout()
    plt.savefig(iav.SALIENCY_PATH + 'ms-pacman/obj-sal-map.pdf')
  else:
    # runner.run_experiment()
    # out = runner._sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    # print(f'Length of `out`: {len(out)}')
    runner._environment.restore_morel_model()
    runner._environment.reset()
    for index in range(500):
      obs, _, _, _ = runner._environment.step(runner._environment.action_space.sample())
      if index % 10 == 0 and index > 280:
        _, ax = plt.subplots(2, 4)
        ax[0, 0].imshow(obs[..., 0], cmap='gist_gray')
        ax[0, 1].axis('off')
        ax[0, 2].axis('off')
        ax[0, 3].axis('off')
        ax[1, 0].imshow(obs[..., 1], cmap='gist_gray')
        ax[1, 1].imshow(obs[..., 2], cmap='gist_gray')
        ax[1, 2].imshow(obs[..., 3], cmap='gist_gray')
        ax[1, 3].imshow(obs[..., 4], cmap='gist_gray')
        plt.show()
    
    # print('Reset. Starting a new iteration.')
    # runner._run_one_iteration(0)
    # print('Done with iteration.')


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
