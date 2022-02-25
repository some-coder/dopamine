# coding=utf-8
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
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.

## Networks
We are subclassing keras.models.Model in our network definitions. Each network
class has two main functions: `.__init__` and `.call`. When we create our
network the `__init__` function is called and necessary layers are defined. Once
we create our network, we can create the output operations by doing `call`s to
our network with different inputs. At each call, the same parameters will be
used.

More information about keras.Model API can be found here:
https://www.tensorflow.org/api_docs/python/tf/keras/models/Model

## Network Types
Network types are namedtuples that define the output signature of the networks
used. Please use the appropriate signature as needed when defining new networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from absl import logging

import cv2
import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

from typing import Dict, Tuple, Optional, Union
import os
import re
from PIL import Image
from enum import Enum
import matplotlib.pyplot as plt



class DQNScreenMode(Enum):
  """
  A way of presenting the screen as input.
  """
  OFF = 'off'
  GRAYSCALE = 'grayscale'
  RGB = 'rgb'


# The way of representing the screen in the DQN input.
DQN_SCREEN_MODE = DQNScreenMode.OFF

# Whether to introduce object channels. Number of channels may vary per game.
# If `DQNScreenMode` is set to `OFF`, then `DQN_USE_OBJECTS` is automatically
# `True`.
DQN_USE_OBJECTS = True
if DQN_SCREEN_MODE == DQN_SCREEN_MODE.OFF:
  DQN_USE_OBJECTS = True  # for if it were `False`, we would have nothing

# the location on disk of the object images
if re.search('(s3366235)', os.environ['HOME']):
  DQN_OBJECTS_LOC = os.path.join('/', 'data', 's3366235', 'master-thesis', 'objects')
else:
  DQN_OBJECTS_LOC = os.path.join(os.environ['HOME'], 'Documents', 'test', 'objects')

# The number of object channels to use, *if* we use objects. Should minimally
# be the number of objects for your game.
DQN_NUM_OBJ = 4
if DQN_SCREEN_MODE == DQNScreenMode.OFF:
  DQN_NUM_OBJ += 1  # for walls and floors

# The number of channels dedicated to the raw screen footage.
# May be zero in the case that `DQN_SCREEN_MODE == .OFF`.
if DQN_SCREEN_MODE == DQNScreenMode.OFF:
  DQN_SCREEN_LAY = 0
elif DQN_SCREEN_MODE == DQNScreenMode.GRAYSCALE:
  DQN_SCREEN_LAY = 1
elif DQN_SCREEN_MODE == DQNScreenMode.RGB:
  DQN_SCREEN_LAY = 3

# The number of channels dedicated to the objects.
# May be zero in the case that `DQN_USE_OBJECTS == False`.
DQN_OBJ_LAY = DQN_NUM_OBJ if DQN_USE_OBJECTS else 0

NATURE_DQN_OBSERVATION_SHAPE = (
  84,
  84,
  DQN_SCREEN_LAY + DQN_OBJ_LAY
)
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = collections.namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])
ImplicitQuantileNetworkType = collections.namedtuple(
    'iqn_network', ['quantile_values', 'quantiles'])



@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True):
  """Wraps an Atari 2600 Gym environment with some basic preprocessing.

  This preprocessing matches the guidelines proposed in Machado et al. (2017),
  "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems for General Agents".

  The created environment is the Gym wrapper around the Arcade Learning
  Environment.

  The main choice available to the user is whether to use sticky actions or not.
  Sticky actions, as prescribed by Machado et al., cause actions to persist
  with some probability (0.25) when a new command is sent to the ALE. This
  can be viewed as introducing a mild form of stochasticity in the environment.
  We use them by default.

  Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.

  Returns:
    An Atari 2600 environment with some standard preprocessing.
  """
  assert game_name is not None
  atari_check_num_objects(game_name)
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
  # handle this time limit internally instead, which lets us cap at 108k frames
  # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
  # restoring states.
  env = env.env
  env = AtariPreprocessing(
    env,
    objects=atari_objects_map(game_name),
    bg_color=atari_background_color(game_name))
  return env


@gin.configurable(denylist=['variables'])
def maybe_transform_variable_names(variables, legacy_checkpoint_load=False):
  """Maps old variable names to the new ones.

  The resulting dictionary can be passed to the tf.compat.v1.train.Saver to load
  legacy checkpoints into Keras models.

  Args:
    variables: list, of all variables to be transformed.
    legacy_checkpoint_load: bool, if True the variable names are mapped to
        the legacy names as appeared in `tf.slim` based agents. Use this if
        you want to load checkpoints saved before tf.keras.Model upgrade.
  Returns:
    dict or None, of <new_names, var>.
  """
  logging.info('legacy_checkpoint_load: %s', legacy_checkpoint_load)
  if legacy_checkpoint_load:
    name_map = {}
    for var in variables:
      new_name = var.op.name.replace('bias', 'biases')
      new_name = new_name.replace('kernel', 'weights')
      name_map[new_name] = var
  else:
    name_map = None
  return name_map


def atari_objects_map(game_name: str) -> \
    Optional[Dict[str, Tuple[np.ndarray, Union[float, Tuple[float, ...]], bool]]]:
  """Returns a mapping from objects to three-triples.

  The three-triples consist of the following elements:
  1. A template to match. Only if `DQN_SCREEN_MODE` is `.RGB`, this template
    has 3 channels; it otherwise has only 1.
  2. A threshold or a tuple of two or more thresholds. The matching
    threshold. If multiple thresholds are defined, you are responsible
    for invoking the desired threshold on a use-case basis.
  3. A flag. Indicates whether to use a mask (`True`) or not (`False`).

  Args:
    game_name: The name of the game. Example: `'Pong'`.
  Returns:
    Possibly a mapping.
  """
  out: Optional[Dict[str, Tuple[np.ndarray, Union[float, Tuple[float, ...]], bool]]] = None
  li: Optional[Tuple[Tuple[str, Union[float, Tuple[float, ...]], bool]]] = None

  if game_name == 'Pong':
    li = (
      ('paddle-piece-wide', 0.01, False),
      ('ball-padded', 0.15, False)
    )
    if DQN_SCREEN_MODE == DQNScreenMode.OFF:
      li = li + (('wall', 0.014, False),)
  elif game_name == 'FishingDerby':
    raise ValueError('You need to update the values for FishingDerby!')
    # li = (
    #   ('tackle', 0.8),
    #   ('fish', 0.6),
    #   ('shark', 0.6))
  elif game_name == 'MsPacman':
    li = (
      ('yellow-2', 0.001, False),  # for matching Ms. Pac-Man
      ('blinky-red-padded', 0.25, False),
      ('pellet-padded', 0.1, False),
      ('power-pellet-padded', 0.067, False)
    )
    if DQN_SCREEN_MODE == DQNScreenMode.OFF:
      li = li + (('background', (105e3, 64e3, 119e3, 119e3), True),)
  if li is not None:
    out = {}
    pth = os.path.join(DQN_OBJECTS_LOC, game_name)
    for name, thr, use_mask in li:
      obj_pth = '%s.png' % (name,)
      tmpl = Image.open(os.path.join(pth, obj_pth))
      tmpl = np.array(tmpl if DQN_SCREEN_MODE == DQNScreenMode.RGB else tmpl.convert('L'))
      if use_mask:
        mask_pth = '%s-mask.png' % (name,)
        mask = Image.open(os.path.join(pth, mask_pth))
        mask = np.array(mask)  # is already 2D (without third depth dimension)
        out[name] = (tmpl, thr, mask)
      else:
        out[name] = (tmpl, thr, None)
  return out


def atari_background_color(game_name: str) -> \
    Union[Tuple[np.uint8], Tuple[np.uint8, np.uint8, np.uint8]]:
  """Yields the background color for the specified Atari game.

  Args:
    game_name: The name of the Atari 2600 game to get the background
      color of.
  Returns:
    col: The background color. If `DQN_SCREEN_MODE` is `DQNScreenMode.RGB`,
      then it yields a triple of RGB `uint8`s. If it's `False`, it will
      yield a single value: the grayscale background 'color'.
  """
  if game_name == 'Pong':
    if DQNScreenMode.RGB:
      return (np.uint8(144), np.uint8(72), np.uint8(17))
    else:
      return (np.uint8(87),)
  elif game_name == 'FishingDerby':
    if DQNScreenMode.RGB:
      return (np.uint8(24), np.uint8(26), np.uint8(167))
    else:
      return (np.uint8(41),)
  elif game_name == 'MsPacman':
    if DQNScreenMode.RGB:
      return (np.uint8(0), np.uint8(28), np.uint8(136))
    else:
      return (np.uint8(32),)
  else:
    raise ValueError('Game \'%s\' has no background color info.' % (game_name))


def atari_check_num_objects(game_name: str) -> None:
  """Checks whether `DQN_NUM_OBJ` has the right number of objects for the game.

  Args:
    game_name: The name of the Atari 2600 game to check the number of objects
      for.
  """
  num_obj = DQN_NUM_OBJ - (1 if DQN_SCREEN_MODE == DQNScreenMode.OFF else 0)
  if game_name == 'Pong' and num_obj == 2:
    return
  elif game_name == 'FishingDerby' and num_obj == 3:
    return
  elif game_name == 'MsPacman' and num_obj == 4:
    return
  else:
    raise ValueError(
      'Game \'%s\' doesn\'t have %d objects!' %
      (game_name, DQN_NUM_OBJ))


class NatureDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(NatureDQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)

    return DQNNetworkType(self.dense2(x))


class RainbowNetwork(tf.keras.Model):
  """The convolutional network used to compute agent's return distributions."""

  def __init__(self, num_actions, num_atoms, support, name=None):
    """Creates the layers used calculating return distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      name: str, used to crete scope for network parameters.
    """
    super(RainbowNetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)


class ImplicitQuantileNetwork(tf.keras.Model):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  def __init__(self, num_actions, quantile_embedding_dim, name=None):
    """Creates the layers used calculating quantile values.

    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      name: str, used to create scope for network parameters.
    """
    super(ImplicitQuantileNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.quantile_embedding_dim = quantile_embedding_dim
    # We need the activation function during `call`, therefore set the field.
    self.activation_fn = tf.keras.activations.relu
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state, num_quantiles):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: `tf.Tensor`, contains the agent's current state.
      num_quantiles: int, number of quantile inputs.
    Returns:
      collections.namedtuple, that contains (quantile_values, quantiles).
    """
    batch_size = state.get_shape().as_list()[0]
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    state_vector_length = x.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random.uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
        1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
          state_vector_length, activation=self.activation_fn,
          kernel_initializer=self.kernel_initializer)
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)
    x = self.dense1(x)
    quantile_values = self.dense2(x)
    return ImplicitQuantileNetworkType(quantile_values, quantiles)


@gin.configurable(denylist=['objects'])
class AtariPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84, objects=None, bg_color=None):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.
      objects: dict or None, the mapping from game objects to their templates
        and thresholds, if such a mapping exists for the game.
      bg_color: `Union[Tuple[np.uint8], Tuple[np.uint8, np.uint8, np.uint8]]`
        or `None`. The background color of the game.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    obs_dims = self.environment.observation_space
    # Stores temporary observations used for aggregating over two successive
    # frames.
    self.screen_buffer = [
        np.empty(obs_dims.shape[:2] + (NATURE_DQN_OBSERVATION_SHAPE[2],), dtype=np.uint8),
        np.empty(obs_dims.shape[:2] + (NATURE_DQN_OBSERVATION_SHAPE[2],), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

    self.objects = objects
    self.bg_color = bg_color
    self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    self.game_name = re.match('[a-zA-Z]+', self.environment.unwrapped.spec.id).group().replace('NoFrameskip', '')
    self.is_ms_pacman = self.game_name == 'MsPacman'

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, NATURE_DQN_OBSERVATION_SHAPE[2]),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.lives = self.environment.ale.lives()
    
    if DQN_SCREEN_MODE == DQNScreenMode.RGB:
      self._fetch_rgb_observation(self.screen_buffer[0])
    elif DQN_SCREEN_MODE == DQNScreenMode.GRAYSCALE:
      self._fetch_grayscale_observation(self.screen_buffer[0])
    elif DQN_SCREEN_MODE == DQNScreenMode.OFF:
      self._fetch_screen_off_observation(self.screen_buffer[0])

    self.screen_buffer[1].fill(0)
    if DQN_SCREEN_MODE == DQNScreenMode.RGB:
      return self._color_average_and_resize()
    else:
      # this includes the sitatuation where the screen is off
      return self._pool_and_resize()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    accumulated_reward = 0.

    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # image (grayscale or colour) from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      if is_terminal:
        break
      # We aggregate over the last two frames.
      elif time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        if DQN_SCREEN_MODE == DQNScreenMode.RGB:
          self._fetch_rgb_observation(self.screen_buffer[t])
        elif DQN_SCREEN_MODE == DQNScreenMode.GRAYSCALE:
          self._fetch_grayscale_observation(self.screen_buffer[t])
        elif DQN_SCREEN_MODE == DQNScreenMode.OFF:
          self._fetch_screen_off_observation(self.screen_buffer[t])

    # Aggregate over the last two observations (frames)
    if DQN_SCREEN_MODE == DQNScreenMode.RGB:
      observation = self._color_average_and_resize()
    else:
      # this includes the situation where we have the screen turned off
      observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_objects_observation(
      self,
      obs: np.ndarray) -> Optional[np.ndarray]:
    """Returns `DQN_NUM_OBJ` object layers, if `DQN_USE_OBJECTS` is `True`.

    The returned observation is stored in `obj_obs`. (So, NumPy arrays are
    'pass by reference'.)

    Note that we use `cv2.TM_SQDIFF` instead of `cv2.TM_SQDIFF_NORMED` if
    an object has a non-`None` `mask` entry associated to it: only
    `cv2.TM_CCORR_NORMED` and `cv2.SQDIFF` are defined for use with masks.

    Args:
      obs: The 'normal' game observation from which to derive object layers.
    
    Returns:
      out: The observation layers, or `None` if `DQN_USE_OBJECTS` was
      set to `False`.
    """
    if not DQN_USE_OBJECTS:
      return None  # if we get here, the caller (me) likely made a mistake
    
    keys = list(self.objects.keys())
    obs[..., DQN_SCREEN_LAY:].fill(0)  # Remove all previous contents. Use pass-by-reference.
    
    # stand-in for `obs[..., :DQN_SCREEN_LAY]` in case we only store
    # object channels
    if DQN_SCREEN_MODE == DQNScreenMode.OFF:
      screen = self.environment.ale.getScreenGrayscale()
    else:
      screen = None

    # fill the object channels based on either `obs` or `screen`
    for index in range(DQN_NUM_OBJ):
      special_case = self.is_ms_pacman and index == DQN_NUM_OBJ - 1 and DQN_SCREEN_MODE == DQNScreenMode.OFF
      
      tmpl, thr, mask = self.objects[keys[index]]  # `mask` may be `None`
      sd = cv2.matchTemplate(
        screen if DQN_SCREEN_MODE == DQNScreenMode.OFF else obs[..., :DQN_SCREEN_LAY],
        tmpl,
        cv2.TM_SQDIFF_NORMED if mask is None else cv2.TM_SQDIFF,
        mask=mask)
      if special_case:
        # separate thresholds based on the maze in which Ms. Pac-Man is in 
        if screen[1, 0] == 146:
          locs = np.where(sd < thr[0])
        elif screen[1, 0] == 121:
          locs = np.where(sd < thr[1])
        elif screen[1, 0] == 170:
          locs = np.where(sd < thr[2])
        elif screen[1, 0] == 132:
          locs = np.where(sd < thr[3])
      else:
        locs = np.where(sd < thr)  # minimise the squared difference
      if special_case:
        # special situation: thicken lines of path for Ms. Pac-Man
        obs[locs[0], locs[1], DQN_SCREEN_LAY + index] = 255  # create 'thin' path
        obs[..., DQN_SCREEN_LAY + index] = \
          cv2.dilate(obs[..., DQN_SCREEN_LAY + index], self.dilate_kernel, iterations=1)
      else:
        # just paste the template's silhouette onto matching locations
        for y, x in zip(*locs):
          obs[
            y:(y + tmpl.shape[0]),
            x:(x + tmpl.shape[1]),
            DQN_SCREEN_LAY + index
          ] = 255  # set to maximal contrast
    
    return obs

  def _fetch_rgb_observation(self, output):
    """Returns the current observation in full colour (RGB).

    The returned observation is stored in 'output'.

    Args:
      output: Numpy array. Screen buffer to hold the returned observation.
    
    Returns:
      observation: Numpy array. The current observation in RGB.
    """
    output[:, :, :3] = self.environment.ale.getScreenRGB()
    return self._fetch_objects_observation(output) if DQN_USE_OBJECTS else output

  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    output[:, :, 0] = self.environment.ale.getScreenGrayscale()
    return self._fetch_objects_observation(output) if DQN_USE_OBJECTS else output

  def _fetch_screen_off_observation(self, output):
    """Returns the current observation. Only the object channels are supplied.

    The returned observation is stored in `output`.

    Args:
      output: numpy array, screen buffer to hold the returned observation.
    
    Returns:
      observation: numpy array, the current observation in object channels.
    """
    return self._fetch_objects_observation(output)

  def _transform_observation(self, resz, screen, objs=None):
    """Transforms the observation to the size of the screen.
    
    Args:
      resz: The resized output. To be written to.
      screen: The raw screen footage. RGB or grayscale.
      objs: `None` or a rank 3 tensor. The channels store objects.
    
    Returns:
      Nothing. See the output in `resz` instead.
    """
    if DQN_SCREEN_MODE == DQNScreenMode.RGB:
      resz[:, :, :DQN_SCREEN_LAY] = cv2.resize(
        screen[..., :DQN_SCREEN_LAY],
        (self.screen_size, self.screen_size),
        cv2.INTER_AREA)
    elif DQN_SCREEN_MODE == DQNScreenMode.GRAYSCALE:
      resz[:, :, :DQN_SCREEN_LAY] = cv2.resize(
        screen[..., :DQN_SCREEN_LAY],
        (self.screen_size, self.screen_size),
        cv2.INTER_AREA)[..., np.newaxis]
    if not DQN_USE_OBJECTS:
      return resz
    for index in range(DQN_NUM_OBJ):
      # No enclosing square braces around `index`: `resize` for 1 channel
      # yields 2D image (not a 3D one).
      resz[..., DQN_SCREEN_LAY + index] = cv2.resize(
        objs[..., [index]],
        (self.screen_size, self.screen_size),
        cv2.INTER_AREA)
    return resz

  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so. The call even works if we have
    # `DQN_SCREEN_LAY` set to `0`. In that case, no real operation takes place.
    if self.frame_skip > 1:
      np.maximum(
        self.screen_buffer[0][..., :DQN_SCREEN_LAY],
        self.screen_buffer[1][..., :DQN_SCREEN_LAY],
        out=self.screen_buffer[0][..., :DQN_SCREEN_LAY])
    
    resz = np.zeros(NATURE_DQN_OBSERVATION_SHAPE, dtype=np.uint8)
    if DQN_USE_OBJECTS:
      if self.frame_skip > 1:
        # If we have objects, also max-pool over the object channels.
        np.maximum(
          self.screen_buffer[0][..., DQN_SCREEN_LAY:],
          self.screen_buffer[1][..., DQN_SCREEN_LAY:],
          out=self.screen_buffer[0][..., DQN_SCREEN_LAY:])
      # resizing happens per-channel, so we can jointly resize
      # the RGB (or grayscale) channel(s) along with the object channels
      self._transform_observation(
        resz,
        self.screen_buffer[0][..., :DQN_SCREEN_LAY],
        self.screen_buffer[0][..., DQN_SCREEN_LAY:])
      int_screen = np.asarray(resz, dtype=np.uint8)
      return int_screen
    else:
      # just resize the RGB (or grayscale) channel(s)
      self._transform_observation(
        resz,
        self.screen_buffer[0][..., :DQN_SCREEN_LAY])
      int_screen = np.asarray(resz, dtype=np.uint8)
      return int_screen
  
  def _color_average_and_resize(self):
    """Averages over the color of two successive frames, and resizes the result.

    Returns:
      transformed_screen: Numpy array. Color-averaged, resized screen.
    """
    screen_0 = self.screen_buffer[0][..., :DQN_SCREEN_LAY]
    screen_1 = self.screen_buffer[1][..., :DQN_SCREEN_LAY]

    if self.frame_skip > 1:
      screen_0 = np.mean([screen_0, screen_1], axis=0).astype(np.uint8)
    
    if DQN_USE_OBJECTS:
      obj_channels_0 = self.screen_buffer[0][..., DQN_SCREEN_LAY:]
      obj_channels_1 = self.screen_buffer[1][..., DQN_SCREEN_LAY:]
      if self.frame_skip > 1:
        np.maximum(obj_channels_0, obj_channels_1, out=obj_channels_0)
      # resizing happens per-channel, so we can jointly resize
      # the RGB (or grayscale) channel(s) along with the object channels
      transformed_screen = self._transform_observation(screen_0, obj_channels_0)
      int_screen = np.asarray(transformed_screen, dtype=np.uint8)
      return int_screen
    else:
      transformed_screen = self._transform_observation(screen_0)
      int_screen = np.asarray(transformed_screen, dtype=np.uint8)
      return int_screen
