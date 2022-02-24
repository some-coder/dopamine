import os
import re
from dopamine.discrete_domains.atari_lib import \
  DQNScreenMode, \
  DQN_SCREEN_MODE, \
  DQN_USE_OBJECTS, \
  DQN_NUM_OBJ, \
  DQN_SCREEN_LAY, \
  DQN_OBJ_LAY, \
  NATURE_DQN_STACK_SIZE
from PIL import Image
import numpy as np
from collections import namedtuple


MS_PACMAN_PELLET_FILE = 'ms-pacman-pellet-locs.npy'
SALIENCY_PATH = \
    (
      '/data/s3366235/master-thesis/'
      if re.search('s3366235', os.environ['HOME']) else
      '/home/niels/Documents/test/'
    ) + 'saliency/'


Difference = namedtuple('Difference', ['dx', 'dy', 'show'])


def inspect_action_valuations(runner, name):
  """Runs an interactive program for inspecting action valuations.

  Args:
    runner: A `run_experiment.Runner` object to work with.
    name: Name of the directory to which to save information
      from this inspection.
  """
  try:
    os.makedirs(SALIENCY_PATH)
  except FileExistsError:
    print('Directory \'%s\' already exists. Skipping.' % (SALIENCY_PATH,)) 
  
  if input('(Create new observation sequence? [y/n]) ') == 'y':
    obs = runner.observations_sequence(SALIENCY_PATH)
    save_observations_sequence_end(obs, SALIENCY_PATH, '%s/original' % (name,))

  print('(Loading from directory \'%s%s\'...)' % (SALIENCY_PATH, name + '/original'))
  obs = load_observations_sequence_end(SALIENCY_PATH, '%s/original' % (name,))
  
  obs = interactively_manipulate_observation(runner, SALIENCY_PATH, name, obs)
  
  while input('(Perform state-action evaluations? [y/n]) ') == 'y':
    sub_dir = input('(Which subdirectory should be considered?) ')
    obs = load_observations_sequence_end(SALIENCY_PATH, '%s/%s' % (name, sub_dir))
    evals = runner.state_action_evaluations(obs)
    print('\tEVALS: %s' % (str(evals),))


def save_ms_pacman_pellet_removal_states(runner):
  try:
      os.makedirs(SALIENCY_PATH)
  except FileExistsError:
      print('Directory \'%s\' already exists. Skipping.' % (SALIENCY_PATH,))

  all_axes = DQN_SCREEN_LAY + DQN_OBJ_LAY

  locs = np.load(SALIENCY_PATH + '/' + MS_PACMAN_PELLET_FILE)
  obs = load_observations_sequence_end(SALIENCY_PATH, 'ms-pacman/original')
  all_locs = np.zeros((2, 2, NATURE_DQN_STACK_SIZE), dtype=np.uint8)
  for loc_idx in range(locs.shape[0]):
    print('LOCATION INDEX %2d...' % (loc_idx,))
    obs_copy = np.copy(obs)
    
    for i in range(NATURE_DQN_STACK_SIZE):
      all_locs[i // 2, np.mod(i, 2), :] = \
        [locs[loc_idx, i]] * NATURE_DQN_STACK_SIZE
    
    manip = manipulate_object(
      runner,
      obs_copy,
      3,
      Difference(dx=0, dy=0, show=False),
      all_locs)
    save_observations_sequence_end(
      [
        manip[0, ..., (all_axes * frame):(all_axes * (frame + 1))]
        for frame in range(NATURE_DQN_STACK_SIZE)
      ],
      SALIENCY_PATH,
      'ms-pacman/pellet-%d' % (loc_idx,))
    print('DONE.')


def create_object_saliency_map(runner, name, obj_list):
  """Creates an object saliency map given a set of objects.

  The objects are specified by absolute directory paths;
  each of the directories stores the same observations
  as the original, except that the object in question has
  been removed. This removal allows for a comparison.

  Args:
    runner: The `run_experiment.Runner` to work with.
    name: The name of the `SALIENCY_PATH` subdirectory.
      Commonly the name of an Atari 2600 game. Should
      not include an ending `/`.
    obj_list: A list of subdirectories under
      `SALIENCY_PATH/name`.
  Returns:
    map: The object saliency map.
  """
  map = np.zeros((84, 84), dtype=np.float32)
  
  original_obs = load_observations_sequence_end(
    SALIENCY_PATH,
    name + '/original'
  )
  original_eval = runner.state_action_evaluations(original_obs)
  best_act, best_val = list(original_eval.items())[0]
  
  for act, val in original_eval.items():
    if val > best_val:
      best_act, best_val = act, val
  for ctr, obj in enumerate(obj_list):
    print(
      'Considering object \'%s\' (%2d/%2d, %6.3lf%%)' %
      (
        obj,
        ctr + 1,
        len(obj_list),
        ((ctr + 1) / len(obj_list)) * 1e2
      )
    )
    obj_obs = load_observations_sequence_end(
      SALIENCY_PATH,
      name + '/' + obj
    )
    obj_val = runner.state_action_evaluations(obj_obs)[best_act]
    val_diff = best_val - obj_val
    for layer in range(DQN_NUM_OBJ if DQN_USE_OBJECTS else 0):
      # We consider the last (newest) frame only: the valuation applies
      # to the complete sequence of four observations, so it doesn't
      # really matter which frame we pick.
      layer_idx = -(DQN_NUM_OBJ - layer)
      obs_diff = original_obs[0, ..., layer_idx] - obj_obs[0, ..., layer_idx]
      map[np.where(obs_diff > 0)] = val_diff
  return map


def save_observations_sequence_end(obs, save_location, save_name):
  """Saves the end of a sequence of observations to disk.

  Args:
    obs: The full observation sequence. Only the last
      four observations are saved to disk.
    save_location: The absolute path to the directory
      to save to.
    save_name: The name of the directory in which the sequence
      will be saved. This directory will be a direct subdirectory
      of `save_location`.
  """
  try:
    print('(Attempting to create directory \'%s\'...)' % (os.path.join(save_location, save_name),))
    os.makedirs(os.path.join(save_location, save_name))
    print('(Done.)')
  except FileExistsError:
    print('\t(Overwriting contents of \'%s/%s\'...)' % (save_location, save_name))
  num_obj = DQN_NUM_OBJ if DQN_USE_OBJECTS else 0
  for frame in range(NATURE_DQN_STACK_SIZE):
    # frame 1 is the third-to-last frame, ..., frame 4 is the last frame
    if DQN_SCREEN_MODE == DQNScreenMode.RGB:
      Image.fromarray(
        obs[-(NATURE_DQN_STACK_SIZE - frame)][..., :3]
      ).save(
        os.path.join(
          save_location,
          save_name,
          '%d.png' % (frame + 1,)
        )
      )
    elif DQN_SCREEN_MODE == DQNScreenMode.GRAYSCALE:
      Image.fromarray(
        obs[-(NATURE_DQN_STACK_SIZE - frame)][..., 0], 'L'
      ).save(
        os.path.join(
          save_location,
          save_name,
          '%d.png' % (frame + 1,)
        )
      )
    for obj_idx in range(num_obj):
      Image.fromarray(
        obs[-(NATURE_DQN_STACK_SIZE - frame)][..., DQN_SCREEN_LAY + obj_idx],
        'L'
      ).save(
        os.path.join(
          save_location, save_name, '%d-obj-%d.png' % (frame + 1, obj_idx + 1)
        )
      )


def load_observations_sequence_end(load_location, load_name):
  """Loads the end of a sequence of observations from disk.

  Args:
    load_location: The absolute path to the directory
      to load from.
    load_name: The name of the directory from which the sequence
      will be loaded. This directory will be a direct subdirectory
      of `load_location`.
  Returns:
    obs: The last four observations from the sequence. Returned
      as a NumPy array.
  """
  all_axes = DQN_SCREEN_LAY + DQN_OBJ_LAY
  layers = NATURE_DQN_STACK_SIZE * all_axes
  num_obj = DQN_NUM_OBJ if DQN_USE_OBJECTS else 0

  obs = np.zeros((1, 84, 84, layers), dtype=np.uint8)
  for frame in range(NATURE_DQN_STACK_SIZE):
    # store frame screen image (RGB or grayscale)
    img = Image.open(
      os.path.join(load_location, load_name, '%d.png' % (frame + 1,)))
    
    if DQN_SCREEN_MODE == DQNScreenMode.RGB:
      obs[..., (all_axes * frame):(all_axes * frame + DQN_SCREEN_LAY)] = \
        np.array(img)
    elif DQN_SCREEN_MODE == DQNScreenMode.GRAYSCALE:
      obs[..., (all_axes * frame):(all_axes * frame + DQN_SCREEN_LAY)] = \
        np.array(img)[..., np.newaxis]

    # store frame screen objects
    for obj_idx in range(num_obj):
      obj_img = Image.open(
        os.path.join(
          load_location,
          load_name,
          '%d-obj-%d.png' % (frame + 1, obj_idx + 1)))
      pre = all_axes * frame + DQN_SCREEN_LAY
      obs[..., (pre + obj_idx):(pre + obj_idx + 1)] = \
        np.array(obj_img)[..., np.newaxis]
  
  return obs


def get_manipulation():
  """Gets a manipulation to apply to the state from standard input.

  Returns:
    manip: A three-tuple. The first element contains the object index.
      This is the layer, starting inclusively from the first screen
      channel. (So, if you have an RGB screen and want the second
      object channel, then say `3 + 2 - 1 = 4`.) The second element
      stores the locations of the object through all four
      frames of the state. The third and last element contains
      the manipulation: a `(dy, dx, show)` triple. See
      `Difference` for more info.
  """
  # first, get the index of the object to be manipulated
  obj_idx = int(input('(Index of object to manipulate? [This is the layer.]) '))

  # second, get the location of the object in all four frames
  all_locs = np.zeros((2, 2, NATURE_DQN_STACK_SIZE), dtype=np.int8)
  for frame in range(NATURE_DQN_STACK_SIZE):
    print('\t[FRAME %d/%d]' % (frame + 1, NATURE_DQN_STACK_SIZE))
    all_locs[0, 0, frame] = int(input('\t    (Top-left X. Inclusive.) '))
    all_locs[0, 1, frame] = int(input('\t    (Top-left Y. Inclusive.) '))
    all_locs[1, 0, frame] = int(input('\t(Bottom-right X. Inclusive.) '))
    all_locs[1, 1, frame] = int(input('\t(Bottom-right Y. Inclusive.) '))

  show = True if input('(Still show the object [y/n]?) ') == 'y' else False
  if not show:
    return obj_idx, all_locs, Difference(dy=0, dx=0, show=show)
  dy = int(input('(Delta Y. Higher means lower.) '))
  dx = int(input('(Delta X. Higher means more to the right.) '))
  return obj_idx, all_locs, Difference(dy=dy, dx=dx, show=show)


def manipulate_object(runner, obs, obj_idx, diff, all_locs):
  """Yields a new observation that sees one object being manipulated.

  A single channel of `all_locs` stores X values in the zeroeth
  column and Y values in the first column. The zeroeth row stores
  the upper-left point of the bounding box (BB) and the first
  row stores the lower-right BB. Both X and Y coordinates are
  inclusive. So they don't work like e.g. NumPy ranges, where
  the Y coordinate is explicit.

  Args:
    runner: The `run_experiment.Runner` object to work with.
    obs: The state. A stack of four observations. Each observation
      consists of a raw screen signal plus one or more object
      channels.
    obj_idx: The index of the object layer in which we would
      like to manipulate an object.
    diff: The manipulation to apply to the object. Either applies
      a disappearance or a translation.
    all_locs: A 2-by-2-by-4 tensor. Rows store points, columns
      X and Y coordinates,
      and the depth stores the two bounding box points for
      each single frame.
  Returns:
    obs_mnp: `obs`, but with manipulation `diff` applied. Note
      that `obs_mnp` and `obs` are distinct; `obs` is not
      altered by this method.
  """
  modif = np.zeros(obs.shape, dtype=np.uint8)
  modif[:] = obs
  
  # copy the screen layer(s)
  modif[0, ..., :DQN_SCREEN_LAY] = \
    obs[0, ..., :DQN_SCREEN_LAY]
  frm_layers = DQN_SCREEN_LAY + DQN_OBJ_LAY
  
  for obs_idx in range(obs.shape[-1]):
    if obs_idx % frm_layers != obj_idx:
      continue  # not the right layer
    loc = all_locs[..., obs_idx // frm_layers]
    x_mn, x_mx, y_mn, y_mx = (loc[0][0], loc[1][0] + 1, loc[0][1], loc[1][1] + 1)
    
    # get the old screen and object appearances
    imt_screen = np.copy(modif[0, y_mn:y_mx, x_mn:x_mx, frm_layers * (obs_idx // frm_layers)])
    imt_obj = np.copy(modif[0, y_mn:y_mx, x_mn:x_mx, obs_idx])
    
    # clear the old screen and object
    modif[0, y_mn:y_mx, x_mn:x_mx, frm_layers * (obs_idx // frm_layers)] = runner._environment.bg_color
    modif[0, y_mn:y_mx, x_mn:x_mx, obs_idx] = 0
    
    # place the new screen and object
    if diff.show:
      modif[
        0,
        (y_mn + diff.dy):(y_mx + diff.dy),
        (x_mn + diff.dx):(x_mx + diff.dx),
        frm_layers * (obs_idx // frm_layers)] = imt_screen
      modif[
        0,
        (y_mn + diff.dy):(y_mx + diff.dy),
        (x_mn + diff.dx):(x_mx + diff.dx),
        obs_idx] = imt_obj
  
  return modif


def interactively_manipulate_observation(runner, pth, name, obs):
  all_axes = DQN_SCREEN_LAY + DQN_OBJ_LAY
  manip_ctr = 0
  while input('(Perform manipulation %d? [y/n]) ' % (manip_ctr,)) == 'y':
    obj_idx, all_locs, diff = get_manipulation()

    manip_obs = \
      manipulate_object(runner, obs, obj_idx, diff, all_locs)
    save_observations_sequence_end(
      [
        manip_obs[0, ..., (all_axes * frame):(all_axes * (frame + 1))]
        for frame in range(4)
      ],
      pth,
      '%s/manip-%d' % (name, manip_ctr)
    )
    obs = load_observations_sequence_end(pth, '%s/manip-%d' % (name, manip_ctr))
    manip_ctr += 1
