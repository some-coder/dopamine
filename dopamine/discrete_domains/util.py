"""Utility symbols for discrete DRL domains."""


from dopamine.discrete_domains.atari_lib import \
    DQN_NUM_OBJ, \
    DQN_SCREEN_MODE, \
    DQN_USE_OBJECTS, \
    NATURE_DQN_OBSERVATION_SHAPE, \
    DQNScreenMode, \
    create_atari_environment, \
    atari_objects_map, \
    atari_background_color

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import re
import os
from pathlib import Path

from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class AtariActionMap:
    """A mapping from integers to movements in an Atari 2600 game.
    
    The `up` and `down` actions are always defined. `left` and `right` may not
    be defined for the simplest games, such as Pong.
    """
    up: int
    down: int
    left: Optional[int] = None
    right: Optional[int] = None
    noop: Optional[int] = None


@dataclass
class PongActionMap(AtariActionMap):
    """A mapping from integers to Atari 2600 Pong movements."""
    up: int = 2
    down: int = 5
    noop: Optional[int] = 0


@dataclass
class MsPacmanActionMap(AtariActionMap):
    """A mapping from integers to Atari 2600 Ms. Pac-Man movements."""
    up: int = 1
    down: int = 4
    left: Optional[int] = 3
    right: Optional[int] = 2


@dataclass
class ObjectSaliencyMapBufferEntry:
    """An entry into an object saliency map buffer."""
    raw_rgb_frame: np.ndarray
    preprocessed_frame: np.ndarray

    @staticmethod
    def as_grayscale(rgb: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(rgb).convert('L'))

    @property
    def raw_gray_frame(self) -> np.ndarray:
        return ObjectSaliencyMapBufferEntry.as_grayscale(
            self.raw_rgb_frame
        )


class ObjectSaliencyMapBuffer:
    """A buffer designed to store visual DRL frames to create object saliency
    maps.
    """

    def __init__(self, buf_size: int = 4) -> None:
        """Constructs a simple buffer.

        :param buf_size: The maximum number of elements the buffer may store.
        """
        self.buf: List[ObjectSaliencyMapBufferEntry] = []
        self.max_len: int = buf_size

    def enqueue(
            self,
            raw_rgb_frame: np.ndarray,
            preprocessed_frame: np.ndarray) -> None:
        """Enqueues an entry at the end of the buffer.

        If the buffer is at its maximum number of elements, the oldest element
        gets discarded.

        :param raw_rgb_frame: The raw RGB frame.
        :param preprocessed_frame: The preprocessed frame.
        """
        value = ObjectSaliencyMapBufferEntry(
            raw_rgb_frame,
            preprocessed_frame
        )
        if len(self.buf) == self.max_len:
            self.buf = self.buf[1:] + [value]
        else:
            self.buf += [value]
    
    def empty(self) -> None:
        """Clear the buffer's contents."""
        self.buf = []


@dataclass
class ObjectBoundingBox:
    """A bounding box for an object in a to-be-preporcessed image.
    
    The minima in X and Y are inclusive; the maxima in X and Y are exclusive.
    """
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    name: Optional[str] = None
    buf_entry: Optional[int] = None


class ObjectSaliencyMapAssistant:
    """A convenience class meant to help in the creation of object saliency
    maps.
    
    This class is intended for use at the interactive command prompt. By
    creating an `ObjectSaliencyMapAssistant` and using its interface, you
    may more quickly create object saliency maps. It achieves this by
    abstracting away the low-level OpenAI Gym and Google Dopamine API
    interactions as much as possible, without completely hiding them.
    """

    SUPPORTED_GAMES: List[str] = ['Pong', 'MsPacman']
    DILATE_KERNEL = \
        cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
    MS_PACMAN_LIVES_BACKGROUND: int = 0  # completely black
    
    # maps grayscale tones to names of instances withina shared category
    PONG_PADDLE_NAMES: Dict[int, str] = \
        {147: 'green-paddle', 148: 'orange-paddle'}
    MS_PACMAN_GHOST_NAMES: Dict[int, str] = \
        {
            110: 'blinky-red',
            131: 'clyde-yellow',
            151: 'inky-blue',
            132: 'pinky-pink'
        }
    MS_PACMAN_WALL_NAMES: Dict[int, str] = \
        {146: 'pink', 121: 'gray', 170: 'blue', 132: 'green'}

    SAVE_DIR = Path(os.environ['HOME']) / Path(f'Documents/test/saliency-2')

    def __init__(self, game: str) -> None:
        assert DQN_USE_OBJECTS
        assert game in self.SUPPORTED_GAMES
        self.env = create_atari_environment(game)
        self.map = self.action_map(game)
        self.buf = ObjectSaliencyMapBuffer()
        self.game = game
        self.objects_map = atari_objects_map(self.game)
        self.background_color = atari_background_color(self.game)

    @staticmethod
    def action_map(game: str) -> AtariActionMap:
        if game == 'Pong':
            return PongActionMap()
        elif game == 'MsPacman':
            return MsPacmanActionMap()
        else:
            raise NotImplementedError(f'Game {game} not has no action map!')

    def start(self) -> None:
        self.env.reset()
        self.env.full_env.render()
    
    def step(self, action: str) -> None:
        int_action: int
        if action == 'up':
            int_action = self.map.up
        elif action == 'down':
            int_action = self.map.down
        elif action == 'left':
            int_action = self.map.left
        elif action == 'right':
            int_action = self.map.right
        elif action == 'noop':
            int_action = self.map.noop
        else:
            raise ValueError(f'Action {action} isn\'t a legal value.')
        obs, _, _, _ = self.env.step(int_action)
        self.buf.enqueue(
            raw_rgb_frame=self.env.environment.ale.getScreenRGB(),
            preprocessed_frame=obs
        )
        self.env.full_env.render()
    
    def multi_step(self, action: str, repetitions: int) -> None:
        for _ in range(repetitions):
            self.step(action)

    def reset(self) -> None:
        self.env.reset()
        self.buf.empty()
        self.env.full_env.render()

    def manually_designate_object_bounding_box(self) -> ObjectBoundingBox:
        plt.imshow(self.env.environment.ale.getScreenRGB())
        plt.show(block=False)
        bb_args: Dict[str, int] = {}
        for bb_arg in ('x_min', 'y_min', 'x_max', 'y_max'):
            bb_arg_val = input(f'({bb_arg}?) ')
            while not bb_arg_val.isdigit():
                print('(Not an integer!)', end=' ')
                bb_arg_val = input(f'{bb_arg}? ')
            bb_args[bb_arg] = int(bb_arg_val)
        bb_args['name'] = input(f'(Name?) ')
        bb_args['buf_entry'] = int(input(f'(Index of buffer entry?) '))
        plt.close()
        return ObjectBoundingBox(**bb_args)   

    def automatic_obb_name(
            self,
            template_name: str,
            x_min: int,
            y_min: int,
            img: np.ndarray) -> str:
        assert DQN_SCREEN_MODE != DQNScreenMode.RGB  # won't work
        if self.game == 'Pong':
            if template_name == 'paddle-piece-wide':
                return self.PONG_PADDLE_NAMES[int(img[0, 6])]
            elif template_name == 'ball-padded':
                return 'ball'
        elif self.game == 'MsPacman':
            if template_name == 'blinky-red-padded':
                return self.MS_PACMAN_GHOST_NAMES[int(img[-4, 3])]
            elif template_name == 'pellet-padded':
                return f'pellet-y-{y_min}-x-{x_min}'
            elif template_name == 'power-pellet-padded':
                return f'power-pellet-y-{y_min}-x-{x_min}'
        raise ValueError(
            f'Couldn\'t find a name for \'{template_name}\'!'
        )

    def automatically_designate_object_bounding_box(
            self,
            name: str,
            buf_entry: int) -> List[ObjectBoundingBox]:
        if name not in self.objects_map.keys():
            raise KeyError(
                'No such key. Try one of the following: ' +
                ', '.join([f"'{k}'" for k in self.objects_map.keys()])
            )
        template, threshold, mask = self.objects_map[name]
        if mask is not None:
            raise ValueError(
                f'Object \'{name}\' uses masks, which are not supported!'
            )
        if DQN_SCREEN_MODE == DQNScreenMode.RGB:
            img = self.env.environment.ale.getScreenRGB()
        else:
            img = self.env.environment.ale.getScreenGrayscale()
        sq_diff = cv2.matchTemplate(
            image=img,
            templ=template,
            method=cv2.TM_SQDIFF_NORMED
        )
        locs = np.where(sq_diff < threshold)
        bbs: List[ObjectBoundingBox] = []
        for _, (y, x) in enumerate(zip(*locs)):
            sub_img = img[y:(y + template.shape[0]), x:(x + template.shape[1])]
            bbs.append(
                ObjectBoundingBox(
                    x_min=x,
                    x_max=x + template.shape[1],
                    y_min=y,
                    y_max=y + template.shape[0],
                    name=self.automatic_obb_name(name, x, y, sub_img),
                    buf_entry=buf_entry
                )
            )
        return bbs

    @staticmethod
    def names_to_obb_map(obb_list: List[ObjectBoundingBox]) -> \
            Dict[str, List[ObjectBoundingBox]]:
        obb_names: Set[str] = set(entry.name for entry in obb_list)
        obb_map: Dict[str, List[ObjectBoundingBox]] = \
            {key: [] for key in obb_names}
        for entry in obb_list:
            obb_map[entry.name].append(entry)
        return obb_map

    @staticmethod
    def map_keys_starting_with(map: Dict[str, Any], start: str) -> List[str]:
        return [key for key in map.keys() if re.search(f'^({start})', key)]

    def cleaned_obb_map(self, obb_map: Dict[str, List[ObjectBoundingBox]]) -> \
            Dict[str, List[ObjectBoundingBox]]:
        if self.game == 'Pong':
            pass  # no work needed: assumed to be fully manual
        elif self.game == 'MsPacman':
            ghost_re: str = '|'.join(('blinky', 'clyde', 'inky', 'pinky'))
            for key in self.map_keys_starting_with(obb_map, ghost_re):
                reduction: Dict[int, Optional[ObjectBoundingBox]] = \
                    {bi: None for bi in tuple(range(4))}
                first_non_empty: Optional[ObjectBoundingBox] = None
                for entry in obb_map[key]:
                    if reduction[entry.buf_entry] is None:
                        reduction[entry.buf_entry] = entry
                    if first_non_empty is None:
                        first_non_empty = entry
                assert first_non_empty is not None
                for sub_key in [k for k, v in reduction.items() if v is None]:
                    reduction[sub_key] = first_non_empty
                    reduction[sub_key].buf_entry = sub_key  # fix buffer indexing
                obb_map[key] = list(reduction.values())  # `List[ObjectBoundingBox]`
            for key in self.map_keys_starting_with(obb_map, 'pellet'):
                if len(obb_map[key]) != 4:
                    del obb_map[key]  # no occlusion at any step
            for key in self.map_keys_starting_with(obb_map, 'power-pellet'):
                obb_map[key] = [obb_map[key][0]] * 4  # ignore the flashing
                for index, entry in enumerate(obb_map[key]):
                    entry.buf_entry = index  # fix buffer indexing
        return obb_map
    
    def quick_ms_pacman_obb_list(self) -> List[ObjectBoundingBox]:
        assert self.game == 'MsPacman'
        self.start()
        self.multi_step('left', 68)
        out: List[ObjectBoundingBox] = []
        for index in range(4):
            print(f'FRAME {index + 1}/4')
            out.append(self.manually_designate_object_bounding_box())
            out[-1].name = 'ms-pacman'
            out[-1].buf_entry = index  # ensure these two fields are legal
            out += self.automatically_designate_object_bounding_box(
                name='blinky-red-padded', buf_entry=index
            )
            out += self.automatically_designate_object_bounding_box(
                name='pellet-padded', buf_entry=index
            )
            out += self.automatically_designate_object_bounding_box(
                name='power-pellet-padded', buf_entry=index
            )
            self.step('left')
        return out


    @staticmethod
    def normal_treatment_applied_to_object_channel(
            raw: np.ndarray,
            threshold: float,
            sq_diff: np.ndarray,
            template: np.ndarray) -> np.ndarray:
        locs = np.where(sq_diff < threshold)
        out = np.zeros(shape=raw.shape[:2])
        for y, x in zip(*locs):
            out[
                y:(y + template.shape[0]),
                x:(x + template.shape[1])
            ] = 255
        return out

    @staticmethod
    def object_channel_requires_special_treatment(
            game: str,
            channel_index: int) -> bool:
        return \
            game == 'MsPacman' and \
            channel_index == (DQN_NUM_OBJ - 1) and \
            DQN_SCREEN_MODE == DQNScreenMode.OFF

    @staticmethod
    def special_treatment_applied_to_object_channel(
            raw: np.ndarray,
            thresholds: Tuple[float, float, float, float],
            sq_diff: np.ndarray) -> np.ndarray:
        locs: Optional[Tuple[np.ndarray, np.ndarray]] = None
        wall_enum = enumerate(
            ObjectSaliencyMapAssistant.MS_PACMAN_WALL_NAMES.keys()
        )
        for index, wall_thr in wall_enum:
            if raw[1, 0] == wall_thr:
                locs = np.where(sq_diff < thresholds[index])
                break
        assert locs is not None  # threshold must've been found
        out = np.zeros(shape=raw.shape[:2])
        out[locs[0], locs[1]] = 255
        out = cv2.dilate(
            src=out,
            kernel=ObjectSaliencyMapAssistant.DILATE_KERNEL,
            iterations=1
        )
        return out

    def object_channels_from_raw_image(self, raw: np.ndarray) -> np.ndarray:
        obj_chn = np.zeros(shape=raw.shape[:2] + (len(self.objects_map),))
        for idx, (_, (tpl, thr, mask)) in enumerate(self.objects_map.items()):
            sq_diff = cv2.matchTemplate(
                image=raw,
                templ=tpl,
                method=cv2.TM_SQDIFF_NORMED if mask is None else cv2.TM_SQDIFF,
                mask=mask
            )
            if self.object_channel_requires_special_treatment(self.game, idx):
                obj_chn[..., idx] = \
                    self.special_treatment_applied_to_object_channel(
                        raw, thr, sq_diff
                    )
            else:
                obj_chn[..., idx] = \
                    self.normal_treatment_applied_to_object_channel(
                        raw, thr, sq_diff, tpl
                    )
        return obj_chn

    def determined_background_color(self, obb: ObjectBoundingBox) -> int:
        assert DQN_SCREEN_MODE != DQNScreenMode.RGB  # not supported now
        if self.game == 'MsPacman' and obb.y_min > 171:
            return self.MS_PACMAN_LIVES_BACKGROUND
        else:
            return self.background_color

    def input_with_single_object_removed(
            self,
            obb: ObjectBoundingBox,
            buf_entry: ObjectSaliencyMapBufferEntry) -> \
                Tuple[np.ndarray, np.ndarray]:
        if DQN_SCREEN_MODE == DQNScreenMode.RGB:
            raw_copy = np.copy(buf_entry.raw_rgb_frame)
        else:
            raw_copy = np.copy(buf_entry.raw_gray_frame)
        raw_copy[obb.y_min:obb.y_max, obb.x_min:obb.x_max] = \
            self.determined_background_color(obb)
        out = self.object_channels_from_raw_image(raw_copy)
        if DQN_SCREEN_MODE != DQNScreenMode.OFF:
            if DQN_SCREEN_MODE == DQNScreenMode.GRAYSCALE:
                raw_copy = np.expand_dims(raw_copy, axis=-1)
            out = np.concatenate((raw_copy, out), axis=-1)
        return out, raw_copy

    def inputs_with_objects_removed(
            self,
            obb_map: Dict[str, List[ObjectBoundingBox]]) -> \
                Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        d: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for name, obbs in obb_map.items():
            d[name] = []
            for index, buf_entry in enumerate(self.buf.buf):
                d[name].append(
                    self.input_with_single_object_removed(
                        obbs[index], buf_entry
                    )
                )
        return d
    
    def resized_inputs(
            self,
            inputs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> \
                Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        for key in inputs.keys():
            for index in range(len(inputs[key])):
                inputs[key][index] = (
                    cv2.resize(
                        inputs[key][index][0],
                        NATURE_DQN_OBSERVATION_SHAPE[:2],
                        cv2.INTER_AREA
                    ),
                    inputs[key][index][1]  # leave raw as-is
                )
        return inputs
    
    def save_inputs(
            self,
            inputs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
            scene_name: str) -> None:
        root = self.SAVE_DIR / Path(scene_name)
        if not os.path.exists(root):
            os.makedirs(root)
        elif len(os.listdir(root)) > 0:
            raise FileExistsError(f'Remove all content in \'{root}\' first!')
        original_list = [
            (be.preprocessed_frame, be.raw_gray_frame) for be in self.buf.buf
        ]
        inputs.update({'original': original_list})
        for obj_name, inp in inputs.items():
            obj_dir = root / Path(obj_name)
            os.makedirs(obj_dir)
            for index, (preprocessed, raw) in enumerate(inp):
                Image.fromarray(
                    raw, mode=('L' if len(raw.shape) == 2 else None)
                ).save(obj_dir / f'raw-{index}.png')
                for chn_idx in range(preprocessed.shape[2]):
                    chn_img = Image.fromarray(
                        preprocessed[..., chn_idx].astype(np.uint8), mode='L'
                    )
                    chn_img.save(obj_dir / f'processed-{index}-{chn_idx}.png')

    @staticmethod
    def repair_scene_directory(
            scene_name: str,
            obj_dir_1: str,
            obj_dir_2: str) -> None:
        root = ObjectSaliencyMapAssistant.SAVE_DIR / Path(scene_name)
        origin_dir = root / 'origin'
        if not os.path.exists(origin_dir):
            os.makedirs(origin_dir)
        elif len(os.listdir(origin_dir)) > 0:
            raise FileExistsError(
                f'Remove all content in \'{origin_dir}\' first!'
            )
        for img_name in os.listdir(root / obj_dir_1):
            img_1 = Image.open(root / obj_dir_1 / img_name)
            img_2 = Image.open(root / obj_dir_2 / img_name)
            merged = np.maximum(np.array(img_1), np.array(img_2))
            merged = merged.astype(np.uint8)
            Image.fromarray(merged).save(origin_dir / img_name)
