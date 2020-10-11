import argparse
import math

import numpy as np
import textworld


def parse_args():
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', default='./tw-games',
                        help='Output directory where the game files are generated.')
    parser.add_argument('--data_path', default='./iqa_dataset',
                        help='Directory where the dataset is stored')
    parser.add_argument('--initial_room', default=None,
                        help='Initial position of the player. Options: {kitchen, pantry, livingroom, bathroom, '
                             'bedroom, backyard, corridor, laundry room}.')
    parser.add_argument('--objects', default=3, type=int,
                        help='Number of objects that need to be placed in the correct location')
    parser.add_argument('--rooms', default=1, type=int, help='Number of rooms')
    parser.add_argument('--take', default=None, type=int,
                        help='Number of objects that need to be retrieved by the agent. Must be less than or equal to '
                             'the number of OBJECTS. If less than OBJECTS, the remaining ones are placed in the '
                             'inventory at the beginning of the game. If unspecified, the default value is equal to the'
                             ' number of OBJECTS')
    parser.add_argument('--drop', default=False, action='store_true', help='Limits the capacity of the inventory')
    parser.add_argument('--distractors', default=False, action='store_true',
                        help='Generate random distractors that are already in their correct location')
    parser.add_argument('--isolated_rooms', default=False, action='store_true',
                        help='Shuffle objects only within the correct room')
    parser.add_argument('--open', default=False, action='store_true',
                        help='Specify that containers need to be opened')
    parser.add_argument('--seed', type=int, default=None, help='General seed used by the random number generators')
    parser.add_argument('--seeds', type=int, nargs=4, default=None,
                        help='Seeds respectively for the map, the objects, the quests and the grammar')
    parser.add_argument("-f", "--force", action="store_true", help='Force recompiling the game')
    parser.add_argument("--play", action="store_true", help="Play the output game")

    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--train", action="store_true", default=False,
                             help="Use only the subset of the entities that is reserved for  the training set")

    split_group.add_argument("--test", action="store_true", default=False,
                             help="Use only the subset of the entities that is reserved for  the test set")

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--silent", action="store_true", help='Do not print any output')
    verbosity_group.add_argument("-v", "--verbose", action="store_true", help='Verbose mode')

    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None,
                        help="Difficulty level: [1 = Easy, 2 = Medium, 3 = Difficult]. Overwrites the other options.")

    return parser.parse_args()


def set_defaults(conf):
    if conf.seed is None:
        conf.seed = np.random.randint(65536)

    if conf.seeds is not None:
        conf.seed = {
            'map': conf.seeds[0],
            'objects': conf.seeds[1],
            'quest': conf.seeds[2],
            'grammar': conf.seeds[3]
        }

    if conf.take is None:
        conf.take = conf.objects

    assert not (conf.train and conf.test)


def get_game_options(conf):
    options = textworld.GameOptions()
    options.seeds = conf.seed
    options.nb_rooms = conf.rooms
    options.force_recompile = conf.force
    return options


def set_difficulty_level(conf, rng):
    if conf.level is None:
        return
    assert conf.level in [1, 2, 3]
    if conf.verbose:
        print('Setting difficulty level to', conf.level)

    object_settings = {
        1: list(range(1, 4)),
        2: list(range(4, 8)),
        3: list(range(8, 11))
    }

    room_settings = {
        1: [1],
        2: [1, 2, 3, 4],
        3: [4, 5]
    }

    def get_rooms(n_obj):
        mean_obj = sum(object_settings[conf.level]) / len(object_settings[conf.level])
        middle = len(room_settings[conf.level]) / 2
        if n_obj > mean_obj:
            return int(rng.choice(room_settings[conf.level][:math.ceil(middle)]))
        elif n_obj < mean_obj:
            return int(rng.choice(room_settings[conf.level][math.floor(middle):]))
        else:
            return int(rng.choice(room_settings[conf.level]))

    conf.objects = int(rng.choice(object_settings[conf.level]))
    index = object_settings[conf.level].index(conf.objects)
    conf.take = int(rng.choice(object_settings[conf.level][:(index + 1)]))
    conf.rooms = get_rooms(conf.objects)


config = parse_args()
set_defaults(config)
game_options = get_game_options(config)
rngs = game_options.rngs
set_difficulty_level(config, rngs["quest"])
