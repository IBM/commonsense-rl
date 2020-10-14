#!/usr/bin/env python

import argparse
import sys
from os.path import join as pjoin

import gym
import networkx as nx
import numpy as np
import textworld
import textworld.gym
from textworld import GameMaker
from textworld.generator.game import Quest, Event
from textworld.utils import encode_seeds
from twc_data import TWCData

INTRO = "Welcome to TextWorld! You find yourself in a messy house. Many things are not in their usual location. " \
        "Let's clean up this place. Once you are done, this little house is going to be spick and span!"
GOAL = "Look for anything that is out of place and put it away in its proper location."

FLOOR = "<FLOOR>"

DEFAULT_FURNITURE = ["dining table", "fridge", "kitchen cupboard", "bed", "wardrobe", "sofa", "end table", "toilet",
                     "shower", "sink", "BBQ", "washing machine", "clothesline", "shelf"]


def parse_args():
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', default='./twc_games',
                        help='Output directory where the game files are generated.')
    parser.add_argument('--data_path', default='./twc_dataset',
                        help='Directory where the dataset is stored')
    parser.add_argument('--initial_room', default=None,
                        help='Initial position of the player. Options: {kitchen, pantry, livingroom, bathroom, '
                             'bedroom, backyard, corridor, laundry room}.')
    parser.add_argument('--objects', default=3, type=int,
                        help='Number of objects that need to be placed in the correct location')
    parser.add_argument('--rooms', default=1, type=int, help='Number of rooms')
    parser.add_argument('--num_games', default=1, type=int, help='Number of games to generate')

    parser.add_argument("--level", choices=['easy', 'medium', 'hard'], default=None,
                        help="The difficulty level of the game. This option overwrites the others. "
                             "Easy games have 1 object and 1 room; "
                             "Medium games have 2 or 3 objects and 1 room; "
                             "Hard games have 6 or 7 objects and 1 or 2 rooms.")

    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--train", action="store_true", default=False,
                             help="Use only the subset of the entities that is reserved for  the training set")

    split_group.add_argument("--test", action="store_true", default=False,
                             help="Use only the subset of the entities that is reserved for  the test set")

    parser.add_argument("--reward", type=int, default=1,
                        help="Reward for placing an object in its correct location.")

    parser.add_argument("--intermediate_reward", type=int, default=0,
                        help="Specify an intermediate reward for actions that are necessary but do not achieve the "
                             "goal of placing an object in its correct location.")

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
    parser.add_argument('--train_distribution_seed', type=int, default=1234,
                        help='Seed of the random number generator used to sample the in/out distribution entities. '
                             'This should normally not be changed.')
    parser.add_argument("--train_size", default=0.67, type=float,
                        help='Fraction of entities to use for the training distribution')
    parser.add_argument("-f", "--force", action="store_true", help='Force recompiling the game')
    parser.add_argument("--play", action="store_true", help="Play the output game")

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--silent", action="store_true", help='Do not print any output')
    verbosity_group.add_argument("-v", "--verbose", action="store_true", help='Verbose mode')

    return parser.parse_args()


def set_defaults(conf):
    if conf.seed is None:
        conf.seed = np.random.randint(65537)

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


def set_difficulty_level(conf):
    rng = conf.rngs['quest']
    if conf.level is None:
        return
    assert conf.level in ['easy', 'medium', 'hard']
    if conf.verbose:
        print('Setting difficulty level to', conf.level)

    object_settings = {
        'easy': [1],
        'medium': [2, 3],
        'hard': [6, 7]
    }

    take_settings = {
        'easy': 1,
        'medium': 1,
        'hard': 5
    }

    room_settings = {
        'easy': [1],
        'medium': [1],
        'hard': [1, 2]
    }

    conf.objects = int(rng.choice(object_settings[conf.level]))
    take_pool = list(range(take_settings[conf.level], conf.objects + 1))
    conf.take = int(rng.choice(take_pool))
    conf.rooms = int(rng.choice(room_settings[conf.level]))


def twc_config():
    config = parse_args()
    set_defaults(config)
    game_options = get_game_options(config)
    config.game_options = game_options
    rngs = game_options.rngs
    config.rngs = rngs
    set_difficulty_level(config)
    return config


class RandomWalk:
    def __init__(self, neighbors, size=(5, 5), max_attempts=200, rng=None):
        self.max_attempts = max_attempts
        self.neighbors = neighbors
        self.rng = rng or np.random.RandomState(1234)
        self.grid = nx.grid_2d_graph(size[0], size[1], create_using=nx.OrderedGraph())
        self.nb_attempts = 0

    def _walk(self, graph, node, remaining):
        if len(remaining) == 0:
            return graph

        self.nb_attempts += 1
        if self.nb_attempts > self.max_attempts:
            return None

        nodes = list(self.grid[node])
        self.rng.shuffle(nodes)
        for node_ in nodes:
            neighbors = self.neighbors[graph.nodes[node]["name"]]
            if node_ in graph:
                if graph.nodes[node_]["name"] not in neighbors:
                    continue

                new_graph = graph.copy()
                new_graph.add_edge(node, node_,
                                   has_door=False,
                                   door_state=None,
                                   door_name=None)

                new_graph = self._walk(new_graph, node_, remaining)
                if new_graph:
                    return new_graph

            else:
                neighbors = [n for n in neighbors if n in remaining]
                self.rng.shuffle(neighbors)

                for neighbor in neighbors:
                    new_graph = graph.copy()
                    new_graph.add_node(node_, id="r_{}".format(len(new_graph)), name=neighbor)

                    new_graph.add_edge(node, node_,
                                       has_door=False,
                                       door_state=None,
                                       door_name=None)

                    new_graph = self._walk(new_graph, node_, remaining - {neighbor})
                    if new_graph:
                        return new_graph

        return None

    def place_rooms(self, rooms):
        rooms = [rooms]
        nodes = list(self.grid)
        self.rng.shuffle(nodes)

        for start in nodes:
            graph = nx.OrderedGraph()
            room = rooms[0][0]
            graph.add_node(start, id="r_{}".format(len(graph)), name=room, start=True)

            for group in rooms:
                self.nb_attempts = 0
                graph = self._walk(graph, start, set(group) - {room})
                if not graph:
                    break

            if graph:
                return graph
        return None


class TWCGameMaker:
    def __init__(self, config):
        self.config = config
        self.data = TWCData(config)
        self.maker = GameMaker(config.game_options)
        self.num_games = 0

    def reset(self):
        self.maker = GameMaker(self.config.game_options)

    def make_game(self):
        rng_grammar = self.config.rngs["grammar"]
        self.maker.grammar = textworld.generator.make_grammar(self.maker.options.grammar, rng=rng_grammar)

        self.place_rooms()

        placed_objects = []

        while len(placed_objects) < self.config.objects:
            if self.config.verbose:
                print()
                print("====== Placing furniture ======")

            furniture = self.place_furniture()
            if not furniture:
                print()
                print(f"Could not generate the game with the provided configuration")
                sys.exit(-1)

            if self.config.verbose:
                print()
                print("====== Placing objects ======")

            placed_objects += self.place_objects()
            assert len(placed_objects) == len(set(placed_objects))

        if self.config.verbose:
            print()
            print("====== Shuffling objects ======")

        self.move_objects(placed_objects)

        if self.config.verbose and self.config.distractors:
            print()
            print("====== Placing distractors ======")

        self.place_distractors()

        self.set_container_properties()
        self.limit_inventory_size()
        self.maker.quests = self.generate_quests(placed_objects)

        self.check_properties()

        uuid = self.generate_uuid()
        game = self.maker.build()
        self.num_games += 1

        self.set_metadata(game, placed_objects)

        if self.config.verbose:
            print()
            print("====== Goal Locations ======")
            for obj, locations in game.metadata["goal_locations"].items():
                print(f'{obj} ->', ", ".join(locations))

        self.config.game_options.path = pjoin(self.config.output_dir, uuid)

        result = textworld.generator.compile_game(game, self.config.game_options)
        self.reset()
        return result

    def place_rooms(self):
        rng = self.config.rngs["map"]
        assert self.config.rooms <= len(self.data.rooms)
        initial_room = self.config.initial_room or rng.choice(self.data.rooms)
        rooms_to_place = self.pick_rooms(initial_room)
        if self.config.verbose:
            print("Rooms:", rooms_to_place)
        self.create_map(rooms_to_place)
        room = self.maker.find_by_name(initial_room)
        self.maker.set_player(room)

    def pick_name(self, names):
        rng = self.config.rngs["objects"]
        names = list(names)
        rng.shuffle(names)
        for name in names:
            if self.maker.find_by_name(name) is None:
                return name
        assert False

    def pick_rooms(self, initial_room):
        assert self.config.rooms <= len(self.data.rooms)
        rng = self.config.rngs["map"]
        visited = {initial_room}
        neighbors = set(self.data.map[initial_room])
        neighbors -= visited

        while len(visited) < self.config.rooms:
            room = rng.choice(list(neighbors))
            visited.add(room)
            neighbors |= set(self.data.map[room])
            neighbors -= visited

        return list(visited)

    def pick_correct_location(self, locations):
        rng = self.config.rngs["objects"]
        locations = list(locations)
        rng.shuffle(locations)
        for location in locations:
            holder = None
            if "." in location:
                room_name = location.split(".")[0]
                holder_name = location.split(".")[1]
                room = self.maker.find_by_name(room_name)
                if room is not None:
                    holder = next((e for e in room.content if e.infos.name == holder_name), None)
            else:
                holder = self.maker.find_by_name(location)
            if holder:
                return holder
        return None

    def pick_wrong_object_location(self, object_name, prefer_correct_room=None):
        rng = self.config.rngs["objects"]
        correct_locations = self.data.objects[object_name]["locations"]
        rng.shuffle(correct_locations)

        holder_names = {location.split(".")[-1] for location in correct_locations}
        forbidden = illegal_locations(self.data.objects[object_name])
        holder_names |= forbidden

        if prefer_correct_room is None:
            prefer_correct_room = self.config.isolated_rooms

        assert prefer_correct_room in [True, False]
        correct_room = self.find_correct_room(object_name)

        # Try to pick location in correct room
        if correct_room and prefer_correct_room:
            room_furniture = [e for e in correct_room.content if e.infos.type in ["c", "s"]]
            wrong_holders = [e for e in room_furniture if e.infos.name not in holder_names]
            if FLOOR not in holder_names:
                wrong_holders.append(correct_room)
            rng.shuffle(wrong_holders)
            if len(wrong_holders) > 0:
                return wrong_holders[0]

        # Pick a random supporter or container
        all_supporters = list(self.maker.findall("s"))
        all_containers = list(self.maker.findall("c"))
        all_rooms = self.maker.rooms

        all_holders = all_supporters + all_containers
        if FLOOR not in holder_names:
            all_holders += all_rooms

        rng.shuffle(all_holders)
        wrong_holders = [e for e in all_holders if e.infos.name not in holder_names]

        if len(wrong_holders) > 0:
            return wrong_holders[0]

        # No wrong location found. Create new furniture
        pool = [f for f in self.data.locations.keys() if f not in holder_names]
        return self.place_random_entity(pool)

    def find_correct_room(self, object_name):
        correct_locations = self.data.objects[object_name]["locations"]
        holder_names = {location.split(".")[-1] for location in correct_locations}

        for location in correct_locations:
            if "." in location:
                room_name = location.split(".")[0]
                return self.maker.find_by_name(room_name)
        for holder_name in holder_names:
            holder = self.maker.find_by_name(holder_name)
            if holder:
                return holder.parent

    def place_at(self, name, holder):
        entity = self.maker.new(type=self.data.entities[name]["type"], name=name)
        entity.infos.noun = name
        if "adjs" in self.data.entities[name] and self.data.entities[name]["adjs"]:
            entity.infos.adj = self.data.entities[name]["adjs"][0]
        if "desc" in self.data.entities[name]:
            entity.infos.desc = self.data.entities[name]["desc"][0]
        if "indefinite" in self.data.entities[name]:
            entity.infos.indefinite = self.data.entities[name]["indefinite"]
        for property_ in self.data.entities[name]["properties"]:
            entity.add_property(property_)
        holder.add(entity)
        self.log_entity_placement(entity, holder)
        return entity

    def log_entity_placement(self, entity, holder):
        name = entity.infos.name
        if self.config.verbose:
            if self.data.entities[name]["category"] == "furniture":
                print(f"{entity.infos.name} added to the {holder.infos.name}")
            elif holder.type == "r":
                print(f"{entity.infos.name} added to the floor in the {holder.infos.name}")
            else:
                print(f"{entity.infos.name} added to the {holder.infos.name} in the {holder.parent.infos.name}")

    def attempt_place_entity(self, name):
        if self.maker.find_by_name(name):
            return
        holder = self.pick_correct_location(self.data.entities[name]["locations"])
        if holder is None:
            return None
        return self.place_at(name, holder)

    def place_entities(self, names):
        return [self.attempt_place_entity(name) for name in names]

    def place_random_entities(self, nb_entities, pool=None):
        rng = self.config.rngs["objects"]
        if pool is None:
            pool = list(self.data.entities.keys())
        if len(pool) == 0:
            return []
        seen = set(e.name for e in self.maker._entities.values())
        candidates = [name for name in pool if name not in seen]
        rng.shuffle(candidates)
        entities = []
        for candidate in candidates:
            if len(entities) >= nb_entities:
                break
            entity = self.attempt_place_entity(candidate)
            if entity:
                entities.append(entity)

        return entities

    def place_random_entity(self, pool):
        entities = self.place_random_entities(1, pool)
        return entities[0] if entities else None

    def place_random_furniture(self, nb_furniture):
        return self.place_random_entities(nb_furniture, self.data.locations.keys())

    def make_graph_map(self, rooms, size=(5, 5)):
        rng = self.config.rngs["map"]
        walker = RandomWalk(neighbors=self.data.map, size=size, rng=rng)
        return walker.place_rooms(rooms)

    def create_map(self, rooms_to_place):
        graph = self.make_graph_map(rooms_to_place)
        rooms = self.maker.import_graph(graph)

        for infos in self.data.doors:
            room1 = self.maker.find_by_name(infos["path"][0])
            room2 = self.maker.find_by_name(infos["path"][1])
            if room1 is None or room2 is None:
                continue  # This door doesn't exist in this world.
            path = self.maker.find_path(room1, room2)
            if path:
                assert path.door is None
                name = self.pick_name(infos["names"])
                door = self.maker.new_door(path, name)
                door.add_property("closed")
        return rooms

    def find_correct_locations(self, obj):
        name = obj.infos.name
        locations = self.data.objects[name]["locations"]
        result = []
        for location in locations:
            if "." in location:
                room_name = location.split(".")[0]
                holder_name = location.split(".")[1]
                room = self.maker.find_by_name(room_name)
                if room is not None:
                    result += [e for e in room.content if e.infos.name == holder_name]
            else:
                holder = self.maker.find_by_name(location)
                if holder:
                    result.append(holder)
        return result

    def generate_quest(self, obj):
        quests = []
        locations = self.find_correct_locations(obj)
        assert len(locations) > 0
        conditions = [self.maker.new_fact(preposition_of(location), obj, location) for location in locations]
        events = [Event(conditions={c}) for c in conditions]
        place_quest = Quest(win_events=events, reward=self.config.reward)
        quests.append(place_quest)
        if self.config.intermediate_reward > 0:
            current_location = obj.parent
            if current_location == self.maker.inventory:
                return quests
            take_cond = self.maker.new_fact('in', obj, self.maker.inventory)
            events = [Event(conditions={take_cond})]
            take_quest = Quest(win_events=events, reward=int(self.config.intermediate_reward))
            quests.append(take_quest)
        return quests

    def generate_goal_locations(self, objs):
        result = {obj.infos.name: [] for obj in objs}
        for obj in objs:
            locations = self.find_correct_locations(obj)
            for loc in locations:
                result[obj.infos.name].append(loc.infos.name)
        return result

    def generate_quests(self, objs):
        return [q for obj in objs for q in self.generate_quest(obj)]

    def set_metadata(self, game, placed_objects):
        game.objective = INTRO + " " + GOAL
        config = dict(vars(self.config))
        del config['game_options']
        del config['rngs']
        metadata = {
            "seeds": self.maker.options.seeds,
            "config": config,
            "entities": [e.name for e in self.maker._entities.values() if e.name],
            "max_score": sum(quest.reward for quest in game.quests),
            "goal": GOAL,
            "goal_locations": self.generate_goal_locations(placed_objects),
            "uuid": self.generate_uuid()
        }
        game.metadata = metadata

    def generate_uuid(self):
        uuid = "tw-iqa-cleanup-{specs}-{seeds}"
        seeds = self.maker.options.seeds
        uuid = uuid.format(specs=prettify_config(self.config),
                           seeds=encode_seeds([seeds[k] + self.num_games for k in sorted(seeds)]))
        return uuid

    def check_properties(self):
        for entity in self.maker._entities.values():
            if entity.type in ["c", "d"] and not \
                    (entity.has_property("closed") or
                     entity.has_property("open") or
                     entity.has_property("locked")):
                raise ValueError("Forgot to add closed/locked/open property for '{}'.".format(entity.name))

    def limit_inventory_size(self):
        inventory_limit = self.config.objects * 2
        nb_objects_in_inventory = self.config.objects - self.config.take
        if self.config.drop:
            inventory_limit = max(1, nb_objects_in_inventory)
        for i in range(inventory_limit):
            slot = self.maker.new(type="slot", name="")
            if i < len(self.maker.inventory.content):
                slot.add_property("used")
            else:
                slot.add_property("free")
            self.maker.nowhere.append(slot)

    def set_container_properties(self):
        if not self.config.open:
            for entity in self.maker._entities.values():
                if entity.has_property("closed"):
                    entity.remove_property("closed")
                    entity.add_property("open")

    def place_distractors(self):
        rng_objects = self.config.rngs["objects"]
        nb_objects = self.config.objects
        if self.config.distractors:
            nb_distractors = max(0, int(rng_objects.randn(1) * 3 + nb_objects))
            self.place_random_entities(nb_distractors, pool=list(self.data.objects.keys()))

    def move_objects(self, placed_objects):
        rng_quest = self.config.rngs["quest"]
        nb_objects_in_inventory = self.config.objects - self.config.take
        shuffled_objects = list(placed_objects)
        rng_quest.shuffle(shuffled_objects)

        for obj in shuffled_objects[:nb_objects_in_inventory]:
            self.maker.move(obj, self.maker.inventory)

        for obj in shuffled_objects[nb_objects_in_inventory:]:
            wrong_location = self.pick_wrong_object_location(obj.infos.name)
            self.maker.move(obj, wrong_location)
            self.log_entity_placement(obj, wrong_location)

        return nb_objects_in_inventory

    def objects_by_furniture(self, furniture):
        result = []
        for o in self.data.objects:
            locations = [loc.split(".")[-1] for loc in self.data.objects[o]["locations"]]
            if furniture in locations:
                result.append(o)
        return result

    def evenly_place_objects(self):
        all_supporters = list(self.maker.findall("s"))
        all_containers = list(self.maker.findall("c"))
        furniture = all_supporters + all_containers

        objects_per_furniture = self.config.objects // len(furniture)

        placed = []
        for holder in furniture:
            pool = self.objects_by_furniture(holder.infos.name)
            placed += self.place_random_entities(objects_per_furniture, pool)

        remainder = self.config.objects - len(placed)
        placed += self.place_random_entities(remainder, list(self.data.objects.keys()))
        return placed

    def place_objects(self, distribute_evenly=True):
        rng = self.config.rngs["objects"]
        if distribute_evenly is None:
            distribute_evenly = rng.choice([True, False])
        if distribute_evenly:
            return self.evenly_place_objects()
        placed_objects = self.place_random_entities(self.config.objects, list(self.data.objects.keys()))
        return placed_objects

    def evenly_place_furniture(self, nb_furniture):
        furniture_per_room = nb_furniture // self.config.rooms
        placed = []
        for room in self.maker.rooms:
            room_name = room.infos.name
            pool = [k for k, v in self.data.locations.items() if room_name in v["locations"]]
            placed += self.place_random_entities(furniture_per_room, pool)

        remainder = nb_furniture - len(placed)
        placed += self.place_random_furniture(remainder)
        return placed

    def place_furniture(self, distribute_evenly=True):
        rng = self.config.rngs["objects"]
        if distribute_evenly is None:
            distribute_evenly = rng.choice([True, False])
        self.place_entities(DEFAULT_FURNITURE)
        upper_bound = max(2 * len(self.maker.rooms), 0.33 * self.config.objects)
        nb_furniture = rng.randint(len(self.maker.rooms), min(upper_bound, len(self.data.locations) + 1))
        if distribute_evenly:
            return self.evenly_place_furniture(nb_furniture)
        else:
            return self.place_random_furniture(nb_furniture)


def illegal_locations(obj):
    forbidden = {'fridge', 'oven', 'dishwasher', 'cutlery drawer', 'trash can', 'grey carpet',
                 'wastepaper basket', 'coat hanger', 'hat rack', 'umbrella stand', 'key holder',
                 'dark carpet', 'toilet', 'pedal bin', 'shower', 'bathtub', 'towel rail',
                 'toilet roll holder', 'bath mat', 'wall hook', 'sink', 'washing machine', 'laundry basket',
                 'clothesline', 'clothes drier'}

    locations = obj["locations"]
    entity_category = obj["category"]
    name = obj["name"]

    if entity_category == "food":
        forbidden.remove('fridge')
        forbidden.add(FLOOR)

    elif entity_category == "clothing":
        forbidden -= {"grey carpet", "dark carpet", "bathtub", "sink", "washing machine", 'laundry basket',
                      'clothesline', 'clothes drier', "bath mat"}
        if not (name.startswith("dirty") or name.startswith("clean") or name.startswith('wet')):
            forbidden |= {"washing machine", 'laundry basket', 'clothesline', 'clothes drier'}
        if "wardrobe" in locations and \
                "bed" not in locations and \
                "chest of drawers" not in locations:
            forbidden -= {"coat hanger", "hat rack", "towel rail", "wall hook"}
        if "hat rack" in locations or "shoe cabinet" in locations:
            forbidden |= {"washing machine", 'laundry basket', 'clothesline', 'clothes drier'}

    elif entity_category == 'kitchenware':
        if name.startswith("clean") or name.startswith("dirty"):
            forbidden -= {"dishwasher", "cutlery drawer"}
        forbidden.add(FLOOR)

    elif entity_category == "object" and "bathroom cabinet" in locations:
        forbidden -= {"toilet", "shower", "bathtub", "bath mat"}

    forbidden -= set(locations)
    return forbidden


def prettify_config(config):
    specs = ["objects", "take", "rooms", "open", "distractors", "drop", "isolated_rooms", "train", "test"]
    dict_config = vars(config)
    specs = [s for s in specs if dict_config[s]]

    def str_value(x):
        return "" if x is True else str(x)

    return "-".join(k + str_value(dict_config[k]) for k in specs)


def preposition_of(entity):
    if entity.type == "r":
        return "at"
    if entity.type in ['c', 'I']:
        return "in"
    if entity.type == "s":
        return "on"
    raise ValueError("Unexpected type {}".format(entity.type))


def play(path):
    env_id = textworld.gym.register_game(path)
    env = gym.make(env_id)
    nb_moves = 0
    score = 0
    try:
        done = False
        obs, _ = env.reset()
        print(obs)
        while not done:
            command = input("> ")
            obs, score, done, _ = env.step(command)
            print(obs)
            nb_moves += 1
    except KeyboardInterrupt:
        pass
    print("Played {} steps, scoring {} points.".format(nb_moves, score))


def main():
    config = twc_config()
    twc_game_maker = TWCGameMaker(config)
    assert config.rooms <= len(twc_game_maker.data.rooms), \
        f"The maximum number of rooms is {len(twc_game_maker.data.rooms)}"
    assert config.objects > 0,\
        "The number of objects should  be greater than 0"
    assert config.take <= config.objects, \
        "The number of objects to find must be less than the total number of objects"
    if config.initial_room:
        assert config.initial_room in twc_game_maker.data.rooms, f"Unknown room {config.initial_room}"
    assert config.intermediate_reward >= 0 and config.reward > 0, \
        "Rewards should be greater than 0"
    assert not (config.play and config.num_games > 1), \
        "You can only play the output game if only 1 game has been created"
    assert config.num_games > 0, "The number of games to generate must be greater than 0"

    if config.verbose:
        print("Config:")
        print("\n".join(f"{k} = {v}" for (k, v) in vars(config).items() if k not in ['game_options', 'rngs']))
        print()
        print("Global seed:", config.seed)
        print()

    game_file = None
    for i in range(config.num_games):
        if config.verbose:
            print(f"Generating game {i + 1}\n")
        game_file = twc_game_maker.make_game()
        if not config.silent:
            print("Game generated: ", game_file)
        if config.verbose:
            print()
        if i + 1 < config.num_games:
            set_difficulty_level(config)

    if config.play:
        play(game_file)


if __name__ == "__main__":
    main()
