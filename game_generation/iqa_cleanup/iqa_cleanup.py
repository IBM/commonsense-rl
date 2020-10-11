import textworld
from textworld.utils import encode_seeds
from textworld.generator.game import Quest, Event, Game
from iqa_cleanup_entities import ROOMS, DOORS, NEIGHBORS, FURNITURE, OBJECTS, ENTITIES, FLOOR, DEFAULT_FURNITURE
from iqa_cleanup_config import config, game_options, rngs
from random_walk import RandomWalk
import os.path
from os.path import join as pjoin
import gym
import textworld.gym
from textworld import GameMaker
import networkx as nx
from pprint import pprint

INTRO = "Welcome to TextWorld! You find yourself in a messy house. Many things are not in their usual location. " \
        "Let's clean up this place. After you'll have done, this little house is going to be spick and span!"
GOAL = "Look for anything that is out of place and put it away in its proper location."


def pick_name(maker, names):
    rng = rngs["objects"]
    names = list(names)
    rng.shuffle(names)
    for name in names:
        if maker.find_by_name(name) is None:
            return name
    assert False


def pick_rooms(start):
    assert config.rooms <= len(ROOMS)
    rng = rngs["map"]
    visited = {start}
    neighbors = set(NEIGHBORS[start])
    neighbors -= visited

    while len(visited) < config.rooms:
        room = rng.choice(list(neighbors))
        visited.add(room)
        neighbors |= set(NEIGHBORS[room])
        neighbors -= visited

    return list(visited)


def pick_correct_location(maker, locations):
    rng = rngs["objects"]
    locations = list(locations)
    rng.shuffle(locations)
    for location in locations:
        holder = None
        if "." in location:
            room_name = location.split(".")[0]
            holder_name = location.split(".")[1]
            room = maker.find_by_name(room_name)
            if room is not None:
                holder = next((e for e in room.content if e.infos.name == holder_name), None)
        else:
            holder = maker.find_by_name(location)
        if holder:
            return holder
    return None


def pick_wrong_object_location(maker, object_name, prefer_correct_room=None):
    rng = rngs["objects"]
    correct_locations = OBJECTS[object_name]["locations"]
    rng.shuffle(correct_locations)

    holder_names = {location.split(".")[-1] for location in correct_locations}
    forbidden = illegal_locations(object_name)
    holder_names |= forbidden

    if prefer_correct_room is None:
        prefer_correct_room = config.isolated_rooms

    assert prefer_correct_room in [True, False]
    correct_room = find_correct_room(maker, object_name)

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
    all_supporters = list(maker.findall("s"))
    all_containers = list(maker.findall("c"))
    all_rooms = maker.rooms

    all_holders = all_supporters + all_containers
    if FLOOR not in holder_names:
        all_holders += all_rooms

    rng.shuffle(all_holders)
    wrong_holders = [e for e in all_holders if e.infos.name not in holder_names]

    if len(wrong_holders) > 0:
        return wrong_holders[0]

    # No wrong location found. Create new furniture
    pool = [f for f in FURNITURE.keys() if f not in holder_names]
    return place_random_entity(maker, pool)


def illegal_locations(obj):
    forbidden = {'fridge', 'oven', 'dishwasher', 'cutlery drawer', 'trash can', 'grey carpet',
                 'wastepaper basket', 'coat hanger', 'hat rack', 'umbrella stand', 'key holder',
                 'dark carpet', 'toilet', 'pedal bin', 'shower', 'bathtub', 'towel rail',
                 'toilet roll holder', 'bath mat', 'wall hook', 'sink', 'washing machine', 'laundry basket',
                 'clothesline', 'clothes drier'}

    obj = OBJECTS[obj]
    locations = obj["locations"]
    entity_class = obj["class"]

    if entity_class == "food":
        forbidden.remove('fridge')
        forbidden.add(FLOOR)

    elif entity_class == "clothing":
        forbidden -= {"grey carpet", "dark carpet", "bathtub", "sink", "washing machine", 'laundry basket',
                      'clothesline', 'clothes drier', "bath mat"}
        if "dirty clothing" not in obj["modifiers"] or "wet clothing" not in obj["modifiers"]:
            forbidden |= {"washing machine", 'laundry basket', 'clothesline', 'clothes drier'}
        if "wardrobe" in locations and \
                "bed" not in locations and \
                "chest of drawers" not in locations:
            forbidden -= {"coat hanger", "hat rack", "towel rail", "wall hook"}
        if "hat rack" in locations or "shoe cabinet" in locations:
            forbidden |= {"washing machine", 'laundry basket', 'clothesline', 'clothes drier'}

    elif entity_class == 'kitchenware':
        name = obj["name"]
        if name.startswith("clean") or name.startswith("dirty"):
            forbidden -= {"dishwasher", "cutlery drawer"}
        forbidden.add(FLOOR)

    elif entity_class == "object" and "bathroom cabinet" in locations:
        forbidden -= {"toilet", "shower", "bathtub", "bath mat"}

    forbidden -= set(locations)
    return forbidden


def find_correct_room(maker, object_name):
    correct_locations = OBJECTS[object_name]["locations"]
    holder_names = {location.split(".")[-1] for location in correct_locations}

    for location in correct_locations:
        if "." in location:
            room_name = location.split(".")[0]
            return maker.find_by_name(room_name)
    for holder_name in holder_names:
        holder = maker.find_by_name(holder_name)
        if holder:
            return holder.parent


def place_at(maker, name, holder):
    entity = maker.new(type=ENTITIES[name]["type"], name=name)
    entity.infos.noun = name
    if "adjs" in ENTITIES[name] and ENTITIES[name]["adjs"]:
        entity.infos.adj = ENTITIES[name]["adjs"][0]
    if "desc" in ENTITIES[name]:
        entity.infos.desc = ENTITIES[name]["desc"][0]
    if "indefinite" in ENTITIES[name]:
        entity.infos.indefinite = ENTITIES[name]["indefinite"]
    for property_ in ENTITIES[name]["properties"]:
        entity.add_property(property_)
    holder.add(entity)
    log_entity_placement(entity, holder)
    return entity


def log_entity_placement(entity, holder):
    name = entity.infos.name
    if config.verbose:
        if ENTITIES[name]["class"] == "furniture":
            print(f"{entity.infos.name} added to the {holder.infos.name}")
        elif holder.type == "r":
            print(f"{entity.infos.name} added to the floor in the {holder.infos.name}")
        else:
            print(f"{entity.infos.name} added to the {holder.infos.name} in the {holder.parent.infos.name}")


def attempt_place_entity(maker, name):
    holder = pick_correct_location(maker, ENTITIES[name]["locations"])
    if holder is None:
        return None
    return place_at(maker, name, holder)


def place_entities(maker, names):
    return [attempt_place_entity(maker, name) for name in names]


def place_random_entities(maker, nb_entities, pool=None):
    rng = rngs["objects"]
    if pool is None:
        pool = list(ENTITIES.keys())
    if len(pool) == 0:
        return []
    seen = set(e.name for e in maker._entities.values())
    candidates = [name for name in pool if name not in seen]
    rng.shuffle(candidates)
    entities = []
    for candidate in candidates:
        if len(entities) >= nb_entities:
            break
        entity = attempt_place_entity(maker, candidate)
        if entity:
            entities.append(entity)

    return entities


def place_random_entity(maker, pool):
    entities = place_random_entities(maker, 1, pool)
    return entities[0] if entities else None


def place_random_furniture(maker, nb_furniture):
    return place_random_entities(maker, nb_furniture, FURNITURE.keys())


def make_graph_map(rooms, size=(5, 5)):
    rng = rngs["map"]
    walker = RandomWalk(neighbors=NEIGHBORS, size=size, rng=rng)
    return walker.place_rooms(rooms)


def create_map(maker, rooms_to_place):
    graph = make_graph_map(rooms_to_place)
    rooms = maker.import_graph(graph)

    for infos in DOORS:
        room1 = maker.find_by_name(infos["path"][0])
        room2 = maker.find_by_name(infos["path"][1])
        if room1 is None or room2 is None:
            continue  # This door doesn't exist in this world.

        path = maker.find_path(room1, room2)
        if path:
            assert path.door is None
            name = pick_name(maker, infos["names"])
            door = maker.new_door(path, name)
            door.add_property("closed")
    return rooms


def find_correct_locations(maker, obj):
    name = obj.infos.name
    locations = OBJECTS[name]["locations"]
    result = []
    for location in locations:
        if "." in location:
            room_name = location.split(".")[0]
            holder_name = location.split(".")[1]
            room = maker.find_by_name(room_name)
            if room is not None:
                result += [e for e in room.content if e.infos.name == holder_name]
        else:
            holder = maker.find_by_name(location)
            if holder:
                result.append(holder)
    return result


def preposition_of(entity):
    if entity.type == "r":
        return "at"
    if entity.type in ['c', 'I']:
        return "in"
    if entity.type == "s":
        return "on"
    raise ValueError("Unexpected type {}".format(entity.type))


def generate_quest(maker, obj):
    locations = find_correct_locations(maker, obj)
    assert len(locations) > 0
    conditions = [maker.new_fact(preposition_of(location), obj, location) for location in locations]
    events = [Event(conditions={c}) for c in conditions]
    return Quest(win_events=events)


def generate_goal_locations(maker, objs):
    result = {obj.infos.name: [] for obj in objs}
    for obj in objs:
        locations = find_correct_locations(maker, obj)
        for loc in locations:
            result[obj.infos.name].append(loc.infos.name)
    return result


def generate_quests(maker, objs):
    return [generate_quest(maker, obj) for obj in objs]


def set_metadata(maker, game, placed_objects):
    game.objective = INTRO + " " + GOAL
    metadata = {
        "seeds": maker.options.seeds,
        "config": vars(config),
        "entities": [e.name for e in maker._entities.values() if e.name],
        "max_score": sum(quest.reward for quest in game.quests),
        "goal": GOAL,
        "goal_locations": generate_goal_locations(maker, placed_objects),
        "uuid": generate_uuid(maker)
    }
    game.metadata = metadata


def generate_uuid(maker):
    uuid = "tw-iqa-cleanup-{specs}-{seeds}"
    seeds = maker.options.seeds
    uuid = uuid.format(specs=prettify_config(),
                       seeds=encode_seeds([seeds[k] for k in sorted(seeds)]))
    return uuid


def check_properties(maker):
    for entity in maker._entities.values():
        if entity.type in ["c", "d"] and not \
                (entity.has_property("closed") or
                 entity.has_property("open") or
                 entity.has_property("locked")):
            raise ValueError("Forgot to add closed/locked/open property for '{}'.".format(entity.name))


def limit_inventory_size(maker):
    inventory_limit = config.objects * 2
    nb_objects_in_inventory = config.objects - config.take
    if config.drop:
        inventory_limit = max(1, nb_objects_in_inventory)
    for i in range(inventory_limit):
        slot = maker.new(type="slot", name="")
        if i < len(maker.inventory.content):
            slot.add_property("used")
        else:
            slot.add_property("free")
        maker.nowhere.append(slot)


def set_container_properties(maker):
    if not config.open:
        for entity in maker._entities.values():
            if entity.has_property("closed"):
                entity.remove_property("closed")
                entity.add_property("open")


def place_distractors(maker):
    rng_objects = rngs["objects"]
    nb_objects = config.objects
    if config.distractors:
        nb_distractors = max(0, int(rng_objects.randn(1) * 3 + nb_objects))
        place_random_entities(maker, nb_distractors, pool=list(OBJECTS.keys()))


def move_objects(maker, placed_objects):
    rng_quest = rngs["quest"]
    nb_objects_in_inventory = config.objects - config.take
    shuffled_objects = list(placed_objects)
    rng_quest.shuffle(shuffled_objects)

    for obj in shuffled_objects[:nb_objects_in_inventory]:
        maker.move(obj, maker.inventory)

    for obj in shuffled_objects[nb_objects_in_inventory:]:
        wrong_location = pick_wrong_object_location(maker, obj.infos.name)
        maker.move(obj, wrong_location)
        log_entity_placement(obj, wrong_location)

    return nb_objects_in_inventory


def objects_by_furniture(furniture):
    result = []
    for o in OBJECTS:
        locations = [loc.split(".")[-1] for loc in OBJECTS[o]["locations"]]
        if furniture in locations:
            result.append(o)
    return result


def evenly_place_objects(maker):
    all_supporters = list(maker.findall("s"))
    all_containers = list(maker.findall("c"))
    furniture = all_supporters + all_containers

    objects_per_furniture = config.objects // len(furniture)

    placed = []
    for holder in furniture:
        pool = objects_by_furniture(holder.infos.name)
        placed += place_random_entities(maker, objects_per_furniture, pool)

    remainder = config.objects - len(placed)
    placed += place_random_entities(maker, remainder, list(OBJECTS.keys()))
    return placed


def place_objects(maker, distribute_evenly=None):
    rng = rngs["objects"]
    if distribute_evenly is None:
        distribute_evenly = rng.choice([True, False])
    if distribute_evenly:
        return evenly_place_objects(maker)
    placed_objects = place_random_entities(maker, config.objects, list(OBJECTS.keys()))
    return placed_objects


def evenly_place_furniture(maker, nb_furniture):
    furniture_per_room = nb_furniture // config.rooms
    placed = []
    for room in maker.rooms:
        room_name = room.infos.name
        pool = [k for k, v in FURNITURE.items() if room_name in v["locations"]]
        placed += place_random_entities(maker, furniture_per_room, pool)

    remainder = nb_furniture - len(placed)
    placed += place_random_furniture(maker, remainder)
    return placed


def place_furniture(maker, distribute_evenly=None):
    rng = rngs["objects"]
    if distribute_evenly is None:
        distribute_evenly = rng.choice([True, False])
    place_entities(maker, DEFAULT_FURNITURE)
    nb_furniture = rng.randint(len(maker.rooms), len(FURNITURE) + 1)
    if distribute_evenly:
        return evenly_place_furniture(maker, nb_furniture)
    else:
        return place_random_furniture(maker, nb_furniture)


def place_rooms(maker):
    rng = rngs["map"]
    assert config.rooms <= len(ROOMS)
    if not config.initial_room:
        config.initial_room = rng.choice(ROOMS)
    rooms_to_place = pick_rooms(config.initial_room)
    if config.verbose:
        print("Rooms:", rooms_to_place)
    create_map(maker, rooms_to_place)
    room = maker.find_by_name(config.initial_room)
    maker.set_player(room)


def prettify_config():
    specs = ["objects", "take", "rooms", "open", "distractors", "drop", "isolated_rooms", "train", "test"]
    dict_config = vars(config)
    specs = [s for s in specs if dict_config[s]]

    def str_value(x):
        return "" if x is True else str(x)

    return "-".join(k + str_value(dict_config[k]) for k in specs)


def make_game() -> textworld.Game:
    maker = GameMaker(game_options)
    rng_grammar = rngs["grammar"]
    maker.grammar = textworld.generator.make_grammar(maker.options.grammar, rng=rng_grammar)

    place_rooms(maker)

    if config.verbose:
        print()
        print("====== Placing furniture ======")

    place_furniture(maker)

    if config.verbose:
        print()
        print("====== Placing objects ======")

    placed_objects = place_objects(maker)

    if config.verbose:
        print()
        print("====== Shuffling objects ======")

    move_objects(maker, placed_objects)

    if config.verbose and config.distractors:
        print()
        print("====== Placing distractors ======")

    place_distractors(maker)

    set_container_properties(maker)
    limit_inventory_size(maker)
    maker.quests = generate_quests(maker, placed_objects)

    check_properties(maker)

    uuid = generate_uuid(maker)
    game = maker.build()

    set_metadata(maker, game, placed_objects)

    if config.verbose:
        print()
        print("====== Goal Locations ======")
        for obj, locations in game.metadata["goal_locations"].items():
            print(f'{obj} ->', ", ".join(locations))

    game_options.path = pjoin(config.output_dir, uuid)

    return textworld.generator.compile_game(game, game_options)


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
    assert config.rooms <= len(ROOMS)
    assert config.objects > 0
    assert config.take <= config.objects
    if config.initial_room:
        assert config.initial_room in ROOMS

    if config.verbose:
        print("Config:")
        print("\n".join(f"{k} = {v}" for (k, v) in vars(config).items()))
        print()
        print("Global seed:", config.seed)
        print()

    game_file = make_game()

    if not config.silent:
        print("Game generated: ", game_file)

    if config.play:
        play(game_file)


if __name__ == "__main__":
    main()
