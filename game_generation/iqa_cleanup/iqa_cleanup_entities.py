import json
from pathlib import Path

from iqa_cleanup_config import config, rngs
from random import Random
import numpy as np

FLOOR = "<FLOOR>"
SAMPLE_SEED = 1234
TRAIN_SPLIT = 0.67


def load_json(path):
    with open(path, "r") as json_file:
        return json.load(json_file)


def fill_empty(entity_definitions, tw_type=None, entity_class=None):
    for k, v in entity_definitions.items():
        if "desc" not in v:
            entity_definitions[k]["desc"] = [None]
        if "adjs" not in v:
            entity_definitions[k]["adjs"] = []
        if tw_type and "type" not in v:
            entity_definitions[k]["type"] = tw_type
        if entity_class and "class" not in v:
            entity_definitions[k]["class"] = entity_class
        if "name" not in v:
            entity_definitions[k]["name"] = k
        if "qualifiers" not in v:
            entity_definitions[k]["qualifiers"] = []
        if "modifiers" not in v:
            entity_definitions[k]["modifiers"] = []
        if "properties" not in v:
            entity_definitions[k]["properties"] = []
            if entity_definitions[k]["type"] == "c":
                entity_definitions[k]["properties"].append("closed")


def expand_names(compact_definition):
    result = {}
    for k, v in compact_definition.items():
        if "names" in v:
            for name in v["names"]:
                result[name] = dict(v)
                result[name]["name"] = name
                del result[name]["names"]
        else:
            result[k] = v
    return result


def apply_qualifiers(entity, qualifier_definitions):
    name = entity["name"]
    result = {}
    if "qualifiers" not in entity or not entity["qualifiers"]:
        return {name: entity}
    for qualifier in entity["qualifiers"]:
        definition = qualifier_definitions[qualifier]
        for prefix in definition["name prefix"]:
            prefixed_name = prefix + " " + name
            clone = dict(entity)
            clone["name"] = prefixed_name
            result[prefixed_name] = clone
    return result


def apply_modifiers(entity, modifier_definitions):
    name = entity["name"]
    if "modifiers" not in entity or not entity["modifiers"]:
        return {name: entity}
    modifiers = list(entity["modifiers"])
    modifiers.append(None)
    for modifier in modifiers:
        if modifier and "opposite prefix" in modifier_definitions[modifier]:
            modifiers = modifiers[:-1]
            break
    rng = rngs["objects"]
    modifier = rng.choice(modifiers)
    if not modifier:
        result = {name: dict(entity)}
        return result
    definition = modifier_definitions[modifier]
    return apply_modifier(entity, definition)


def apply_modifier(entity, modifier):
    result = {}
    name = entity["name"]
    opposite_prefix_key = "opposite prefix"

    prefix = modifier["name prefix"]
    prefixed_name = prefix + " " + name
    clone_prefix = dict(entity)
    clone_prefix["name"] = prefixed_name
    for (k, v) in modifier.items():
        if k not in ["name prefix", opposite_prefix_key]:
            clone_prefix[k] = v
    result[prefixed_name] = clone_prefix

    if opposite_prefix_key in modifier:
        opposite_name = modifier[opposite_prefix_key] + " " + name
        opposite_name = opposite_name.strip()
        result[opposite_name] = dict(entity)
        result[opposite_name]["name"] = opposite_name
    return result


def apply_qualifiers_and_modifiers(entities, qualifiers, modifiers):
    result = {}
    for e in entities.values():
        qualified_entities = apply_qualifiers(e, qualifiers)
        for e_qual in qualified_entities.values():
            modified = apply_modifiers(e_qual, modifiers)
            result.update(modified)
    return result


def _sanity_check(entities, rooms):
    for (k, v) in entities.items():
        assert v["name"] == k
        assert "type" in v
        assert "locations" in v
        locations = v["locations"]
        for loc in locations:
            holder = loc.split(".")[-1]
            assert holder in entities or holder in rooms, \
                f"Unknown location '{holder}' for entity '{k}'"
            if "." in loc:
                room = loc.split(".")[0]
                assert loc.count(".") == 1, f"Could not parse location {loc}"
                assert room in rooms
                assert holder not in rooms
                assert room in entities[holder]["locations"], \
                    f"Unexpected combination '{loc}' for entity '{k}'"


def sample(entities):
    assert not (config.train and config.test)
    if not config.train and not config.test:
        return entities
    rng_sample = Random(SAMPLE_SEED)
    keys = list(entities.keys())
    rng_sample.shuffle(keys)
    split = round(len(keys) * TRAIN_SPLIT)
    if config.train:
        sampled_keys = keys[:split]
    else:
        sampled_keys = keys[split:]
    return {k: entities[k] for k in sampled_keys}


NEIGHBORS = load_json(Path(config.data_path) / "map.json")
FURNITURE = load_json(Path(config.data_path) / "furniture.json")
DOORS = load_json(Path(config.data_path) / "doors.json")
OBJECTS = load_json(Path(config.data_path) / "objects.json")
FOOD = load_json(Path(config.data_path) / "food.json")
KITCHENWARE = load_json(Path(config.data_path) / "kitchenware.json")
CLOTHING = load_json(Path(config.data_path) / "clothing.json")
MODIFIERS = load_json(Path(config.data_path) / "modifiers.json")
QUALIFIERS = load_json(Path(config.data_path) / "qualifiers.json")
ROOMS = list(NEIGHBORS.keys())

fill_empty(FURNITURE, entity_class="furniture")
fill_empty(OBJECTS, tw_type="o", entity_class="object")
fill_empty(KITCHENWARE, tw_type="o", entity_class="kitchenware")
fill_empty(FOOD, tw_type="f", entity_class="food")
fill_empty(CLOTHING, tw_type="o", entity_class="clothing")
FOOD = expand_names(FOOD)


def total_number_of_variations(entity_infos):
    return sum(number_of_variations(e, entity_infos) for e in entity_infos)


def number_of_variations(entity, entity_infos):
    infos = entity_infos[entity]
    nmodifiers = 1
    nqualifiers = 1
    if "modifiers" in infos:
        nmodifiers = 1 + len(infos["modifiers"])
    if "qualifiers" in infos:
        nqualifiers = max(1, int(np.prod([max(1, len(QUALIFIERS[q]["name prefix"])) for q in infos["qualifiers"]])))
    return nqualifiers * nmodifiers

# ## REMOVE BELOW
# OBJECTS.update(FOOD)
# OBJECTS.update(CLOTHING)
# OBJECTS.update(KITCHENWARE)
# ENTITIES = dict(OBJECTS)
# ENTITIES.update(FURNITURE)
#
# print("UNIQUE ENTS: ", len(ENTITIES))
# print("FURNITURE: ", len(FURNITURE))
# print("UNIQUE OBJS: ", len(OBJECTS))
# print("TOTAL ENTS: ", total_number_of_variations(ENTITIES))
# print("TOTAL OBJS: ", total_number_of_variations(OBJECTS))
#
#
# ## END REMOVE


FOOD = sample(FOOD)
CLOTHING = sample(CLOTHING)
KITCHENWARE = sample(KITCHENWARE)
OBJECTS = sample(OBJECTS)

OBJECTS.update(FOOD)
OBJECTS.update(CLOTHING)
OBJECTS.update(KITCHENWARE)
OBJECTS = apply_qualifiers_and_modifiers(OBJECTS, QUALIFIERS, MODIFIERS)
ENTITIES = dict(OBJECTS)
ENTITIES.update(FURNITURE)

DEFAULT_FURNITURE = ["dining table", "fridge", "kitchen cupboard", "bed", "wardrobe", "sofa", "toilet",
                     "shower", "sink", "BBQ", "washing machine", "clothesline"]

_sanity_check(ENTITIES, ROOMS)
