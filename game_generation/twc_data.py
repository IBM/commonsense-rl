from pathlib import Path
import json
from random import Random


def load_json(path):
    with open(path, "r") as json_file:
        return json.load(json_file)


class TWCData:
    def __init__(self, config):
        self.map = load_json(Path(config.data_path) / "twc_map.json")
        self.locations = load_json(Path(config.data_path) / "twc_locations.json")
        self.doors = load_json(Path(config.data_path) / "twc_doors.json")
        self.objects = load_json(Path(config.data_path) / "twc_objects.json")
        self.rooms = list(self.map.keys())

        self.fill_empty(self.locations)
        self.fill_empty(self.objects)

        self.sample_objects(config)

        self.entities = dict(self.objects)
        self.entities.update(self.locations)

        self._sanity_check()

    def sample_objects(self, config):
        assert not (config.train and config.test)
        if not config.train and not config.test:
            return
        rng_sample = Random(config.train_distribution_seed)
        entities = []
        for k, obj in self.objects.items():
            if not obj['entity'] in entities:
                entities.append(obj['entity'])

        rng_sample.shuffle(entities)
        split = round(len(entities) * config.train_size)
        sampled_entities = entities[:split] if config.train else entities[split:]
        sampled_entities = set(sampled_entities)
        self.objects = {k: o for k, o in self.objects.items() if o['entity'] in sampled_entities}

    def _sanity_check(self):
        for (k, v) in self.entities.items():
            assert v["name"] == k
            assert "type" in v
            assert "locations" in v
            locations = v["locations"]
            for loc in locations:
                holder = loc.split(".")[-1]
                assert holder in self.entities or holder in self.rooms, \
                    f"Unknown location '{holder}' for entity '{k}'"
                if "." in loc:
                    room = loc.split(".")[0]
                    assert loc.count(".") == 1, f"Could not parse location {loc}"
                    assert room in self.rooms
                    assert holder not in self.rooms
                    assert room in self.entities[holder]["locations"], \
                        f"Unexpected combination '{loc}' for entity '{k}'"

    @staticmethod
    def fill_empty(entity_definitions):
        for k, v in entity_definitions.items():
            if "desc" not in v:
                entity_definitions[k]["desc"] = [None]
            if "adjs" not in v:
                entity_definitions[k]["adjs"] = []
            if "properties" not in v:
                entity_definitions[k]["properties"] = []
                if entity_definitions[k]["type"] == "c":
                    entity_definitions[k]["properties"].append("closed")
