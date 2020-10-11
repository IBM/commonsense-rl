from textworld import Game
from pathlib import Path

PATH = "../../games/ftwp/"


def extract_entities(games):
    result = set()
    for g in games:
        for entity in g.infos.values():
            result.add((entity.name, entity.type))
    return result


def load_games(path):
    path = Path(path)
    files = path.rglob("*.json")
    return [Game.load(f) for f in files]


def main():
    games = load_games(PATH)
    entities = extract_entities(games)
    for name, t in entities:
        if name:
            print(f'{name}, {t}')


if __name__ == '__main__':
    main()
