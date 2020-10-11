from textworld import Game
from pathlib import Path


def extract_entities(games):
    result = set()
    for g in games:
        for entity in g.infos.values():
            if entity.type == 'o' or entity.type == 'f':
                result.add(entity.name)
    return result


def test_train_test_intersection():
    path = Path("iqa_cleanup_games")
    test = []
    train = []
    for p in path.rglob("*.json"):
        game = Game.load(p)
        if '-test-' in str(p):
            test.append(game)
        elif '-train-' in str(p):
            train.append(game)

    train_entities = extract_entities(train)
    test_entities = extract_entities(test)

    intersection = train_entities & test_entities

    print()
    print('Number of training games:', len(train))
    print('Number of test games:', len(test))
    print('Number of training entities:', len(train_entities))
    print('Number of test entities:', len(test_entities))

    assert len(intersection) == 0
