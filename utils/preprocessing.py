from utils.textworld_utils import load_games, extract_entities
from utils.kg import construct_kg, kg_match, shortest_path_seed_expansion, ego_graph_seed_expansion, save_graph_tsv
from utils.extractor import any_substring_extraction
import networkx as nx
from pathlib import Path

CLEANUP_GAMES_PATH = Path('./games/cleanup')
KG_PATH = '../kg.txt'


def extract_games_knowledge_subgraph(kg, games_path):
    games = load_games(games_path)
    target_entities = {e[0] for e in extract_entities(games)}
    linked_entities = kg_match(any_substring_extraction, target_entities, set(kg.nodes))
    sp_subgraph = shortest_path_seed_expansion(kg, linked_entities, cutoff=5)
    ego_subgraph = ego_graph_seed_expansion(kg, linked_entities, radius=1)
    return nx.compose(ego_subgraph, sp_subgraph)


def extract_cleanup_kg_subgraphs(kg):
    for level_dir in ['easy', 'medium', 'hard']:
        for split_dir in ['train', 'test', 'valid']:
            path = CLEANUP_GAMES_PATH / level_dir / split_dir
            print('Extracting conceptnet subgraph for', path)
            subgraph = extract_games_knowledge_subgraph(kg, path)
            print(f'Nodes: {len(subgraph.nodes)}  -  Edges: {len(subgraph.edges)}\n')
            save_graph_tsv(subgraph, path / "conceptnet_subgraph.txt")


def extract_cleanup_entities():
    for level_dir in ['easy', 'medium', 'hard']:
        for split_dir in ['train', 'test', 'valid']:
            path = CLEANUP_GAMES_PATH / level_dir / split_dir
            games = load_games(path)
            named_entities = [e[0] for e in extract_entities(games)]
            out_path = path / 'entities.txt'
            print('Saving', out_path)
            with open(out_path, 'w') as f:
                f.writelines([f'{e}\n' for e in named_entities])


def save_command_templates():
    templates = "examine {t}\n" \
                "go east\n" \
                "go north\n" \
                "go south\n" \
                "go west\n" \
                "insert {o} into {c}\n" \
                "inventory\n" \
                "put {o} on {s}\n" \
                "take {o}\n"

    for level_dir in ['easy', 'medium', 'hard']:
        for split_dir in ['train', 'test', 'valid']:
            path = CLEANUP_GAMES_PATH / level_dir / split_dir / 'command_templates.txt'
            print('Saving', path)
            with open(path, 'w') as f:
                f.writelines(templates)


if __name__ == '__main__':
    kg, _, _ = construct_kg(KG_PATH)
    extract_cleanup_kg_subgraphs(kg)
    extract_cleanup_entities()
    save_command_templates()
