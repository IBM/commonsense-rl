import networkx as nx
import csv
from utils.kg import add_triplets_to_graph, draw_graph, construct_kg


def shortest_path_subgraph(kg_graph, prev_graph, nodes, path_len=2, add_all_path=False):
    # Get non-neighbor nodes: nodes without edges between them
    new_graph = kg_graph.subgraph(nodes).copy()
    new_graph.remove_edges_from(nx.selfloop_edges(new_graph))
    world_graph = nx.compose(prev_graph, new_graph)
    if path_len < 2:
        return world_graph
    triplets = []
    sg_nodes = list(nx.isolates(world_graph))
    for source in sg_nodes:
        nb = nx.single_source_shortest_path(kg_graph, source, cutoff=path_len)
        if add_all_path:
            triplets.extend([[source, target, ' '.join(nb[target][1:-1])] for target in nodes if
                             source != target and target in nb and len(nb[target]) == path_len + 1])
        else:
            for target in nodes: # Just add one link
                if source != target and target in nb and len(nb[target]) == path_len + 1:
                    triplets.append([source, target, ' '.join(nb[target][1:-1])])
                    break
    add_triplets_to_graph(world_graph, triplets)

    return world_graph

def get_sample_states_from_file():
    action_scores = []

    with open('sample_easy_state_texts.txt', 'r') as fout:
        states = fout.readlines()
    with open('sample_easy_concepts.txt', 'r') as fout:
        entities = fout.readlines()
    with open('sample_easy_action_score_texts.txt', 'r', newline="") as fout:
        csv_reader = csv.reader(fout)
        for line in csv_reader:
            action_scores.append(line)

    return states, entities, action_scores

def get_sample_states():
    texts = ["You've just walked into a Living Room. You try to gain information on your " \
           "surroundings by using a technique you call looking. You can see a closet. " \
           "You idly wonder how they came up with the name TextWorld for this place. " \
           "It's pretty fitting. A closed standard looking antique trunk is in the room. " \
           "You can see a table. The table is usual. On the table you see an apple, a mug, " \
           "a newspaper, a note, a hat and a pencil. You smell a sickening smell, and follow " \
           "it to a couch. The couch is standard. But the thing is empty. Hm. Oh well You see a " \
           "gleam over in a corner, where you can see a tv stand. The tv stand is ordinary. " \
           "On the tv stand you can make out a tv. You don't like doors? Why not try going east, " \
           "that entranceway is unguarded. You are carrying nothing.",
             "On the table, you see an apple, a hat, a key and an umbrella."]
    return texts


if __name__ == '__main__':
    from utils import extractor
    from utils.nlp import Tokenizer

    tk_extractor = extractor.get_extractor('max')
    tokenizer = Tokenizer(extractor=tk_extractor)
    kg_graph, _, _ = construct_kg('./kg/conceptnet/conceptnet_subgraph.txt')
    states, entities, action_scores = get_sample_states_from_file()

    # entities = tokenizer.extract_world_graph_entities(states, kg_graph)
    world_graph = nx.DiGraph()
    for step_entities in entities:
        world_graph = shortest_path_subgraph(kg_graph, world_graph, step_entities)

    print(world_graph.edges())
    draw_graph(world_graph)
