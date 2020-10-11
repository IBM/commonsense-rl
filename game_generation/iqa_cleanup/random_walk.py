import numpy as np
import networkx as nx


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
