import types
import shs_eg

class DummyGraph:
    def __init__(self):
        self._edges = []
        self._nodes = []
    def add_edge(self, a, b):
        self._edges.append((a, b))
        if a not in self._nodes:
            self._nodes.append(a)
        if b not in self._nodes:
            self._nodes.append(b)
    @property
    def edges(self):
        return list(self._edges)
    @property
    def nodes(self):
        return list(self._nodes)
    def subgraph(self, nodes):
        s = set(nodes)
        H = DummyGraph()
        for a, b in self._edges:
            if a in s and b in s:
                H.add_edge(a, b)
        return H

dummy = types.SimpleNamespace(Graph=DummyGraph, subgraph=None)
shs_eg.eg = dummy

G = DummyGraph()
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)
G.add_edge(2, 5)

deg = shs_eg._degree_map(G)
print('deg', deg)
neigh = shs_eg._neighbors_of(G, 2)
print('neigh', sorted(neigh))
H1 = shs_eg._induce_subgraph(G, [1, 2, 3])
print('H1_edges', sorted(H1.edges))
H2 = shs_eg._subgraph_via_api(G, [2, 3, 5])
print('H2_edges', sorted(H2.edges))
