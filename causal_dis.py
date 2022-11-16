import networkx as nx
import matplotlib.pyplot as plt
import stack
from copy import deepcopy
import dowhy 


G = nx.DiGraph()
G.add_nodes_from(['a', 'b', 3, 'd','c'])
G.add_edges_from([('a', 'b'), ('a', 3),('d','a'),('c','b')])
print(list(G.predecessors('d')))   #父节点

G['a']['b']['weight'] = 1
G['a'][3]['weight'] = 2
G['d']['a']['weight'] = 3
G['c']['b']['weight'] = 4
G['c']['b']['weight'] = 5
labels = nx.get_edge_attributes(G,'weight')
pos = nx.spring_layout(G)
fig=plt.figure()
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
print(1 == 1 and 2 == 2)