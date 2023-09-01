import matplotlib.pyplot as plt
import networkx as nx

"""plotting the graph using matplotlib"""


def draw_graph(g, lb, edge_lb):
    pos = nx.spring_layout(g, k=0.2, pos=None, fixed=None, iterations=150,
                           threshold=0.01, weight='weight', scale=2, center=None, dim=2, seed=None)
    plt.figure(figsize=(20, 20))
    nx.draw(g, pos=pos, labels=lb, node_color="yellow", with_labels=True)
    nx.draw_networkx_edges(g, pos=pos, arrowstyle="->",
                           arrowsize=10, width=1)
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_lb, font_color='red')
    plt.show()
