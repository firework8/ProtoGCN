import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


def plot_topology_matrix(topology_matrix, save_name):
    
    plt.imshow(topology_matrix, interpolation='nearest',cmap=plt.cm.GnBu)
    plt.colorbar()   
    
    tick_marks = np.arange(25)
    plt.xticks(tick_marks, tick_marks, fontsize=8)
    plt.yticks(tick_marks, tick_marks, fontsize=8)
    
    plt.show()
    plt.savefig(save_name)
    plt.clf()


def main():
    # Example:
    # Below the graph topology of the id=0 sample is visualised.
    get_graph = np.load('graph.npy', allow_pickle=True)
    vis_graph = get_graph[0]
    save_name = 'vis_graph.jpg'
    plot_topology_matrix(vis_graph, save_name)


if __name__ == '__main__':
    main()
