import networkx as nx
import pandas as pd
import numpy as np
from scipy.linalg import expm
import argparse
parser = argparse.ArgumentParser(description='Network dissimilarity based on communicability sequence')
# general settings
parser.add_argument('--data_name1', default=None, help='network name')
parser.add_argument('--data_name2', default=None, help='network name')
parser.add_argument('--network1_size', default=None, help='network size')
parser.add_argument('--network2_size', default=None, help='network size')
args = parser.parse_args()

# calculate the entropia of the sequence
def Entropia(a):
    b = np.where(a > 0)
    return -np.sum(a[b]*np.log(a[b]))

# calculate the communicability sequence
def Get_Sequence(G, network_size):
    A = nx.to_numpy_array(G, nodelist=list(range(network_size)))
    C = expm(A)
    sequence = np.array(sorted([C[i][j] for i in range(network_size) for j in range(network_size) if j >= i]))
    sequence = sequence / np.sum(sequence)
    return sequence

# calculate the dissimilarity between two networks g1 and g2
def Cal_Dissimility(g1, g2):
    sequence1 = Get_Sequence(g1, args.network1_size)
    sequence2 = Get_Sequence(g2, args.network2_size)
    if len(sequence1) < len(sequence2):
        add_number = len(sequence2) - len(sequence1)
        sequence1 = np.concatenate([np.zeros(add_number), sequence1])
    else:
        add_number = len(sequence1) - len(sequence2)
        sequence2 = np.concatenate([np.zeros(add_number), sequence2])
    sequence = (sequence1 + sequence2) / 2
    d_value = Entropia(sequence) - (Entropia(sequence1) + Entropia(sequence2)) / 2
    return d_value

# construct the network
def Construct_Network(path, flag):
    edgedata = pd.read_csv(path, header=None)
    g = nx.Graph()
    g.add_edges_from(zip(edgedata[:][0], edgedata[:][1]))
    network_size = max(g.nodes)+1
    g.add_nodes_from(range(network_size))
    if flag == 1:
        args.network1_size = network_size
    else:
        args.network2_size = network_size
    return g

if __name__ == '__main__':
    g1 = Construct_Network('../data/%s.txt' % args.data_name1, 1)
    g2 = Construct_Network('../data/%s.txt' % args.data_name2, 2)
    print(Cal_Dissimility(g1, g2))
