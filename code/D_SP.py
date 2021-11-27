import networkx as nx
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Network dissimilarity based on shortest path distance distribution')
# general settings
parser.add_argument('--data_name1', default=None, help='network name')
parser.add_argument('--data_name2', default=None, help='network name')
parser.add_argument('--network1_size', default=None, help='network size')
parser.add_argument('--network2_size', default=None, help='network size')

parser.add_argument('--w1', type=float, default=0.45,
                    help='tunable parameter')
parser.add_argument('--w2', type=float, default=0.45,
                    help='tunable parameter')
parser.add_argument('--w3', type=int, default=0.1,
                    help='tunable parameter')
args = parser.parse_args()

# calculate Jensen-Shannon divergence of the node distance distribution
def NND(g, net_size):
    n = dict(nx.all_pairs_shortest_path_length(g))
    A = np.full((net_size, net_size), 0)
    B = np.full((net_size, net_size), 0)
    for i in (n.keys()):
        for j in range(net_size):
            if (j in n[i].keys()):
                A[i][j] = int(n[i][j])
            else:
                A[i][j] = int(net_size)
    for i in range(net_size):
        myset = set(A[i])
        myset.remove(0)
        for item in myset:
            B[i][item - 1] = list(A[i]).count(int(item))
    B = B / float(net_size - 1)
    pdfm = np.mean(B, axis=0)
    sum1 = 0
    for i in range(net_size):
        for j in range(len(B[0])):
            if (B[i][j] != 0 and pdfm[j] != 0):
                sum1 = sum1 + B[i][j] * np.log(B[i][j] / pdfm[j])
    return sum1 / net_size / np.log(nx.diameter(g) + 1), pdfm

# calculate the entropia of the distribution
def Entropia(a):
    b = np.where(a > 0)
    return -np.sum(a[b]*np.log(a[b]))

# calculate the dissimilarity between two networks g1 and g2
def Cal_Dissimility(g1, g2):
    nnd1, pdfm1 = NND(g1, args.network1_size)
    nnd2, pdfm2 = NND(g2, args.network2_size)
    if len(pdfm1) < len(pdfm2):
        add_number = len(pdfm2) - len(pdfm1)
        pdfm1 = np.concatenate([np.zeros(add_number), pdfm1])
    else:
        add_number = len(pdfm1) - len(pdfm2)
        pdfm2 = np.concatenate([np.zeros(add_number), pdfm2])
    pdfm = (pdfm1 + pdfm2) / 2
    first = np.sqrt(max((Entropia(pdfm) - (Entropia(pdfm1) + Entropia(pdfm2)) / 2) / np.log(2), 0))
    second = abs(np.sqrt(nnd1) - np.sqrt(nnd1))
    third = Cal_Alpha_Centrality(g1, g2)
    return args.w1 * first + args.w1 * second + args.w3/2 * third

# calculate Alpha Centrality
def Cal_Alpha_Centrality(g1, g2):
    alpha_sequence1 = Alpha_Centrality(g1, args.network1_size)
    alpha_sequence2 = Alpha_Centrality(g2, args.network2_size)
    if len(alpha_sequence1) < len(alpha_sequence2):
        add_number = len(alpha_sequence2) - len(alpha_sequence1)
        alpha_sequence1 = np.concatenate([np.zeros(add_number), alpha_sequence1])
    else:
        add_number = len(alpha_sequence1) - len(alpha_sequence2)
        alpha_sequence2 = np.concatenate([np.zeros(add_number), alpha_sequence2])
    g1_complement = nx.Graph()
    g2_complement = nx.Graph()
    g1_complement.add_nodes_from(nx.non_edges(g1))
    g2_complement.add_nodes_from(nx.non_edges(g2))
    alpha_complement_sequence1 = Alpha_Centrality(g1_complement, args.network1_size)
    alpha_complement_sequence2 = Alpha_Centrality(g2_complement, args.network2_size)
    if len(alpha_complement_sequence1) < len(alpha_complement_sequence2):
        add_number = len(alpha_complement_sequence2) - len(alpha_complement_sequence1)
        alpha_complement_sequence1 = np.concatenate([np.zeros(add_number), alpha_complement_sequence1])
    else:
        add_number = len(alpha_complement_sequence1) - len(alpha_complement_sequence2)
        alpha_complement_sequence2 = np.concatenate([np.zeros(add_number), alpha_complement_sequence2])
    alpha_sequence = (alpha_sequence1 + alpha_sequence2) / 2
    first = np.sqrt(max((Entropia(alpha_sequence) - (Entropia(alpha_sequence1) + Entropia(alpha_sequence2)) / 2) / np.log(2), 0))
    alpha_complement_sequence = (alpha_complement_sequence1 + alpha_complement_sequence2) / 2
    second = np.sqrt(max((Entropia(alpha_complement_sequence) - (Entropia(alpha_complement_sequence1) + Entropia(alpha_complement_sequence2)) / 2) / np.log(2), 0))
    return first + second

# calculate Alpha Centrality in detail
def Alpha_Centrality(g, net_size):
    numpy_adjacency = nx.to_numpy_array(g)
    identity_matrix = np.identity(len(numpy_adjacency))
    numpy_weights = [b/(net_size-1) for a, b in nx.degree(g)]
    alpha_cent = np.sort(np.linalg.inv(identity_matrix - 1/net_size * numpy_adjacency.T).dot(numpy_weights))
    return alpha_cent

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