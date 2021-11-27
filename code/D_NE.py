import networkx as nx
import pandas as pd
import numpy as np
from Node2vector import Node2Vec
import argparse
parser = argparse.ArgumentParser(description='Network dissimilarity based on graph embedding')
# general settings
parser.add_argument('--data_name1', default=None, help='network name')
parser.add_argument('--data_name2', default=None, help='network name')
parser.add_argument('--network1_size', default=None, help='network size')
parser.add_argument('--network2_size', default=None, help='network size')
parser.add_argument('--run_number', type=int, default=100,
                    help='number of node embedding iterations')
parser.add_argument('--w', type=int, default=8,
                    help='window size')
parser.add_argument('--s', type=int, default=10,
                    help='number of walks')
parser.add_argument('--l', type=int, default=60,
                    help='walk length')
parser.add_argument('--d', type=int, default=128,
                    help='embedding dims')
parser.add_argument('--w1', type=float, default=0.5,
                    help='tunable parameter')
parser.add_argument('--w2', type=float, default=0.5,
                    help='tunable parameter')
parser.add_argument('--L', type=int, default=10,
                    help='length of bin size')
args = parser.parse_args()

def JS(distribution_matrix, pdfm, number):
    sum1 = 0
    for i in range(number):
        for j in range(len(distribution_matrix[0])):
            if (distribution_matrix[i][j] != 0 and pdfm[j] != 0):
                sum1 = sum1 + distribution_matrix[i][j] * np.log(distribution_matrix[i][j] / pdfm[j])
    return sum1/number/np.log(number+1)

# calculate the entropia of the distribution
def Entropia(a):
    b = np.where(a > 0)
    return -np.sum(a[b]*np.log(a[b]))

# calculate the node embedding by Deepwalk
def GetEmbedding(g, number):
    node_vector_array = np.zeros([number, args.d], dtype=float)
    for _ in range(args.run_number):
        model = Node2Vec(g, walk_length=args.l, num_walks=args.s, p=1, q=1, workers=1)
        model.train(window_size=args.w)
        embeddings = model.get_embeddings(number)
        for i in range(number):
            node_vector_array[i] += list(embeddings[i])
    return node_vector_array/args.run_number

# calculate the distance matrix between arbitrary two nodes
def Cal_Nodedistance(node_embedding, number):
    node_distance_matrix = np.zeros((number, number), dtype=float)
    for i in range(number):
        for j in range(number):
            if (j > i):
                node_distance_matrix[i][j] = node_distance_matrix[j][i] = np.linalg.norm(node_embedding[i] - node_embedding[j])
    return node_distance_matrix
    
# calculate node distribution
def Cal_Eachnode_Distribution(idx, distance_array, bin_length, bin_list, number):
    node_distribution = [0] * bin_length
    for i in range(len(distance_array)):
        for j in range(bin_length):
            if idx != i:
                if j == bin_length - 1:
                    if distance_array[i] >= bin_list[j] and distance_array[i] <= bin_list[j + 1]:
                        node_distribution[j] += 1
                else:
                    if distance_array[i] >= bin_list[j] and distance_array[i] < bin_list[j + 1]:
                        node_distribution[j] += 1
            else:
                break
    return np.array(node_distribution) / (number - 1)

# calculate distribution H in detail
def Cal_Distance_Distribution(nodedistance, bin_list, number):
    node_distribution = np.zeros((number, args.L), dtype=float)
    for i in range(number):
        node_distribution[i] = Cal_Eachnode_Distribution(i, nodedistance[i], args.L, bin_list, number)
    return node_distribution

# calculate distribution H
def Cal_Distribution(g, network_size):
    node_embedding = GetEmbedding(g, network_size)
    node_distance_matrix = Cal_Nodedistance(node_embedding, network_size)
    bin_list = np.linspace(np.min(node_distance_matrix), np.max(node_distance_matrix), args.L + 1)
    node_distribution = Cal_Distance_Distribution(node_distance_matrix, bin_list, network_size)
    return node_distribution

# calculate the dissimilarity between two networks g1 and g2
def Cal_Dissimility(g1, g2):
    H1 = Cal_Distribution(g1, args.network1_size)
    H2 = Cal_Distribution(g2, args.network2_size)
    pdfm1 = np.mean(H1, axis=0)
    js1 = JS(H1, pdfm1, args.network1_size)
    pdfm2 = np.mean(H2, axis=0)
    js2 = JS(H2, pdfm2, args.network2_size)
    pdfm = (pdfm1 + pdfm2) / 2
    first = np.sqrt(max((Entropia(pdfm) - (Entropia(pdfm1) + Entropia(pdfm2)) / 2) / np.log(2), 0))
    second = abs(np.sqrt(js1) - np.sqrt(js2))
    return args.w1 * first + args.w1 * second

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