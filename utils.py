from parse_args import args

import numpy as np
import torch
import scipy.sparse as sp


def load_data():
    data_path = args.data_path
    mobility_adj = np.load(data_path + args.mobility_adj, allow_pickle=True)
    mobility_adj = mobility_adj.squeeze()
    mobility = mobility_adj.copy()
    mobility = mobility / np.mean(mobility)

    poi_similarity = np.load(data_path + args.poi_similarity, allow_pickle=True)
    poi_similarity[np.isnan(poi_similarity)] = 0

    d_adj = np.load(data_path + args.destination_adj, allow_pickle=True)
    d_adj[np.isnan(d_adj)] = 0

    s_adj = np.load(data_path + args.source_adj, allow_pickle=True)
    s_adj[np.isnan(s_adj)] = 0

    neighbor = np.load(data_path + args.neighbor, allow_pickle=True)

    return poi_similarity, s_adj, d_adj, mobility, neighbor


def graph_to_COO(similarity, importance_k):
    graph = torch.eye(180)

    for i in range(180):
        graph[np.argsort(similarity[:, i])[-importance_k:], i] = 1
        graph[i, np.argsort(similarity[:, i])[-importance_k:]] = 1

    edge_index = sp.coo_matrix(graph)
    edge_index = np.vstack((edge_index.row, edge_index.col))
    return edge_index


def create_graph(similarity, importance_k):
    edge_index = graph_to_COO(similarity, importance_k)
    return edge_index


def pair_sample(neighbor):
    positive = torch.zeros(180, dtype=torch.long)
    negative = torch.zeros(180, dtype=torch.long)

    for i in range(180):
        region_idx = np.random.randint(len(neighbor[i]))
        pos_region = neighbor[i][region_idx]
        positive[i] = pos_region
    for i in range(180):
        neg_region = np.random.randint(180)
        while neg_region in neighbor[i] or neg_region == i:
            neg_region = np.random.randint(180)
        negative[i] = neg_region
    return positive, negative


def create_neighbor_graph(neighbor):
    graph = np.eye(180)

    for i in range(len(neighbor)):
        for region in neighbor[i]:
            graph[i, region] = 1
            graph[region, i] = 1
    graph = sp.coo_matrix(graph)
    edge_index = np.stack((graph.row, graph.col))
    return edge_index
