import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class RelationGCN(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(RelationGCN, self).__init__()
        self.gcn_layers = gcn_layers
        self.dropout = dropout

        self.gcns = nn.ModuleList([GCNConv(in_channels=embedding_size, out_channels=embedding_size)
                                   for _ in range(self.gcn_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(embedding_size) for _ in range(self.gcn_layers - 1)])
        self.relation_transformation = nn.ModuleList([nn.Linear(embedding_size, embedding_size)
                                                      for _ in range(self.gcn_layers)])

    def forward(self, features, rel_emb, edge_index, is_training=True):
        n_emb = features
        poi_emb = features
        s_emb = features
        d_emb = features
        poi_r, s_r, d_r, n_r = rel_emb
        poi_edge_index, s_edge_index, d_edge_index, n_edge_index = edge_index
        for i in range(self.gcn_layers - 1):
            tmp = n_emb
            n_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(n_emb, n_r), n_edge_index)))
            n_r = self.relation_transformation[i](n_r)
            if is_training:
                n_emb = F.dropout(n_emb, p=self.dropout)

            tmp = poi_emb
            poi_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(poi_emb, poi_r), poi_edge_index)))
            poi_r = self.relation_transformation[i](poi_r)
            if is_training:
                poi_emb = F.dropout(poi_emb, p=self.dropout)

            tmp = s_emb
            s_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(s_emb, s_r), s_edge_index)))
            s_r = self.relation_transformation[i](s_r)
            if is_training:
                s_emb = F.dropout(s_emb, p=self.dropout)

            tmp = d_emb
            d_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(d_emb, d_r), d_edge_index)))
            d_r = self.relation_transformation[i](d_r)
            if is_training:
                d_emb = F.dropout(d_emb, p=self.dropout)

        n_emb = self.gcns[-1](torch.multiply(n_emb, n_r), n_edge_index)
        poi_emb = self.gcns[-1](torch.multiply(poi_emb, poi_r), poi_edge_index)
        s_emb = self.gcns[-1](torch.multiply(s_emb, s_r), s_edge_index)
        d_emb = self.gcns[-1](torch.multiply(d_emb, d_r), d_edge_index)

        # rel update
        n_r = self.relation_transformation[-1](n_r)
        poi_r = self.relation_transformation[-1](poi_r)
        s_r = self.relation_transformation[-1](s_r)
        d_r = self.relation_transformation[-1](d_r)

        return n_emb, poi_emb, s_emb, d_emb, n_r, poi_r, s_r, d_r


class CrossLayer(nn.Module):
    def __init__(self, embedding_size):
        super(CrossLayer, self).__init__()
        self.alpha_n = nn.Parameter(torch.tensor(0.95))
        self.alpha_poi = nn.Parameter(torch.tensor(0.95))
        self.alpha_d = nn.Parameter(torch.tensor(0.95))
        self.alpha_s = nn.Parameter(torch.tensor(0.95))

        self.attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=4)

    def forward(self, n_emb, poi_emb, s_emb, d_emb):
        stk_emb = torch.stack((n_emb, poi_emb, d_emb, s_emb))
        fusion, _ = self.attn(stk_emb, stk_emb, stk_emb)

        n_f = fusion[0] * self.alpha_n + (1 - self.alpha_n) * n_emb
        poi_f = fusion[1] * self.alpha_poi + (1 - self.alpha_poi) * poi_emb
        d_f = fusion[2] * self.alpha_d + (1 - self.alpha_d) * d_emb
        s_f = fusion[3] * self.alpha_s + (1 - self.alpha_s) * s_emb

        return n_f, poi_f, s_f, d_f


class AttentionFusionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionFusionLayer, self).__init__()
        self.q = nn.Parameter(torch.randn(embedding_size))
        self.fusion_lin = nn.Linear(embedding_size, embedding_size)

    def forward(self, n_f, poi_f, s_f, d_f):
        n_w = torch.mean(torch.sum(F.leaky_relu(self.fusion_lin(n_f)) * self.q, dim=1))
        poi_w = torch.mean(torch.sum(F.leaky_relu(self.fusion_lin(poi_f)) * self.q, dim=1))
        s_w = torch.mean(torch.sum(F.leaky_relu(self.fusion_lin(s_f)) * self.q, dim=1))
        d_w = torch.mean(torch.sum(F.leaky_relu(self.fusion_lin(d_f)) * self.q, dim=1))

        w_stk = torch.stack((n_w, poi_w, s_w, d_w))
        w = torch.softmax(w_stk, dim=0)

        region_feature = w[0] * n_f + w[1] * poi_f + w[2] * s_f + w[3] * d_f
        return region_feature


class PM_Model(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(PM_Model, self).__init__()

        self.relation_gcns = RelationGCN(embedding_size, dropout, gcn_layers)

        self.cross_layer = CrossLayer(embedding_size)

        self.fusion_layer = AttentionFusionLayer(embedding_size)

    def forward(self, features, rel_emb, edge_index, is_training=True):
        poi_emb, s_emb, d_emb, n_emb, poi_r, s_r, d_r, n_r = self.relation_gcns(features, rel_emb,
                                                                                edge_index, is_training)
        n_f, poi_f, s_f, d_f = self.cross_layer(n_emb, poi_emb, s_emb, d_emb)

        region_feature = self.fusion_layer(n_f, poi_f, s_f, d_f)

        n_f = torch.multiply(region_feature, n_r)
        poi_f = torch.multiply(region_feature, poi_r)
        s_f = torch.multiply(region_feature, s_r)
        d_f = torch.multiply(region_feature, d_r)

        return region_feature, n_f, poi_f, s_f, d_f
