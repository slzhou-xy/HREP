import utils
from parse_args import args
from task import predict_crime, clustering, predict_check
from pre_training_model import PM_Model
# from ablation import MyModel

import random
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

seed = 2022
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed)

poi_similarity, s_adj, d_adj, mobility, neighbor = utils.load_data()
poi_edge_index = utils.create_graph(poi_similarity, args.importance_k)
s_edge_index = utils.create_graph(s_adj, args.importance_k)
d_edge_index = utils.create_graph(d_adj, args.importance_k)
n_edge_index = utils.create_neighbor_graph(neighbor)

poi_edge_index = torch.tensor(poi_edge_index, dtype=torch.long).to(args.device)
s_edge_index = torch.tensor(s_edge_index, dtype=torch.long).to(args.device)
d_edge_index = torch.tensor(d_edge_index, dtype=torch.long).to(args.device)
n_edge_index = torch.tensor(n_edge_index, dtype=torch.long).to(args.device)

mobility = torch.tensor(mobility, dtype=torch.float32).to(args.device)
poi_similarity = torch.tensor(poi_similarity, dtype=torch.float32).to(args.device)

features = torch.randn(args.regions_num, args.embedding_size).to(args.device)
poi_r = torch.randn(args.embedding_size).to(args.device)
s_r = torch.randn(args.embedding_size).to(args.device)
d_r = torch.randn(args.embedding_size).to(args.device)
n_r = torch.randn(args.embedding_size).to(args.device)
rel_emb = [poi_r, s_r, d_r, n_r]
edge_index = [poi_edge_index, s_edge_index, d_edge_index, n_edge_index]


def mob_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)) - torch.mul(mob, torch.log(pd_hat)))
    return loss


def train(net):
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=5e-3)
    loss_fn1 = torch.nn.TripletMarginLoss()
    loss_fn2 = torch.nn.MSELoss()

    best_rmse = 10000
    best_mae = 10000
    best_r2 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        region_emb, n_emb, poi_emb, s_emb, d_emb = net(features, rel_emb, edge_index)

        pos_idx, neg_idx = utils.pair_sample(neighbor)

        geo_loss = loss_fn1(n_emb, n_emb[pos_idx], n_emb[neg_idx])

        m_loss = mob_loss(s_emb, d_emb, mobility)

        POI_loss = loss_fn2(torch.mm(poi_emb, poi_emb.T), poi_similarity)
        loss = POI_loss + m_loss + geo_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae, rmse, r2 = predict_crime(region_emb.detach().cpu().numpy())
            # nmi, ari = clustering(region_emb.detach().cpu().numpy())
            # print(nmi, ari)
            if rmse < best_rmse and mae < best_mae and best_r2 < r2:
                best_rmse = rmse
                best_mae = mae
                best_r2 = r2
                best_epoch = epoch
            print(epoch, rmse, mae, r2, loss.item())
            # np.save('emb', region_emb.detach().cpu().numpy())

    print('best_rmse:', best_rmse)
    print('best_mae:', best_mae)
    print('best_r2:', best_r2)
    print('best_epoch:', best_epoch)


def test(net):
    region_emb, _, _, _, _ = net(features, rel_emb, edge_index, False)
    print('>>>>>>>>>>>>>>>>>   crime')
    mae, rmse, r2 = predict_crime(region_emb.detach().cpu().numpy())
    print("MAE:  %.3f" % mae)
    print("RMSE: %.3f" % rmse)
    print("R2:   %.3f" % r2)
    print('>>>>>>>>>>>>>>>>>   check')
    mae, rmse, r2 = predict_check(region_emb.detach().cpu().numpy())
    print("MAE:  %.3f" % mae)
    print("RMSE: %.3f" % rmse)
    print("R2:   %.3f" % r2)
    print('>>>>>>>>>>>>>>>>>   clustering')
    nmi, ari = clustering(region_emb.detach().cpu().numpy())
    print("NMI: %.3f" % nmi)
    print("ARI: %.3f" % ari)

    np.save('emb', region_emb.detach().cpu().numpy())


if __name__ == '__main__':
    net = PM_Model(args.embedding_size, args.dropout, args.gcn_layers).to(args.device)
    print('training-----------------')
    net.train()
    train(net)
    net.eval()
    print('downstream task test-----')
    test(net)
