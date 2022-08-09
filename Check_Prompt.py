from parse_args import args
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.model_selection import KFold
from task import compute_metrics
from sklearn import linear_model


seed = 2022
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed)


class CheckPrompt(nn.Module):
    def __init__(self, weight, b):
        super(CheckPrompt, self).__init__()
        self.lin = nn.Linear(288, 1)
        self.lin.weight = nn.Parameter(torch.concat([weight, weight], dim=1))
        self.lin.bias = nn.Parameter(b)

        self.pre_fix = torch.nn.Parameter(torch.randn(180, 144))

    def forward(self, index, emb):
        tmp = torch.concat([self.pre_fix[index], emb], dim=1)
        tmp = self.lin(tmp)
        return tmp


def check_test():
    check_counts = np.load('../data/check_counts.npy', allow_pickle=True).reshape(180, 1)
    pre_train_emb = np.load('emb.npy', allow_pickle=True)

    index = torch.arange(180)

    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []

    for train_index, test_index in kf.split(index):
        loss_fn = torch.nn.MSELoss()
        reg = linear_model.Ridge(alpha=1.0)
        X_train = pre_train_emb[train_index]
        Y_train = check_counts[train_index]
        X_test = pre_train_emb[test_index]
        Y_test = check_counts[test_index]

        # for weight
        reg.fit(X_train, Y_train)

        X_train = torch.nn.Parameter(torch.tensor(X_train)).to(args.device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(args.device)
        X_test = torch.tensor(X_test).to(args.device)
        Y_test = torch.tensor(Y_test, dtype=torch.float32).to(args.device)

        prompt = CheckPrompt(torch.tensor(reg.coef_, dtype=torch.float32),
                             torch.tensor(reg.intercept_, dtype=torch.float32)).to(args.device)

        optimizer = optim.Adam(prompt.parameters(), lr=0.001, weight_decay=0.05)  # 0.001 0.05

        for i in range(6000):
            optimizer.zero_grad()
            y_pred = prompt(train_index, X_train)
            loss = loss_fn(y_pred, Y_train)

            print(i, loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_test = prompt(test_index, X_test)
            y_preds.append(y_test)
            y_truths.append(Y_test)
    for i in range(5):
        y_preds[i] = y_preds[i].detach().cpu().numpy().squeeze()
        y_truths[i] = y_truths[i].detach().cpu().numpy()
    mae, rmse, r2 = compute_metrics(np.concatenate(y_preds), np.concatenate(y_truths))
    return mae, rmse, r2


if __name__ == '__main__':
    print('check prediction test-----')
    check_mae, check_rmse, check_r2 = check_test()
    print('check:', check_mae, check_rmse, check_r2)
