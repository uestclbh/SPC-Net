import argparse
import datasets
from dataset_loader import DataLoader
from utils import random_planetoid_splits
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
import optuna


def RunExp(n_poly,
           percls_trn,
           val_lb,
           lr1,
           lr2,
           lr3,
           wd1,
           wd2,
           wd3,
           dpb1,
           dpb2,
           dpb3,
           k,
           t,
           b, c, args, data, Net):
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        reg_loss = None
        loss.backward()
        optimizer.step()
        del out

    def test(model, data, acc_flag):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(model(data)[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    print("torch.cuda.is_available():{}".format(torch.cuda.is_available()))

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    tmp_net = Net(data, n_poly, dpb1,dpb2,dpb3, t, b, c,k)


    permute_masks = random_planetoid_splits
    data = permute_masks(args, data)


    model, data = tmp_net.to(device), data.to(device)

    if args.net == 'GPRGNN':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])



    elif args.net == 'PCNet':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': wd1, 'lr': lr1},
             {'params': model.lin2.parameters(), 'weight_decay': wd3, 'lr': lr3},
             {'params': model.prop1.parameters(), 'weight_decay': wd2, 'lr': lr2}])



    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        train(model, optimizer, data)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, args.acc)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc


        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:', epoch)
                    break
    return test_acc, best_val_acc, time_run


def search_hyper_params(trial: optuna.Trial):
    # 展开的项数

    # conv_layer = trial.suggest_categorical('conv_layer', [4, 5, 6])
    # aggr = "gcn"
    lr1 = trial.suggest_categorical("lr1", [0.005, 0.01, 0.05, 0.1])
    lr2 = trial.suggest_categorical("lr2", [0.005, 0.01, 0.05, 0.1])
    lr3 = trial.suggest_categorical("lr3", [0.005, 0.01, 0.05, 0.1])


    wd1 = trial.suggest_categorical("wd1", [0.0, 5e-6, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    wd2 = trial.suggest_categorical("wd2", [0.0, 5e-6, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    wd3 = trial.suggest_categorical("wd3", [0.0, 5e-6, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3])


    n_poly = trial.suggest_categorical('n_poly', [4,5, 6, 7,8,9])
    # pc展开的项数
    k = trial.suggest_float('k',0, 10,step=0.5)
    # 聚合次数
    t = trial.suggest_float('t', 0, 5, step=0.1)
    # 热核超参t
    c = trial.suggest_float('c', 0.0, 0.8, step=0.05)
    # 归一化度矩阵的负次方
    b = trial.suggest_float('b', -5.0, 5.0, step=0.1)
    # 自环添加程度
    dpb1 = trial.suggest_float('dpb1', 0.0, 0.9, step=0.05)
    dpb2 = trial.suggest_float('dpb2', 0.0, 0.9, step=0.05)
    dpb3 = trial.suggest_float('dpt1', 0.0, 0.9, step=0.05)
    early_stop = trial.suggest_categorical('early_stop', [200])
    return maining(args, n_poly,
                   lr1,
                   lr2,
                   lr3,
                   wd1,
                   wd2,
                   wd3,
                   dpb1,
                   dpb2,
                   dpb3,
                   k,
                   K=0, reg_zero=0,
                   t=t,
                   b=b, c=c, early_stop=early_stop)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maining(args, n_poly,
            lr1,
            lr2,
            lr3,
            wd1,
            wd2,
            wd3,
            dpb1,
            dpb2,
            dpb3,
            k,
            K, reg_zero,
            t, b, c, early_stop):
    print(args)
    print("---------------------------------------------")
    print("K:{},reg_zero:{}".format(K, reg_zero))
    gnn_name = args.net
    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'MLP':
        Net = MLP
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'PCNet':
        # and args.gnn_type == 0:
        Net = PCNet1
    # elif gnn_name == 'PCNet' and args.gnn_type == 1:
    #     Net = PCNet2
    # elif gnn_name == 'PCNet' and args.gnn_type == 2:
    #     SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539,
    #              3212139042,
    #              2424918363]
    #     Net = PCNet1

    if args.dataset in ['Actor', 'Chameleon', 'Squirrel']:
        if args.dataset == 'Actor':
            data = datasets.load_dataset('film')
        else:
            data = datasets.load_dataset(args.dataset.lower())
    else:
            data = DataLoader(args.dataset.lower())


    percls_trn = int(round(args.train_rate * len(data.y) / data.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))


    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    print(device)
    data = data.to(device)
    args.early_stopping = early_stop

    results = []
    time_results = []
    for RP in tqdm(range(10)):
        args.runs = RP
        test_acc, best_val_acc, time_run = RunExp(n_poly,
                                                          percls_trn,
                                                          val_lb,
                                                          lr1,
                                                          lr2,
                                                          lr3,
                                                          wd1,
                                                          wd2,
                                                          wd3,
                                                          dpb1,
                                                          dpb2,
                                                          dpb3,
                                                          k,
                                                          t,
                                                          b, c, args, data, Net)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')
        print(torch.cuda.is_available())
    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / 10, "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values = np.asarray(results)[:, 0]
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))


    print(f'{gnn_name} on dataset {args.dataset}, in 10 repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
    return test_acc_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optruns', type=int, default=100)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--name', type=str, default="opt")
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--train_rate', type=float, default=0.025, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.025, help='val set rate.')
    parser.add_argument('--dataset', type=str,
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Chameleon', 'Squirrel', 'Actor',
                                 'Texas', 'Cornell'],
                        default='Cornell')


    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'GPRGNN', 'PCNet', 'MLP'],
                        default='PCNet')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--split', type=int, default=0, help='dataset split')
    parser.add_argument('--acc', type=int, default=0, help='0->ACC,1->AUC(genius)')
    parser.add_argument('--gnn_type', type=int, default=0, help='0->general ,1->ablation, 2->large')
    parser.add_argument('--reproduce', type=int, default=1, help='1->Table1 ,2->Table2, 3->Table3')
    # split 0 random 60% 20% 20%     #split  sparse 2.5% 2.5% 95%
    # split 1 fixed  60% 20% 20%
    # split 2 20,50,1000

    args = parser.parse_args()
    # argparse部分引入参数
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path +
                                        args.name + ".db",
                                study_name=args.name,
                                load_if_exists=True)
    study.optimize(search_hyper_params, n_trials=args.optruns)
    print("best params ", study.best_params)
    print("best valf1 ", study.best_value)

