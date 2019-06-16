import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.datasets.ccf_ai import ccf_ai
from rec.utils import cuda
from dgl import DGLGraph

import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--opt', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--sched', type=str, default='none')
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--use-feature', action='store_true')
parser.add_argument('--sgd-switch', type=int, default=-1)
parser.add_argument('--n-negs', type=int, default=10)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--hard-neg-prob', type=float, default=0.1)
parser.add_argument('--decay_factor', type=float, default=0.98)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--zero_h', action='store_true')
args = parser.parse_args()

print(args)

cache_file = 'ccf-collect.pkl'

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        db = pickle.load(f)
else:
    db = ccf_ai('/data2/ruofan/MyFiles/GCN_ace/')
    with open(cache_file, 'wb') as f:
        pickle.dump(db, f)
###
# ml = MovieLens('./ml-1m')
####
g = db.g
neighbors = db.neighbors.to(dtype=torch.long)

n_hidden = 256
n_layers = args.layers
batch_size = 256
margin = 0.9

n_negs = args.n_negs
hard_neg_prob = np.linspace(0,args.hard_neg_prob,args.epochs)

sched_lambda = {
        'none': lambda epoch: 1,
        'decay': lambda epoch: max(args.decay_factor ** epoch, 1e-4),
        }
loss_func = {
        'hinge': lambda diff: (diff + margin).clamp(min=0).mean(),
        'bpr': lambda diff: (1 - torch.sigmoid(-diff)).mean(),
        }

in_features = n_hidden
emb = nn.ModuleDict()
emb['year'] = nn.Embedding(
    g.ndata['year'].max().item() + 1,
    in_features,
    padding_idx=0
        )
if 'venue' in g.ndata.keys():
    emb['venue'] = nn.Embedding(
        g.ndata['venue'].max().item() + 1,
        in_features,
        padding_idx=0
            )
emb['fos'] = nn.Sequential(
    nn.Linear(300, in_features),
    nn.LeakyReLU(),
    )

model = cuda(PinSage(
    g.number_of_nodes(),
    [n_hidden] * (n_layers + 1),
    20,
    0.5,
    20,
    emb=emb,
    G=g,
    zero_h=args.zero_h
    ))
opt = getattr(torch.optim, args.opt)(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda[args.sched])


def forward(model, g_prior, nodeset, train=True):
    if train:
        return model(g_prior, nodeset)
    else:
        with torch.no_grad():
            return model(g_prior, nodeset)


def filter_nid(nids, nid_from):
    nids = [nid.numpy() for nid in nids]
    nid_from = nid_from.numpy()
    np_mask = np.logical_and(*[np.isin(nid, nid_from) for nid in nids])
    return [torch.from_numpy(nid[np_mask]) for nid in nids]


def runtrain(g_prior_edges, g_train_edges, train):
    global opt
    if train:
        model.train()
    else:
        model.eval()

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: cuda(v) for k, v in g.ndata.items()})
    edge_batches = g_train_edges[torch.randperm(g_train_edges.shape[0])].split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id, batch in enumerate(tq):
            count += batch.shape[0]
            src, dst = g.find_edges(batch)
            dst_neg = []
            for i in range(len(dst)):
                if np.random.rand() < args.hard_neg_prob:
                    nb = torch.LongTensor(neighbors[dst[i].item()])
                    mask = ~(g.has_edges_between(nb, src[i].item()).byte())
                    dst_neg.append(np.random.choice(nb[mask].numpy(), n_negs))
                else:
                    dst_neg.append(np.random.randint(
                        0, len(db.papers), n_negs))


            dst_neg = torch.LongTensor(dst_neg)
            dst = dst.view(-1, 1).expand_as(dst_neg).flatten()
            src = src.view(-1, 1).expand_as(dst_neg).flatten()
            dst_neg = dst_neg.flatten()

            mask = (g_prior.in_degrees(dst_neg) > 0) & \
                   (g_prior.in_degrees(dst) > 0) & \
                   (g_prior.in_degrees(src) > 0)
            src = src[mask]
            dst = dst[mask]
            dst_neg = dst_neg[mask]
            if len(src) == 0:
                continue

            nodeset = cuda(torch.cat([src, dst, dst_neg]))
            src_size, dst_size, dst_neg_size = \
                    src.shape[0], dst.shape[0], dst_neg.shape[0]

            h_src, h_dst, h_dst_neg = (
                    forward(model, g_prior, nodeset, train)
                    .split([src_size, dst_size, dst_neg_size]))

            diff = (h_src * (h_dst_neg - h_dst)).sum(1)
            loss = loss_func[args.loss](diff)
            acc = (diff < 0).sum()
            assert loss.item() == loss.item()

            grad_sqr_norm = 0
            if train:
                opt.zero_grad()
                loss.backward()
                for name, p in model.named_parameters():
                    assert (p.grad != p.grad).sum() == 0
                    grad_sqr_norm += p.grad.norm().item() ** 2
                opt.step()

            sum_loss += loss.item()
            sum_acc += acc.item() / n_negs
            avg_loss = sum_loss / (batch_id + 1)
            avg_acc = sum_acc / count
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': '%.3f' % avg_loss,
                            'avg_acc': '%.3f' % avg_acc,
                            'grad_norm': '%.6f' % np.sqrt(grad_sqr_norm)})

    return avg_loss, avg_acc

def id_remap(pids, offset, period):
    hids = pids[np.where((pids >= offset) & ((pids - offset) % period == 0))]
    return (hids - offset) // period

def runtest(g_prior_edges, epoch, validation=True):
    model.eval()
    period = 1
    offset = epoch % period
    n_users = len(db.authors.index)
    n_items = len(db.papers.index)

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: cuda(v) for k, v in g.ndata.items()})

    user_offset = 0
    hs = []
    with torch.no_grad():
        with tqdm.trange(offset, n_users + n_items, period) as tq:
            for node_id in tq:
                if user_offset == 0 and node_id >= n_items:
                    user_offset = node_id

                nodeset = cuda(torch.LongTensor([node_id]))
                h = forward(model, g_prior, nodeset, False)
                hs.append(h)
    h = torch.cat(hs, 0)

    rr = []

    with torch.no_grad():
        with tqdm.trange(user_offset, n_items + n_users, period) as tq:
            for u_nid in tq:
                # uid = db.user_ids[u_nid]
                uid = u_nid
                uhid = (u_nid - offset)//period

                pids_exclude = db.links[
                    (db.links['idx_A'] == uid) &
                    (db.links['train'] | db.links['test' if validation else 'valid'])
                    ]['idx_P'].values
                pids_candidate = db.links[
                    (db.links['idx_A'] == uid) &
                    db.links['valid' if validation else 'test']
                    ]['idx_P'].values

                pids = np.setdiff1d(range(len(db.paper_ids_map)), pids_exclude)

                hids = id_remap(pids, offset, period)
                hids_candidate = id_remap(pids_candidate, offset, period)

                dst = torch.from_numpy(hids)
                src = torch.zeros_like(dst).fill_(uhid)
                h_dst = h[dst]
                h_src = h[src]

                score = (h_src * h_dst).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()

                rank_map = {v: i for i, v in enumerate(hids[score_sort_idx])}
                rank_candidates = np.array([rank_map[p_nid] for p_nid in hids_candidate])
                rank = 1 / (rank_candidates + 1) if len(rank_candidates)!= 0 else np.array([1/ len(score_sort_idx)])
                rr.append(rank.mean())
                tq.set_postfix({'rank': rank.mean()})

    return np.array(rr)


def train():
    global opt, sched
    log_val = open(f'val_{args.suffix}.log', 'w')
    log_test = open(f'test_{args.suffix}.log', 'w')
    log_train = open(f'train_{args.suffix}.log', 'w')
    best_mrr = 0
    for epoch in range(args.epochs):
        args.hard_neg_prob = hard_neg_prob[epoch]
        db.refresh_mask()
        g_prior_edges = g.filter_edges(lambda edges: edges.data['prior'])
        g_train_edges = g.filter_edges(lambda edges: edges.data['train'] & ~edges.data['inv'])
        g_prior_train_edges = g.filter_edges(
                lambda edges: edges.data['prior'] | edges.data['train'])
        #
        if (epoch+1)%10==0:
            print('Epoch %d validation' % epoch)
            with torch.no_grad():
                valid_mrr = runtest(g_prior_train_edges, epoch, True)
                log_val.write(f'{valid_mrr.mean()}\n')
                if best_mrr < valid_mrr.mean():
                    best_mrr = valid_mrr.mean()
                    torch.save(model.state_dict(), f'model_best_{args.suffix}.pt')
            print(pd.Series(valid_mrr).describe())
            print('Epoch %d test' % epoch)
            with torch.no_grad():
                test_mrr = runtest(g_prior_train_edges, False)
                log_test.write(f"{test_mrr.mean()}\n")
            print(pd.Series(test_mrr).describe())

        print('Epoch %d train' % epoch)
        avg_loss, avg_acc = runtrain(g_prior_edges, g_train_edges, True)
        log_train.write(f'{avg_loss} {avg_acc}\n')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_itr{epoch}_{args.suffix}.pt')

        if epoch == args.sgd_switch:
            opt = torch.optim.SGD(model.parameters(), lr=0.05)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda['decay'])
        # elif epoch < args.sgd_switch:
        sched.step()
    log_train.close()
    log_val.close()
    log_test.close()


if __name__ == '__main__':
    train()
