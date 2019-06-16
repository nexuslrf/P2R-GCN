import json
import dgl
import torch
import numpy as np
import tqdm
import sys
import re
import string
import spacy
import os
from functools import partial
import pickle
from .. import randomwalk
import time
import torch.multiprocessing as mp
import pandas as pd

class MP_mask:
    def __init__(self, train_edges, author_w_link, author_r_link, paper_link, queueSize=50):
        self.train_edges = train_edges
        self.author_w_link = author_w_link
        self.author_r_link = author_r_link
        self.paper_link = paper_link
        self.Q = mp.Queue(maxsize=queueSize)

    def generate_mask(self, round=5):
        while True:
            train_masks = [[] for i in range(round)]
            for i in range(len(self.author_w_link)):
                author_edges = self.author_w_link[i] + self.author_r_link[i]
                if len(author_edges) >= round:
                    seg = len(author_edges) // round
                    np.random.shuffle(author_edges)
                    for j in range(round-1):
                        train_masks[j] += author_edges[j*seg:(j+1)*seg]
                    train_masks[round-1] += author_edges[(round-1)* seg:]
                else:
                    tmp = np.random.permutation(range(len(author_edges)))
                    for j in range(len(tmp)):
                        train_masks[tmp[j]].append(author_edges[j])

            for line in train_masks:
                train_mask = np.intersect1d(self.train_edges,line)
                prior_mask = np.setdiff1d(self.train_edges,line)

                prior_mask = set(prior_mask)
                for i in range(len(self.paper_link)):
                    paper_edges = np.intersect1d(self.paper_link[i], self.train_edges)
                    sel = False
                    for j in paper_edges:
                        if j in prior_mask:
                            sel = True
                            break
                    if not sel:
                        prior_mask.add(paper_edges[np.random.randint(len(paper_edges))])

                prior_mask = list(prior_mask)
                while self.Q.full():
                    time.sleep(2)

                self.Q.put((prior_mask, train_mask))

    def start(self):
        p = mp.Process(target=self.generate_mask, args=())
        p.daemon = True
        p.start()

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

class ccf_ai(object):
    """docstring for ccf_ai"""

    def __init__(self, dir='../../'):
        super(ccf_ai, self).__init__()
        self.dir = dir
        tmp = torch.load(self.dir+'ccf-ai-venues.pt')
        self.AI_venues = tmp['AI_A_C'] # ['computer vision and pattern recognition'] #tmp['AI_A_C']# + tmp['AI_A_J'] #+ tmp['AI_B_C'] + tmp['AI_B_J']
        # load raw data
        # file = open(self.dir+"AI_A_C.txt")
        # AI_A_C = file.readlines()
        # file = open(self.dir+"AI_B_C.txt")
        # AI_B_C = file.readlines()
        # file = open(self.dir+"AI_A_J.txt")
        # AI_A_J = file.readlines()
        # file = open(self.dir+"AI_B_J.txt")
        # AI_B_J = file.readlines()
        # db_raw = AI_A_C #+ AI_A_J # + AI_B_C + AI_B_J
        db_raw = open(self.dir + 'AI_collection.txt').readlines()

        # Word Embedding
        nlp = spacy.load('/tmp/en_vectors_wiki_lg/')
        self.vec_paper = np.zeros((len(db_raw), 300))
        # self.vec_paper = np.load(self.dir+'fos2vec_AI_AB.npy')


        self.paper_w = np.zeros(len(db_raw))
        # ccf_set = np.zeros(len(db_raw), dtype=int)
        self.papers = []
        self.paper_ids_map = {}
        self.authors = []
        self.author_ids_map = {}
        ######### pd


        trange = tqdm.tqdm(range(len(db_raw)))
        for i in trange:
            sum_w = 0.
            d = json.loads(db_raw[i])

            paper_id = ''
            paper_title = ''
            paper_year = 0
            paper_venue = ''
            fos_pair = []

            if 'fos' in d:
                fos = d['fos']
                for f in fos:
                    fos_pair.append((f['name'], f['w']))
                    self.vec_paper[i, :] = self.vec_paper[i, :] + \
                                          nlp(f['name']).vector * f['w']
                    sum_w += f['w']
                self.vec_paper[i, :] = self.vec_paper[i, :] / sum_w
                self.paper_w[i] = sum_w

            if 'venue' in d:
                paper_venue = d['venue']['raw']

            if 'year' in d:
                paper_year = d['year']

            if 'id' in d:
                paper_id = d['id']
                if paper_id not in self.paper_ids_map:
                    self.paper_ids_map[paper_id] = len(self.papers)

            if 'title' in d:
                paper_title = d['title']

            self.papers.append({
                'gid': i,
                'id': paper_id,
                'title': paper_title,
                'year': paper_year,
                'venue': paper_venue,
                'fos': fos_pair
            })

            if 'authors' in d:
                da = d['authors']
                for j in da:
                    if j['id'] not in self.author_ids_map:
                        self.authors.append({
                            'gid': len(self.author_ids_map) + len(db_raw),
                            'id': j['id'],
                            'name': j['name']
                        })
                        self.author_ids_map[j['id']] = len(self.authors) - 1

        self.papers = pd.DataFrame(self.papers).set_index('gid').astype({'year': 'category','venue': 'category'})
        self.authors = pd.DataFrame(self.authors).set_index('gid').astype('category')

        self.edge_list = []
        self.edge_wt = []
        self.edge_ref = []
        self.author_write = [[] for i in range(len(self.author_ids_map))]
        self.author_ref = [[] for i in range(len(self.author_ids_map))]
        self.author_w_link = [[] for i in range(len(self.author_ids_map))]
        self.author_r_link = [[] for i in range(len(self.author_ids_map))]

        self.paper_ref_paper = [[] for i in range(len(self.paper_ids_map))]  # idx of paper
        self.paper_author = [[] for i in range(len(self.paper_ids_map))]
        self.paper_link = [[] for i in range(len(self.paper_ids_map))]

        cnt = 0
        for i in tqdm.tqdm(range(len(db_raw))):
            d = json.loads(db_raw[i])
            ref = []
            if 'references' in d:
                ref = d['references']
            auth = []
            if 'authors' in d:
                auth = d['authors']
            for j in auth:
                if i not in self.author_write[self.author_ids_map[j['id']]]:
                    self.author_write[self.author_ids_map[j['id']]].append(i)

                self.paper_author[i].append(self.author_ids_map[j['id']])
                for k in ref:
                    if k in self.paper_ids_map:
                        if self.paper_ids_map[k] not in self.author_ref[self.author_ids_map[j['id']]]:
                            self.author_ref[self.author_ids_map[j['id']]].append(self.paper_ids_map[k])
                        self.paper_ref_paper[i].append(self.paper_ids_map[k])

        for i in range(len(self.author_ids_map)):
            for j in range(len(self.author_write[i])):
                tmp_write = self.author_write[i][j]
                self.author_w_link[i].append(cnt)
                self.paper_link[tmp_write].append(cnt)
                self.edge_wt.append((i + len(self.paper_ids_map), tmp_write, cnt))
                cnt += 1

        for i in range(len(self.author_ids_map)):
            for j in range(len(self.author_ref[i])):
                tmp_ref = self.author_ref[i][j]
                if tmp_ref not in self.author_write[i]:
                    self.author_r_link[i].append(cnt)
                    self.paper_link[tmp_ref].append(cnt)
                    self.edge_ref.append((i + len(self.paper_ids_map), tmp_ref, cnt))
                    cnt += 1

        print('Count: ', cnt)

        self.edge_list = self.edge_wt + self.edge_ref

        self.writes = pd.DataFrame(self.edge_wt, columns=['idx_A', 'idx_P', 'gid']).set_index('gid')
        self.writes['rt'] = True
        self.refs = pd.DataFrame(self.edge_ref, columns=['idx_A', 'idx_P', 'gid']).set_index('gid')
        self.refs['rt'] = False
        self.links = pd.concat([self.writes, self.refs], axis=0)

        # self.links['cnt'] = self.links.groupby(['idx_A'])['idx_P'].transform('count')


        self.vec_auth = np.zeros((len(self.author_ids_map), 300))
        trange = tqdm.tqdm(range(len(self.author_ids_map)))
        for i in trange:
            sum_w = 0.
            for j in self.author_write[i]:
                if self.paper_w[j] == 0:
                    continue
                self.vec_auth[i, :] += (self.vec_paper[j, :] / self.paper_w[j])
                sum_w += self.paper_w[j]
            if sum_w != 0.:
                self.vec_auth[i, :] = self.vec_auth[i, :] / sum_w
        # self.vec_auth = np.load(self.dir + 'auth2vec.npy')

        self.data_split()
        self.build_graph()
        self.find_neighbors(0.2, 2000, 1000)

    def build_graph(self):
        self.g = dgl.DGLGraph()
        self.g.add_nodes(len(self.paper_ids_map) + len(self.author_ids_map))
        # write edges:
        src = self.writes['idx_A'].values
        dst = self.writes['idx_P'].values
        self.g.add_edges(src, dst,
                    data={'inv': torch.zeros(len(self.edge_wt), dtype=torch.uint8),
                          'wt': torch.ones(len(self.edge_wt), dtype=torch.uint8)})
        self.g.add_edges(dst, src,
                    data={'inv': torch.ones(len(self.edge_wt), dtype=torch.uint8),
                          'wt': torch.ones(len(self.edge_wt), dtype=torch.uint8)})
        # ref edges:
        src = self.refs['idx_A'].values
        dst = self.refs['idx_P'].values
        self.g.add_edges(src, dst,
                    data={'inv': torch.zeros(len(self.edge_ref), dtype=torch.uint8),
                          'wt': torch.zeros(len(self.edge_ref), dtype=torch.uint8)})
        self.g.add_edges(dst, src,
                    data={'inv': torch.ones(len(self.edge_ref), dtype=torch.uint8),
                          'wt': torch.zeros(len(self.edge_ref), dtype=torch.uint8)})
        self.g.ndata['year'] = torch.zeros(self.g.number_of_nodes(), dtype=torch.int64)
        self.g.ndata['year'][:len(self.paper_ids_map)] = \
            torch.LongTensor(self.papers['year'].cat.codes.values + 1)
        self.g.ndata['fos'] = torch.zeros((self.g.number_of_nodes(), 300))
        self.g.ndata['fos'][:len(self.paper_ids_map), :] = torch.from_numpy(self.vec_paper)
        self.g.ndata['fos'][len(self.paper_ids_map):, :] = torch.from_numpy(self.vec_auth)
        if len(self.AI_venues)>1:
            self.g.ndata['venue'] = torch.zeros(self.g.number_of_nodes(), dtype=torch.int64)
            self.g.ndata['venue'][:len(self.paper_ids_map)] = \
                torch.LongTensor(self.papers['venue'].cat.codes.values + 1)
        print(f"Verification: edge list: {len(self.edge_list)} g_edges: {self.g.number_of_edges()}")


    # def data_split(self):
    #     self.test_edges = []
    #     for i in range(len(self.author_ids_map)):
    #         author_edges = list(set(self.author_write[i] + self.author_ref[i]))
    #         if len(author_edges) >= 5:
    #             tmp = np.random.permutation(len(author_edges))[:len(author_edges) // 5]
    #             for j in tmp:
    #                 self.test_edges.append((i + len(self.paper_ids_map), author_edges[j]))
    #
    #     print(f"split {len(self.test_edges)} edges!")

    def split_user(self, df, filter_counts=False):
        df_new = df.copy()
        df_new['prob'] = 0

        if filter_counts:
            df_new_sub = (df_new['paper_count'] >= 10).nonzero()[0]
        else:
            df_new_sub = df_new['train'].nonzero()[0]
        prob = np.linspace(0, 1, df_new_sub.shape[0], endpoint=False)
        np.random.shuffle(prob)
        df_new['prob'].iloc[df_new_sub] = prob
        return df_new

    def data_split(self):
        ### Too Slow!!!!
        # self.links = self.links.groupby('idx_A', group_keys=False).apply(
        #         partial(self.split_user, filter_counts=True))
        # self.links['train'] = self.links['prob'] <= 0.8
        # self.links['valid'] = (self.links['prob'] > 0.8) & (self.links['prob'] <= 0.9)
        # self.links['test'] = self.links['prob'] > 0.9
        # self.links.drop(['prob'], axis=1, inplace=True)

        # to save edges ids
        self.train_edges = []
        self.test_edges = []
        self.valid_edges = []

        for i in tqdm.tqdm(range(len(self.author_ids_map))):
            author_edges = self.author_w_link[i] + self.author_r_link[i]
            if len(author_edges) >= 10:
                seg = len(author_edges) // 10
                np.random.shuffle(author_edges)
                self.train_edges += author_edges[:seg*8]
                self.valid_edges += author_edges[seg*8:seg*9]
                self.test_edges  += author_edges[seg*9:]
            else:
                self.train_edges += author_edges

        self.train_edges = set(self.train_edges)
        for i in tqdm.tqdm(range(len(self.paper_ids_map))):
            paper_edges = self.paper_link[i]
            sel = False
            for j in paper_edges:
                if j in self.train_edges:
                    sel = True
                    break
            if not sel:
                self.train_edges.add(paper_edges[np.random.randint(len(paper_edges))])

        self.train_edges = list(self.train_edges)
        self.valid_edges = np.setdiff1d(self.valid_edges, self.train_edges)
        self.test_edges = np.setdiff1d(self.test_edges, self.train_edges)
        self.train_edges = np.array(self.train_edges)
        self.test_edges = np.array(self.test_edges)
        self.valid_edges = np.array(self.valid_edges)
        train = np.zeros(len(self.links), dtype=int)
        test = np.zeros(len(self.links), dtype=int)
        valid = np.zeros(len(self.links), dtype=int)

        train[self.train_edges] = 1
        valid[self.valid_edges] = 1
        test[self.test_edges] = 1
        self.links['train'] = train > 0
        self.links['valid'] = valid > 0
        self.links['test'] = test > 0



    def find_neighbors(self, restart_prob, max_nodes, top_T, batch_size=3000):
        self.neighbor_probs = torch.zeros((self.g.number_of_nodes(), top_T))
        self.neighbors = torch.zeros((self.g.number_of_nodes(), top_T), dtype=torch.int)
        num_batch = self.g.number_of_nodes() // batch_size
        for i in tqdm.tqdm(range(num_batch)):
            tmp_probs, tmp_nb = randomwalk.random_walk_distribution_topt(
                self.g, self.g.nodes()[i * batch_size:(i + 1) * batch_size],
                restart_prob, max_nodes, top_T)
            self.neighbor_probs[i * batch_size:(i + 1) * batch_size, :] = tmp_probs
            self.neighbors[i * batch_size:(i + 1) * batch_size, :] = tmp_nb
        tmp_probs, tmp_nb = randomwalk.random_walk_distribution_topt(
            self.g, self.g.nodes()[(i + 1) * batch_size:],
            restart_prob, max_nodes, top_T)
        self.neighbor_probs[(i + 1) * batch_size:, :] = tmp_probs
        self.neighbors[(i + 1) * batch_size:, :] = tmp_nb
        # self.neighbor_probs = torch.load(self.dir+'neighbor_probs.pt')
        # self.neighbors = torch.load(self.dir+'neighbors.pt')

    def generate_mask(self, round=10):
        while True:
            train_masks = [[] for i in range(round)]
            for i in tqdm.tqdm(range(len(self.author_ids_map))):
                author_edges = self.author_w_link[i] + self.author_r_link[i]
                if len(author_edges) >= round:
                    seg = len(author_edges) // round
                    np.random.shuffle(author_edges)
                    for j in range(round-1):
                        train_masks[j] += author_edges[j*seg:(j+1)*seg]
                    train_masks[round-1] += author_edges[(round-1)* seg:]
                else:
                    tmp = np.random.permutation(range(len(author_edges)))
                    for j in range(len(tmp)):
                        train_masks[tmp[j]].append(author_edges[j])

            for line in train_masks:
                train_mask = np.intersect1d(self.train_edges,line)
                prior_mask = np.setdiff1d(self.train_edges,line)
                yield prior_mask, train_mask

    def refresh_mask(self):
        if not hasattr(self, 'masks'):
        #     self.masks = self.generate_mask()
        # prior_mask, train_mask = next(self.masks)
            self.masks = MP_mask(self.train_edges, self.author_w_link, self.author_r_link, self.paper_link)
            self.masks.start()

        self.prior_mask, self.train_mask = self.masks.getitem()

        #

        valid_tensor = torch.zeros(self.g.number_of_edges()//2, dtype=torch.uint8)
        test_tensor = torch.zeros(self.g.number_of_edges()//2, dtype=torch.uint8)
        train_tensor = torch.zeros(self.g.number_of_edges()//2, dtype=torch.uint8)
        prior_tensor = torch.zeros(self.g.number_of_edges()//2, dtype=torch.uint8)

        valid_tensor[self.valid_edges] = 1
        test_tensor[self.test_edges] = 1
        train_tensor[self.train_mask] = 1
        prior_tensor[self.prior_mask] = 1

        edge_data = {
                'prior': prior_tensor,
                'valid': valid_tensor,
                'test': test_tensor,
                'train': train_tensor,
                }

        self.g.edges[self.links['idx_A'].values, self.links['idx_P'].values].data.update(edge_data)
        self.g.edges[self.links['idx_P'].values, self.links['idx_A'].values].data.update(edge_data)



def main():
    db = ccf_ai('/data2/ruofan/MyFiles/GCN_ace/')
    with open('ccf-ai.pkl', 'wb') as f:
        pickle.dumps(db, f)

if __name__ == '__main__':
    main()



