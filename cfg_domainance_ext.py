import argparse
import networkx as nx
import pickle
from queue import Queue
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def load_cfg(cfg_path):
    with open(cfg_path, 'rb') as fd:
        g:nx.DiGraph = pickle.load(fd)
    return g

def get_domain_set(g:nx.DiGraph):
    print(g)
    num_syscall = 0
    main_node = None
    syscall_list = []

    for nd, data in g.nodes(data=True):
        if data['main'] == 1: 
            main_node = nd
            print(f'main node is : {nd}.')
        if data['syscall'] == 1: 
            syscall_list.append(nd)
            num_syscall += 1
    print(f'totally {num_syscall} syscall instruction basic blocks find.')

    dom = nx.algorithms.dominance.immediate_dominators(g, main_node)
    domain_set = set()

    for nd in syscall_list:
        while nd != main_node:
            domain_set.add(nd) 
            nd = dom[nd]

    domain_set.add(main_node)
    return domain_set

def sample_first_order_subgraph(target_nodes, g:nx.DiGraph):
    sampled_nodes = set()
    for node in target_nodes:
        predecessors = list(g.predecessors(node))  
        successors = list(g.successors(node))    

        sampled_nodes.add(node)
        sampled_nodes.update(predecessors)
        sampled_nodes.update(successors)

    # 从图中提取包含所有采样节点的子图
    subgraph = g.subgraph(sampled_nodes).copy()
    return subgraph

# Graph-Skeleton: ~1% Nodes are Sufficient to Represent Billion-Scale Graph 
# https://github.com/caolinfeng/GraphSkeleton/blob/master/skeleton_compress/src/skeleton.cc#L349  get_bridge_and_corr_mask
# BFS + shortest path
# d1: bridge node
# d2: neighbor node
def graph_sketch(g:nx.DiGraph, target_nodes, d1, d2):
    g_direct = g

    g = g.to_undirected()
    vis, prev, dist = {}, {}, {}

    dist = {nd: d1+1 for nd in g.nodes()}
    vis = {nd: 0 for nd in g.nodes()}

    bridge_nodes, neighbor_nodes = set(), set()
    q = Queue()
    for nd in target_nodes: q.put((nd, nd, 0))

    # BFS 由近到远遍历
    while not q.empty():
        u, s, d = q.get()
        for v in g.neighbors(u):
            if v in target_nodes: continue
            if vis[v] == 0:
                vis[v] = 1
                dist[v] = d + 1 
                prev[v] = s 
                q.put((v, s, d+1))

                if dist[v] <= d2: neighbor_nodes.add(v)

            elif vis[v] == 1 and prev[v] != s:
                vis[v] = 2
                q.put((v, s, d+1))

                if d+1 + dist[v] <= d1:
                    neighbor_nodes.discard(v) 
                    bridge_nodes.add(v) 

    sample_nodes = bridge_nodes | neighbor_nodes | target_nodes
    print(f'bridge nodes num : {len(bridge_nodes)} , and neighbor nodes num (except bridge nodes) : {len(neighbor_nodes)}')
    sg = g_direct.subgraph(sample_nodes).copy()
    return sg


def extent_domain_node(domain_set, g:nx.DiGraph):

    sg1 = sample_first_order_subgraph(domain_set, g)
    print(f'1-order subgraph: {sg1}')
    # sg = sample_first_order_subgraph(set(sg.nodes()), g)

    # graph sketch
    d1 = 4 
    d2 = 1 
    sg = graph_sketch(g, domain_set, d1, d2)
    print(f'graph sketch: {sg}')
    return sg

def visual_node_emb(node_embeddings, node_names, target_nodes, png_path):
    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)
    plt.figure(figsize=(10, 8))

    colors = ['red' if node in target_nodes else 'blue' for node in node_names]
    plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c=colors, alpha=0.7, edgecolors='k')
    plt.savefig(png_path)

def node_embedding_learning(g:nx.DiGraph, target_nodes, cfg_path):

    g = g.to_undirected()

    model = Node2Vec(g, dimensions=64, walk_length=10, num_walks=10, workers=1).fit()
    target_embeddings = {node: model.wv[node] for node in target_nodes}

    node_embeddings = np.array([model.wv[node] for node in model.wv.index_to_key])
    node_names = model.wv.index_to_key
    png_path = cfg_path.replace('.pkl', '_emb.png')
    visual_node_emb(node_embeddings, node_names, target_nodes, png_path)

    emb_path = cfg_path.replace('.pkl', '.emb')
    with open(emb_path, 'wb') as fd:
        pickle.dump(target_embeddings, fd)

    return target_embeddings

def arg_init():
    parser = argparse.ArgumentParser(description="CFG domain point extract and embedding learning")
    parser.add_argument("cfg_path", help="cfg path")
    args = parser.parse_args()

    cfg_path = args.cfg_path
    return cfg_path

def main(cfg_path):
    g = load_cfg(cfg_path)
    domain_set = get_domain_set(g)
    sg = extent_domain_node(domain_set, g)

    # node2vec
    target_embeddings = node_embedding_learning(sg, domain_set, cfg_path)

if __name__ == '__main__':
    cfg_path = arg_init()
    main(cfg_path)
