import paddle

import pgl 

import pandas as pd
import numpy as np
import random


# 从数据中读取边信息
def load_edges(path, num_nodes, self_loop=True, add_inverse_edge=True):
    # 读取数据
    edges = pd.read_csv(path).values

    # 是否是双向连接， 即有向图还是无向图
    if add_inverse_edge:
        edges = np.vstack([edges, edges[:, ::-1]])

    # 引入自连接， 1 <-> 1, 2 <-> 2
    if self_loop:
        src = np.arange(0, num_nodes)
        dst = np.arange(0, num_nodes)
        self_loop = np.vstack([src, dst]).T
        edges = np.vstack([edges, self_loop])
    
    return edges


# 加载训练数据，切分验证集
def load_train(path, split_ratio=0.8):
    # 读取数据
    df = pd.read_csv(path)
    ids = df['nid'].values
    labels = df['label'].values 

    # 划分训练集、验证集索引
    index = list(range(len(ids)))
    train_index = random.sample(index, int(len(ids) * split_ratio))
    eval_index = list(set(index) - set(train_index))
    
    # 分割训练集、验证集，转为tensor
    train_ids = paddle.to_tensor(ids[train_index])
    train_labels = paddle.to_tensor(labels[train_index])
    eval_ids = paddle.to_tensor(ids[eval_index])
    eval_labels = paddle.to_tensor(labels[eval_index])

    return train_ids, train_labels, eval_ids, eval_labels


# 加载预测数据
def load_test(path):
    df = pd.read_csv(path)
    return paddle.to_tensor(df['nid'].values)


# 创建图
def build_graph(config):
    # 读取节点特征
    node_feat = np.load(config.feat)
    # 获取节点数
    num_nodes = node_feat.shape[0]  
    # 读取边信息
    edges = load_edges(config.edges, num_nodes)
    # 创建图
    graph = pgl.Graph(num_nodes=num_nodes, edges=edges, node_feat={'feat': node_feat})

    return graph










    