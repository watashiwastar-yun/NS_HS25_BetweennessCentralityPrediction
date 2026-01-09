from networkit import *
import networkx as nx
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import pickle
import scipy.sparse as sp
import copy
import random
import numpy as np
import torch


def get_out_edges(g_nkit, range_nodes):
    global all_out_dict
    all_out_dict = dict()
    # Initialize for all nodes we care about
    for i in range_nodes:
        all_out_dict[i] = set()
        
    for i in range_nodes:
        # i is now the 0-based index in g_nkit
        g_nkit.forEdgesOf(i, nkit_outedges)
            
    return all_out_dict

def get_in_edges(g_nkit, range_nodes):
    global all_in_dict
    all_in_dict = dict()
    for i in range_nodes:
        all_in_dict[i] = set()
        
    for i in range_nodes:
        g_nkit.forInEdgesOf(i, nkit_inedges)
            
    return all_in_dict


def nkit_inedges(u, v, weight, edgeid):
    # u and v are indices in NetworKit
    # We want to store: for target u, v is a source
    if u in all_in_dict:
        all_in_dict[u].add(v)


def nkit_outedges(u, v, weight, edgeid):
    # u -> v
    if u in all_out_dict:
        all_out_dict[u].add(v)

    
def nx2nkit(g_nx, nodelist):
    # Map raw node IDs (e.g., 10429) to 0-based contiguous indices
    node_map = {node: i for i, node in enumerate(nodelist)}
    node_num = len(nodelist)
    
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1, e2 in g_nx.edges():
        if e1 in node_map and e2 in node_map:
            u = node_map[e1]
            v = node_map[e2]
            g_nkit.addEdge(u, v)
        
    return g_nkit


def clique_check(index, node_sequence, all_out_dict, all_in_dict):
    # Refactored to work with indices directly.
    # node_sequence is kept in arg list for compatibility but not used for ID lookup
    # index is the 0-based index of the node
    
    # In dictionary, keys are indices
    node_idx = index 
    
    if node_idx not in all_in_dict:
        # This node has no incoming edges, so it cannot be part of a clique structure
        # where an incoming node needs to connect to outgoing nodes.
        # Or, more likely, it means all_in_dict was not populated for this node.
        # If it's not in all_in_dict, it means it has no in-edges, so the loop over in_nodes won't run.
        # The original logic would have failed with KeyError.
        # If it has no in-edges, the condition "for in_n in in_nodes" is vacuously true,
        # so it would return True. Let's stick to that.
        in_nodes = set()
    else:
        in_nodes = all_in_dict[node_idx]

    if node_idx not in all_out_dict:
        # Similar logic for out-edges.
        out_nodes = set()
    else:
        out_nodes = all_out_dict[node_idx]

    for in_n in in_nodes:
        # Check if neighbors of 'in_n' contain all 'out_nodes' (except in_n itself)
        # Essentially checking transitivity: in -> node -> out implies in -> out
        
        tmp_out_nodes = set(out_nodes)
        tmp_out_nodes.discard(in_n)
        
        # Note: all_out_dict must contain in_n for this check to work
        # If in_n is not in the subset of nodes we computed edges for, this might fail.
        # But we compute for all nodes usually.
        
        if in_n not in all_out_dict:
             # If we don't have info for neighbor, we can't verify clique. Conservative: False?
             # Or assume it's NOT a clique?
             # Original logic assumed full access.
             # Let's assume we computed edges for relevant nodes.
             # If in_n has no outgoing edges, then tmp_out_nodes.issubset(empty_set) is only true if tmp_out_nodes is empty.
             # If tmp_out_nodes is not empty, then it's not a clique.
             if len(tmp_out_nodes) > 0:
                 return False
             
        elif not tmp_out_nodes.issubset(all_out_dict[in_n]):
            return False
    
    return True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def graph_to_adj(list_graph, list_n_sequence, list_node_num, model_size):
    
    list_adjacency = list()
    list_adjacency_t = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    
    for i in range(len(list_graph)):
        # convert to MultiDiGraph
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        # remove self loops
        self_loops = [i for i in nx.selfloop_edges(graph)]
        graph.remove_edges_from(self_loops)
        
        node_sequence = list_n_sequence[i]
        
        # get adjacency matrix
        adj_temp = nx.adjacency_matrix(graph, nodelist=node_sequence, weight='weight_AKA_average_time')
        
        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        # get degree - for TOPOLOGY only
        binary_adj = adj_temp.copy()
        binary_adj.data[:] = 1.0
        
        # arr_temp1 和 arr_temp2 应该是1维数组
        arr_temp1 = np.sum(binary_adj, axis=1).flatten()  # 添加 flatten()
        arr_temp2 = np.sum(binary_adj.transpose(), axis=1).flatten()  # 添加 flatten()
        
        # get degree matrix
        arr_multi = np.multiply(arr_temp1, arr_temp2)
        
        arr_multi = np.where(arr_multi > 0, 1.0, 0.0)
        
        degree_arr = arr_multi  # 这是1维数组
        
        # 修改这里：确保 degree_arr 是1维数组
        if degree_arr.ndim == 2:
            degree_arr = degree_arr.flatten()
        
        non_zero_ind = np.nonzero(degree_arr)[0]  # 非零索引
        
        # convert to nkit graph
        g_nkit = nx2nkit(graph, node_sequence)
        
        # get edge dictionaries
        all_out_dict = get_out_edges(g_nkit, range(len(node_sequence)))
        all_in_dict = get_in_edges(g_nkit, non_zero_ind)
        
        # 修改这里：使用1维索引
        for index in non_zero_ind:
            is_zero = clique_check(index, node_sequence, all_out_dict, all_in_dict)
            if is_zero == True:
                degree_arr[index] = 0.0  # 改为1维索引
        
        # 将 degree_arr 转换为对角矩阵用于乘法
        degree_diag = sp.diags(degree_arr, format='csr')
        adj_temp = adj_temp.dot(degree_diag)  # 矩阵乘法
        adj_temp_t = degree_diag.dot(adj_temp_t)  # 矩阵乘法
        
        # 也可以使用点乘，但矩阵乘法更高效
        # adj_temp = adj_temp.multiply(degree_arr.reshape(-1, 1))
        # adj_temp_t = adj_temp_t.multiply(degree_arr.reshape(1, -1))

        rand_pos = 0
        top_mat = csr_matrix((rand_pos, rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind, remain_ind))
        
        list_rand_pos.append(rand_pos)
        
        # adding extra padding to adj mat
        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat, adj_temp, bottom_mat))
        
        adj_temp_t = csr_matrix(adj_temp_t)
        adj_mat_t = sp.block_diag((top_mat, adj_temp_t, bottom_mat))
        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_t = sparse_mx_to_torch_sparse_tensor(adj_mat_t)
        list_adjacency_t.append(adj_mat_t)
    
    return list_adjacency, list_adjacency_t

def ranking_correlation(y_out,true_val,node_num,model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()


    kt,_ = kendalltau(predict_arr[:node_num],true_arr[:node_num])

    return kt


def loss_cal(y_out, true_val, num_nodes, device, model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))
    
    # 确保 order_y_true 是1维数组
    _, order_y_true = torch.sort(-true_val[:num_nodes])
    order_y_true = order_y_true.flatten()  # 添加这行确保是1维
    
    sample_num = num_nodes * 20
    
    ind_1 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)
    ind_2 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)
    
    rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()
    
    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)
    
    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)
    
    return loss_rank

