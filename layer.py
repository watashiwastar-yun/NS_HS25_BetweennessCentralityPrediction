import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F



class GNN_Layer(Module):
    """
    Layer defined for GNN-Bet
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class GNN_Layer_Init(Module):
    """
    First layer of GNN_Init, for embedding lookup
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer_Init, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        support = self.weight
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GNN_Layer_Init_WithFeatures(Module):
    """
    First layer of GNN with node features support.
    Combines learnable positional embeddings with input node features.
    """

    def __init__(self, in_features, node_feat_dim, out_features, bias=True):
        super(GNN_Layer_Init_WithFeatures, self).__init__()
        self.in_features = in_features  # model_size (for positional embedding)
        self.node_feat_dim = node_feat_dim  # number of node features
        self.out_features = out_features
        
        # Learnable positional embedding (like original GNN_Layer_Init)
        self.pos_weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        # Weight matrix for node features
        self.feat_weight = Parameter(torch.FloatTensor(node_feat_dim, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.pos_weight.size(1))
        self.pos_weight.data.uniform_(-stdv, stdv)
        self.feat_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, node_features):
        """
        Args:
            adj: sparse adjacency matrix (model_size, model_size)
            node_features: node feature matrix (model_size, node_feat_dim)
        Returns:
            output: (model_size, out_features)
        """
        # Positional embedding path (structure-based)
        pos_support = self.pos_weight
        pos_output = torch.spmm(adj, pos_support)
        
        # Node feature path (feature-based)
        feat_output = torch.mm(node_features, self.feat_weight)
        
        # Combine both paths
        output = pos_output + feat_output
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + '+' + str(self.node_feat_dim) + ' -> ' \
               + str(self.out_features) + ')'