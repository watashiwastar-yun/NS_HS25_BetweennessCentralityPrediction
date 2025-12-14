import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer
from layer import GNN_Layer_Init
import torch 


class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout):
        super(GNN_Bet, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput,nhid)
        self.gc2 = GNN_Layer(nhid,nhid)
        self.gc3 = GNN_Layer(nhid,nhid)
        self.gc4 = GNN_Layer(nhid,nhid)

        self.dropout = dropout

        self.linear1 = nn.Linear(nhid,2*nhid)
        self.linear2 = nn.Linear(2*nhid,2*nhid)
        self.linear3 = nn.Linear(2*nhid,1)



    def forward(self,adj1,adj2):

        #Layers for aggregation operation
        x_1 = F.normalize(F.relu(self.gc1(adj1)),p=2,dim=1)
        x2_1 = F.normalize(F.relu(self.gc1(adj2)),p=2,dim=1)


        x_2 = F.normalize(F.relu(self.gc2(x_1, adj1)),p=2,dim=1)
        x2_2 = F.normalize(F.relu(self.gc2(x2_1, adj2)),p=2,dim=1)


        x_3 = F.normalize(F.relu(self.gc3(x_2,adj1)),p=2,dim=1)
        x2_3 = F.normalize(F.relu(self.gc3(x2_2,adj2)),p=2,dim=1)

        
        x_4 = F.relu(self.gc4(x_3,adj1))
        x2_4 = F.relu(self.gc4(x2_3,adj2))

        #Score Calculations
        #to-do: make a MLP layer and import here
        score1_1 = F.relu(self.linear1(x_1))
        score1_1 = F.dropout(score1_1,self.dropout)
        score1_1 = F.relu(self.linear2(score1_1))
        score1_1 = F.dropout(score1_1,self.dropout)
        score1_1 = self.linear3(score1_1)

        
        score1_2 = F.relu(self.linear1(x_2))
        score1_2 = F.dropout(score1_2,self.dropout)
        score1_2 = F.relu(self.linear2(score1_2))
        score1_2 = F.dropout(score1_2,self.dropout)
        score1_2 = self.linear3(score1_2)
        
        score1_3 = F.relu(self.linear1(x_3))
        score1_3 = F.dropout(score1_3,self.dropout)
        score1_3 = F.relu(self.linear2(score1_3))
        score1_3 = F.dropout(score1_3,self.dropout)
        score1_3 = self.linear3(score1_3)
        
        
        score1_4 = F.relu(self.linear1(x_4))
        score1_4 = F.dropout(score1_4,self.dropout)
        score1_4 = F.relu(self.linear2(score1_4))
        score1_4 = F.dropout(score1_4,self.dropout)
        score1_4 = self.linear3(score1_4)
        


        score2_1 = F.relu(self.linear1(x2_1))
        score2_1 = F.dropout(score2_1,self.dropout)
        score2_1 = F.relu(self.linear2(score2_1))
        score2_1 = F.dropout(score2_1,self.dropout)
        score2_1 = self.linear3(score2_1)

        
        score2_2 = F.relu(self.linear1(x2_2))
        score2_2 = F.dropout(score2_2,self.dropout)
        score2_2 = F.relu(self.linear2(score2_2))
        score2_2 = F.dropout(score2_2,self.dropout)
        score2_2 = self.linear3(score2_2)

        score2_3 = F.relu(self.linear1(x2_3))
        score2_3 = F.dropout(score2_3,self.dropout)
        score2_3 = F.relu(self.linear2(score2_3))
        score2_3 = F.dropout(score2_3,self.dropout)
        score2_3 = self.linear3(score2_3)
        
        
        score2_4 = F.relu(self.linear1(x2_4))
        score2_4 = F.dropout(score2_4,self.dropout)
        score2_4 = F.relu(self.linear2(score2_4))
        score2_4 = F.dropout(score2_4,self.dropout)
        score2_4 = self.linear3(score2_4)
        
        
        
        score1 = score1_1 + score1_2 + score1_3 + score1_4
        score2 = score2_1 + score2_2 + score2_3 + score2_4

        x = torch.mul(score1,score2)

        return x



class FlexibleGNN_Bet(nn.Module):
    """
    Flexible GNN model for betweenness centrality prediction.
    Supports any number of GNN layers (n_layers >= 1).
    
    Args:
        n_layers (int): Number of GNN layers (must be >= 1)
        ninput (int): Input dimension (should be model_size)
        nhid (int): Hidden dimension for GNN layers
        dropout (float): Dropout rate for MLP layers
    """
    def __init__(self, n_layers=4, ninput=15000, nhid=20, dropout=0.6):
        super(FlexibleGNN_Bet, self).__init__()
        
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Create GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer: GNN_Layer_Init (special initialization)
        self.gnn_layers.append(GNN_Layer_Init(ninput, nhid))
        
        # Subsequent layers: standard GNN_Layer
        for _ in range(1, n_layers):
            self.gnn_layers.append(GNN_Layer(nhid, nhid))
        
        # MLP layers for score computation
        self.linear1 = nn.Linear(nhid, 2 * nhid)
        self.linear2 = nn.Linear(2 * nhid, 2 * nhid)
        self.linear3 = nn.Linear(2 * nhid, 1)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, adj1, adj2):
        """
        Forward pass of the flexible GNN.
        
        Args:
            adj1 (torch.Tensor): Adjacency matrix for forward graph (A)
            adj2 (torch.Tensor): Adjacency matrix for transpose graph (A^T)
            
        Returns:
            torch.Tensor: Predicted betweenness centrality scores
        """
        # Lists to store outputs from each layer
        x_outputs = []  # For forward graph
        x2_outputs = [] # For transpose graph
        
        # First layer processing (special case)
        # Apply GNN layer, ReLU activation, and L2 normalization
        x = F.normalize(F.relu(self.gnn_layers[0](adj1)), p=2, dim=1)
        x2 = F.normalize(F.relu(self.gnn_layers[0](adj2)), p=2, dim=1)
        x_outputs.append(x)
        x2_outputs.append(x2)
        
        # Intermediate layers (2nd to (n-1)th layer)
        for i in range(1, self.n_layers - 1):
            # Apply GNN layer, ReLU activation, and L2 normalization
            x = F.normalize(F.relu(self.gnn_layers[i](x_outputs[-1], adj1)), p=2, dim=1)
            x2 = F.normalize(F.relu(self.gnn_layers[i](x2_outputs[-1], adj2)), p=2, dim=1)
            x_outputs.append(x)
            x2_outputs.append(x2)
        
        # Last layer (no normalization)
        if self.n_layers > 1:
            x = F.relu(self.gnn_layers[-1](x_outputs[-1], adj1))
            x2 = F.relu(self.gnn_layers[-1](x2_outputs[-1], adj2))
            x_outputs.append(x)
            x2_outputs.append(x2)
        
        # Initialize score accumulators
        score1 = 0  # Score for forward graph
        score2 = 0  # Score for transpose graph
        
        # Compute scores from each layer
        for i in range(self.n_layers):
            # Get output from i-th layer
            x_layer = x_outputs[i]
            x2_layer = x2_outputs[i]
            
            # Compute score for forward graph
            s1 = F.relu(self.linear1(x_layer))
            s1 = F.dropout(s1, self.dropout)
            s1 = F.relu(self.linear2(s1))
            s1 = F.dropout(s1, self.dropout)
            s1 = self.linear3(s1)
            score1 = score1 + s1
            
            # Compute score for transpose graph
            s2 = F.relu(self.linear1(x2_layer))
            s2 = F.dropout(s2, self.dropout)
            s2 = F.relu(self.linear2(s2))
            s2 = F.dropout(s2, self.dropout)
            s2 = self.linear3(s2)
            score2 = score2 + s2
        
        # Final prediction: element-wise multiplication
        final_score = torch.mul(score1, score2)
        
        return final_score
    
    def get_model_info(self):
        """
        Get model information including parameter count.
        
        Returns:
            dict: Model information
        """
        info = {
            'n_layers': self.n_layers,
            'total_params': self.total_params,
            'gnn_params': sum(p.numel() for p in self.gnn_layers.parameters()),
            'mlp_params': sum(p.numel() for p in [self.linear1, self.linear2, self.linear3].parameters()),
        }
        return info


def create_gnn_model_variant(n_layers, ninput=15000, nhid=20, dropout=0.6, variant_name=""):
    """
    Factory function to create GNN model variants.
    
    Args:
        n_layers (int): Number of GNN layers
        ninput (int): Input dimension
        nhid (int): Hidden dimension
        dropout (float): Dropout rate
        variant_name (str): Name for model variant
        
    Returns:
        FlexibleGNN_Bet: Configured GNN model
    """
    model = FlexibleGNN_Bet(
        n_layers=n_layers,
        ninput=ninput,
        nhid=nhid,
        dropout=dropout
    )
    
    # Add model variant name as attribute
    model.variant_name = variant_name or f"GNN_{n_layers}L"
    
    return model

