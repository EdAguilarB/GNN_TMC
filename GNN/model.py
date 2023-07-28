import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(23)
    
    
class GCN_original(torch.nn.Module):
    def __init__(self, num_features, embedding_size, improved):
        # Init parent
        super(GCN_original, self).__init__()
        torch.manual_seed(23)

        # GCN layers
        self.initial_conv = GCNConv(num_features, 
                                    embedding_size, 
                                    improved=improved)
        
        self.conv1 = GCNConv(embedding_size, embedding_size, improved=improved)
        self.conv2 = GCNConv(embedding_size, embedding_size, improved=improved)
        self.conv3 = GCNConv(embedding_size, embedding_size, improved=improved)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index, edge_weight = None):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index, edge_weight)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index, edge_weight)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index, edge_weight)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index, edge_weight)
        hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden
    


class GCN_loop(torch.nn.Module):
    def __init__(self, num_features, embedding_size, gnn_layers, improved):
        # Init parent
        super(GCN_loop, self).__init__()
        torch.manual_seed(23)

        self.gnn_layers = gnn_layers

        # GCN layers
        self.initial_conv = GCNConv(num_features, 
                                    embedding_size, 
                                    improved=improved)
        

        self.conv_layers = ModuleList([])
        for _ in range(self.gnn_layers - 1):
            self.conv_layers.append(GCNConv(embedding_size,
                                            embedding_size,
                                            improved=improved))

        # Output layer
        self.out = Linear(embedding_size*2, 1)


    def forward(self, x, edge_index, batch_index, edge_weight = None):

        x = x.float()

        # First Conv layer
        hidden = self.initial_conv(x, edge_index, edge_weight)
        hidden = F.tanh(hidden)

        # Other Conv layers
        for i in range(self.gnn_layers-1):
            hidden = self.conv_layers[i](hidden, edge_index, edge_weight)
            hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out


class GAT_original(torch.nn.Module):
    def __init__(self, num_features, embedding_size, heads, concat=True,dropout=0.1):
        # Init parent
        super(GAT_original, self).__init__()
        torch.manual_seed(23)

        # GCN layers
        self.initial_conv = GATConv(num_features, 
                                    embedding_size, 
                                    heads=heads,
                                    concat=concat,
                                    dropout=dropout)
        
        self.conv1 = GCNConv(embedding_size*heads, embedding_size, heads = heads, concat = concat, dropout=dropout)
        self.conv2 = GCNConv(embedding_size*heads, embedding_size, heads = heads, concat = concat, dropout=dropout)
        self.conv3 = GCNConv(embedding_size*heads, embedding_size, heads = heads, concat = concat, dropout=dropout)

        # Output layer
        self.out = Linear(embedding_size*2*heads, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out
    

class GAT_loop(torch.nn.Module):
    def __init__(self, num_features, embedding_size, heads, gnn_layers, concat=True,dropout=0.1):
        # Init parent
        super(GAT_loop, self).__init__()
        torch.manual_seed(23)

        self.gnn_layers = gnn_layers

        self.initial_conv = GATConv(num_features, 
                                    embedding_size, 
                                    heads=heads,
                                    concat=concat,
                                    dropout=dropout)
        

        self.conv_layers = ModuleList([])
        for _ in range(self.gnn_layers - 1):
            self.conv_layers.append(GATConv(embedding_size*heads, 
                                            embedding_size, 
                                            heads = heads, 
                                            concat = concat, 
                                            dropout=dropout))

        # Output layer
        self.out = Linear(embedding_size*2*heads, 1)


    def forward(self, x, edge_index, batch_index):

        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        for i in range(self.gnn_layers-1):
            hidden = self.conv_layers[i](hidden, edge_index)
            hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden