from typing import Optional, List, Tuple 

import torch 
from torch import nn,Tensor


class GraphConv(nn.Module):
    
    """
    Graph convolution class. Provides 3 ways to use it:
    - default: both X and A are provided during forward, only W is learned.
    - auto_graph: only X provided during forward, both A and W are learned. 
    - custom_graph: only X provided during forward, A is provided at init, W is learned.
      Useful to share weights between convolutions. 
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int = 0,
        custom_graph: nn.Parameter = None,
        use_custom_graph: bool = False,
        use_auto_graph: bool = False,
        use_activation: bool = True
    ) -> None:
        super().__init__()
        if use_auto_graph: assert num_nodes > 0, "num_nodes must be greater than 0 when using use_auto_graph." 
        if use_custom_graph: assert custom_graph is not None, "custom graph must be set when using use_custom_graph."
        self.use_auto_graph = use_auto_graph
        self.use_custom_graph = use_custom_graph
        
        self.graph = nn.Parameter(torch.eye(num_nodes)) if use_auto_graph else None 
        self.graph = custom_graph if use_custom_graph else self.graph 
        
        self.block = nn.Sequential(
            nn.Linear(
                in_features = in_features,
                out_features = out_features
            ),
            nn.ReLU(
                inplace = True
            ) if use_activation else nn.Identity()
        )
        
    def forward(self, X: Tensor, A: Optional[Tensor] = None) -> Tensor:
        """
        X: (B, N, in_features) input nodes features.
        A: (B, N, N) adjacency matrix.
        Output: (B, N, out_features) output nodes features.
        """
        B = X.shape[0]
        if self.use_auto_graph or self.use_custom_graph: A = self.graph
        else: assert A is not None, "missing adjacency matrix as second parameter."
        A = A.unsqueeze(0).repeat(B, 1, 1) # Use same graph for entire batch.
        A = self.laplacian(A) # Normalize graph.
        X = torch.bmm(A, X) # Merge neighboring nodes features.
        X = self.block(X) # Learn something. 
        return X 
            
    def laplacian(self, A: Tensor) -> Tensor:
        """
        A: (B, N, N) adjacency matrix.
        Output: (B, N, N) laplacian normalized adjacency matrix.
        """
        B, N, N = A.shape 
        D_hat = (torch.sum(A, dim=1) + 1e-5) ** (-0.5)
        L = D_hat.view(B, N, 1) * A * D_hat.view(B, 1, N)
        return L


class GraphPool(nn.Module):

    """
    Learnable pooling operation that changes the number of nodes in the graph.
    """

    def __init__(
        self, 
        in_nodes: int, 
        out_nodes: int
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_nodes, 
            out_features=out_nodes
        )

    def forward(self, X: Tensor) -> Tensor:
        """
        X: (B, in_nodes, in_features) input nodes features.
        Output: (B, out_nodes, in_features) output nodes features.
        """
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X
    

class GraphNet(nn.Module):
    
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        blocks_features: List[int],
        num_nodes: int,       
    ) -> None:
        super().__init__()
        num_blocks = len(blocks_features) + 1
        last_block = num_blocks - 1
        
        self.blocks = nn.Sequential(*[
            GraphConv(
                in_features = blocks_features[i-1] if i > 0 else in_features,
                out_features = blocks_features[i] if i < last_block else out_features,
                num_nodes = num_nodes,
                use_auto_graph = True,
                use_activation = (i < last_block)
            ) for i in range(num_blocks)
        ])
        
    def forward(self, X: Tensor) -> Tensor:
        return self.blocks(X) 
    

class GraphUNet(nn.Module):
    
    def __init__(
        self, 
        in_features: Tuple[int, int], 
        out_features: Tuple[int, int],
        blocks_features: List[Tuple[int, int]],
        bottleneck_size: int 
    ) -> None:
        super().__init__()
        
        num_blocks = len(blocks_features) 
        last_block = num_blocks - 1
        
        # Initialize graphs adjacency parameters 
        graphs = [nn.Parameter(torch.eye(in_features[0]))]
        graphs += [nn.Parameter(torch.eye(num_nodes)) for num_nodes, _ in blocks_features]
        graphs = nn.ParameterList(graphs)
        
        # Build encoder and decoder 
        encoder = []
        decoder = []
        
        for i in range(num_blocks):
            encoder += [GraphConv(
                in_features = blocks_features[i-1][1] if i > 0 else in_features[1],
                out_features = blocks_features[i][1],
                custom_graph = graphs[i],
                use_custom_graph = True
            )]
            encoder += [GraphPool(
                in_nodes = blocks_features[i-1][0] if i > 0 else in_features[0],
                out_nodes = blocks_features[i][0]
            )]
            decoder += [GraphPool(
                in_nodes = blocks_features[num_blocks-i-1][0],
                out_nodes = blocks_features[num_blocks-i-2][0] if i < last_block else out_features[0]
            )]
            decoder += [GraphConv(
                in_features = blocks_features[num_blocks-i-1][1], 
                out_features = blocks_features[num_blocks-i-2][1] if i < last_block else out_features[1],
                custom_graph = graphs[num_blocks-i-1],
                use_custom_graph = True,
                use_activation = (i < last_block)
            )]
            
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        # Build bottleneck 
        self.bottleneck = nn.Sequential(
            nn.Linear(
                in_features = blocks_features[last_block][1],
                out_features = bottleneck_size
            ),
            nn.ReLU(inplace = True),
            nn.Linear(
                in_features = bottleneck_size,
                out_features = blocks_features[last_block][1]
            ),
            nn.ReLU(inplace = True),
        )
        
        
    def forward(self, X: Tensor) -> Tensor:
        X = self.encoder(X)
        X = self.bottleneck(X)
        X = self.decoder(X)
        return X 
        
    