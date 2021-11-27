from torch import nn, Tensor
    
class DivergenceKL1D(nn.Module):
    
    def __init__(
        self, 
        size: int, 
        use_softmax: bool = True
    ):
        super().__init__() 
        self.size = size
        self.kl_divergence = nn.KLDivLoss(
            reduction="batchmean"
        )
        self.softmax = nn.Softmax(
            dim = 1
        ) if use_softmax else nn.Identity()
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.view(-1, self.size)
        input = self.softmax(input)
        target = target.view(-1, self.size)
        target = self.softmax(target)
        return self.kl_divergence(input.log(), target)
    
    
class DivergenceJS1D(nn.Module):
    
    def __init__(
        self, 
        size: int, 
        use_softmax: bool = True
    ):
        super().__init__() 
        self.size = size
        
        self.kl_divergence = nn.KLDivLoss(
            reduction="batchmean"
        )
        self.softmax = nn.Softmax(
            dim = 1
        ) if use_softmax else nn.Identity()
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.view(-1, self.size)
        input = self.softmax(input)
        target = target.view(-1, self.size)
        target = self.softmax(target)
        mean = 0.5 * (input + target)
        kl_input_mean = self.kl_divergence(input.log(), mean) 
        kl_target_mean = self.kl_divergence(target.log(), mean)
        return 0.5 * (kl_input_mean + kl_target_mean)