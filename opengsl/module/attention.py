import torch
from opengsl.module.functional import normalize, symmetry, knn, enn, apply_non_linearity, removeselfloop
import torch.nn as nn
import torch.nn.functional as F
import math
# torch.autograd.set_detect_anomaly(True)

class Unified_attention(nn.Module):
    """
        A unified attention mechanism for GSL ans GT.
        Four steps:
            1. Similarity Computation
            2. Sparsification.
            3. fuse Q
            3. Normalization.
    """
    def __init__(self, n_nodes, Q, conf=None):
        super(Unified_attention, self).__init__()
        if getattr(conf, 'unify', None) is None:
            conf.unify = {}
        self.conf = conf
        self.use_attention = conf.unify.get('use_attention', False)
        self.use_similarity = conf.unify.get('use_similarity', False)
        self.update_beta = conf.unify.get('update_beta', False)
        self.use_fuse = conf.unify.get('use_fuse', False)
        self.use_norm = conf.unify.get('use_norm', False)
        self.custom_similarity = lambda z: z@z.T
        self.custom_sparsify = lambda adj: adj
        self.custom_fuse = lambda adj, original_adj: adj
        self.custom_norm = lambda adj: adj
        '''
            def custom_func(): xxx
            self.custom_attention = custom_func
        '''
        
        self.gamma = conf.unify.get('gamma', 2)
        self.gamma_star = self.gamma / (self.gamma - 1)
        self.eps = 1e-6
        
        self.alpha = conf.unify.get('alpha', 0)
        if conf.dataset.get('add_loop', False):
            Q = Q + torch.eye(Q.shape[0],device='cuda').to_sparse()
        self.Q = normalize(Q, style='row')
        
        if self.use_attention and self.use_similarity:
            self.diag = conf.unify.get('diag', False)
            dim = conf.unify['dim']
            if self.diag:
                self.W = nn.Parameter(torch.ones(dim, device='cuda'))
                self.W.data.fill_(1.0)
            else:
                self.W = nn.Parameter(torch.empty(dim, dim, device='cuda'))
                nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
            self.optimizer_W = torch.optim.Adam([
                {'params': self.W, 'lr': conf.training['lr_dae'], 'weight_decay': 0}
            ])
        
        if self.use_attention and self.update_beta:
            self.beta_vector = nn.Parameter(
                torch.full(
                    size=(n_nodes, 1),
                    fill_value=conf.unify.get('beta_init', 0.5),
                    dtype=torch.float32,
                    device=torch.device("cuda")
                )
            )
            self.optimizer_beta  = torch.optim.SGD([
                {'params': self.beta_vector, 'lr': conf.unify['lr_beta'], 'weight_decay': 0}
            ])

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
        if self.use_attention and self.update_beta:
            self.beta_vector.data.fill_(self.conf.unify.get('beta_init', 0.5))
        if self.use_attention and self.use_similarity:
            if self.diag:
                with torch.no_grad():
                    self.W.fill_(1.0)
            else:
                nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
            
    def step(self):
        if self.use_attention and self.use_similarity:
            self.optimizer_W.step()
        
    def add_Q(self, adj):
        if self.alpha == 1:
            adj = adj * self.Q
        elif self.alpha == 0:
            adj = adj
        else:
            adj = (1 - self.alpha) * adj + self.alpha * (adj * self.Q.to_dense())

        adj = adj / adj.shape[0]
        return adj

    def relu_with_eps(self, tensor):
        if tensor.is_sparse:
            if not tensor.is_coalesced(): tensor = tensor.coalesce()
            values = tensor.values()
            new_values = F.relu(values) + self.eps
            return torch.sparse_coo_tensor(tensor.indices(), new_values, tensor.shape).coalesce()
        else:
            return F.relu(tensor) + self.eps
    
    def similarity(self, embdings):
        if self.use_attention and self.use_similarity:
            if self.training:
                self.optimizer_W.zero_grad()
            if self.diag:
                weighted = embdings * self.W
            else:
                weighted = embdings @ self.W
            sim = weighted @ embdings.T
        else:
            sim = self.custom_similarity(embdings)
        return sim
    
    def sparsify(self, adj):
        if self.use_attention and self.update_beta:
            adj = torch.relu(adj - self.beta_vector.expand(-1, adj.shape[1]))
        else:
            adj = self.custom_sparsify(adj)
        if self.use_attention:
            adj = self.relu_with_eps(adj)
            if self.update_beta: self.beta_update(adj)
            adj = adj ** (1 / (self.gamma - 1))
        return adj
    
    def fuse(self, adj, original_adj=None):
        if self.use_attention and self.use_fuse:
            adj = self.add_Q(adj)
        else:
            adj = self.custom_fuse(adj, original_adj)
        return adj
    
    def norm(self, adj):
        if self.use_attention and self.use_norm:
            adj = normalize(adj, style='row')
        else:
            adj = self.custom_norm(adj)
        return adj
        
    def beta_update(self, adj):
        adj = adj.detach()
        loss_beta = (self.add_Q(adj.to_dense() ** self.gamma_star).sum(dim=-1).pow(1 / self.gamma_star) + self.beta_vector.squeeze()).sum()        
        
        if self.training:
            self.optimizer_beta.zero_grad()
            loss_beta.backward()
            self.optimizer_beta.step()
        
        return loss_beta