import torch
from torch.nn import functional as F
from opengsl.module.functional import normalize
from opengsl.module.encoder import GCNDiagEncoder, GNNEncoder_OpenGSL, APPNPEncoder, GINEncoder
from opengsl.module.fuse import Interpolate
from opengsl.module.transform import Normalize, KNN, Symmetry
from opengsl.module.metric import InnerProduct
from torch_sparse import SparseTensor


class GRCN(torch.nn.Module):

    def __init__(self, num_nodes, n_feat, n_class, device, conf):
        super(GRCN, self).__init__()
        self.num_nodes = num_nodes
        self.n_feat = n_feat
        if conf.model['type'] == 'gcn':
            self.conv_task = GNNEncoder_OpenGSL(n_feat=n_feat, n_class=n_class, **conf.model)
        elif conf.model['type'] == 'appnp':
            self.conv_task = APPNPEncoder(n_feat, conf.model['n_hidden'], n_class,
                                          dropout=conf.model['dropout'], K=conf.model['K_APPNP'],
                                          alpha=conf.model['alpha'], spmm_type=1)
        elif conf.model['type'] == 'gin':
            self.conv_task = GINEncoder(n_feat, conf.model['n_hidden'], n_class,
                                        conf.model['n_layers'], conf.model['mlp_layers'], spmm_type=1)
        self.model_type = conf.gsl['model_type']
        if conf.gsl['model_type'] == 'diag':
            self.conv_graph = GCNDiagEncoder(2, n_feat)
        else:
            self.conv_graph = GNNEncoder_OpenGSL(n_feat, conf.gsl['n_hidden_2'], conf.gsl['n_hidden_1'], **conf.gsl)

        self.K = conf.gsl['K']
        self._normalize = conf.gsl['normalize']   # 用来决定是否对node embedding进行normalize

        self.metric = InnerProduct()
        self.normalize_a = Normalize(add_loop=False)
        self.normalize_e = Normalize('row-norm', p=2)
        self.knn = KNN(self.K, sparse_out=True)
        self.sym = Symmetry(1)
        self.fuse = Interpolate(1, 1)
        
        self.conf = conf
        self.beta_vector  = torch.full(
            size = (num_nodes,1), 
            fill_value = conf.training['beta_init'],
            dtype = torch.float32,
            device = torch.device("cuda"))
        self.beta_vector  = torch.nn.Parameter(self.beta_vector)
        self.gamma = conf.training['gamma']
        self.gamma_star = self.gamma / (self.gamma - 1)
        
        self.optimizer_beta  = torch.optim.Adam([
            {'params': self.beta_vector, 'lr': conf.training['lr_beta'], 'weight_decay': 0}
        ])
        
    def beta_update(self, feats, adj):
        Q = self.Q
        # emb = self.gsl(feats, normalize(adj, add_loop=self.conf.dataset['add_loop'])) # GCN
        node_embeddings = self._node_embeddings(feats, adj)
        Adj_new = self.cal_similarity_graph(node_embeddings)
        Adj_new = self._sparse_graph(Adj_new)
        
        new_adj = Adj_new.to_dense()
        # adj = torch.relu(new_adj - self.beta_vector.expand(-1, new_adj.shape[1])) ** self.gamma_star
        adj = new_adj ** self.gamma_star
        loss_r = (Q * adj).sum(dim=-1).pow(1 / self.gamma_star)
        loss_beta = (loss_r + self.beta_vector.squeeze()).sum()
        
        self.optimizer_beta.zero_grad()
        loss_beta.backward()
        self.optimizer_beta.step()
        
        return loss_beta
    
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def graph_parameters(self):
        return list(self.conv_graph.parameters())

    def base_parameters(self):
        return list(self.conv_task.parameters())

    def cal_similarity_graph(self, node_embeddings):
        # 一个2head的相似度计算
        # 完全等价于普通cosine
        similarity_graph = self.metric(node_embeddings[:, :int(self.n_feat / 2)])
        similarity_graph += self.metric(node_embeddings[:, int(self.n_feat / 2):])
        return similarity_graph

    def _sparse_graph(self, raw_graph):
        new_adj = self.knn(adj=raw_graph)
        new_adj = self.sym(new_adj)
        return new_adj

    def _node_embeddings(self, input, Adj):
        norm_Adj = self.normalize_a(Adj)
        node_embeddings = self.conv_graph(input, SparseTensor.from_torch_sparse_coo_tensor(norm_Adj))
        if self._normalize:
            node_embeddings = self.normalize_e(node_embeddings)
        return node_embeddings

    def forward(self, input, Adj, update_beta=False):
        adjs = {}
        Adj.requires_grad = False
        
        if update_beta:
            self.beta_update(input, Adj)
        
        node_embeddings = self._node_embeddings(input, Adj)
        Adj_new = self.cal_similarity_graph(node_embeddings)
        Adj_new = self._sparse_graph(Adj_new)
        
        adj_diff = Adj_new
        if (1 / (self.gamma - 1)) == 1:
            s_positive = adj_diff
        else:
            s_positive = adj_diff.pow(1 / (self.gamma - 1))
        s = s_positive * self.Q
        s = normalize(s, style='row')
        Adj_new = s
        
        if self.conf.training['original_graph']:
            Adj_final = self.fuse(Adj_new, Adj)
            Adj_final_norm = self.normalize_a(Adj_final.coalesce())
        else:
            Adj_final = Adj_new
            Adj_final_norm = self.normalize_a(Adj_new.coalesce())
        x = self.conv_task(input, SparseTensor.from_torch_sparse_coo_tensor(Adj_final_norm))

        adjs['new'] = Adj_new
        adjs['final'] = Adj_final

        return x, adjs
