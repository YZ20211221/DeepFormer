from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class Flow_Attention(nn.Module):
    # flow attention in normal version
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=5e-4):
        super(Flow_Attention, self).__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
      
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
    
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
     
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
    
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
   
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
       
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x
        



class DeepFormer(nn.Module):
    def __init__(
        self,
        *,
        sequence_length, 
        n_targets
    ):
        super().__init__()
      
        conv_kernel_size = 8
        pool_kernel_size = 4
        
        self.conv_net1 = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320))
        self.conv_net2 = nn.Sequential(
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2))
        self.conv_net3 = nn.Sequential(
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))
            
       
        self.attn_normal = Flow_Attention(44, 44, 44, 4)
     
        self.layers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = num_patches, attn_dim = attn_dim))) for i in range(depth)])

        self.classifier = nn.Sequential(
            nn.Linear(44 * 960, 19),
            nn.ReLU(inplace=True),
            nn.Linear(19, 19),
            nn.Sigmoid())
      

    def forward(self, x):
        x = self.conv_net1(x)  
        x = self.conv_net2(x)
        x = self.conv_net3(x)
        x = self.attn_normal(x, x, x)
        x = x.view(x.size(0), 44 * 960)
        predict = self.classifier(x)  
        
        return predict
       
        
def criterion():
    return nn.BCELoss()

def get_optimizer(lr):
    return (torch.optim.Adam,{"lr": lr, "weight_decay": 1e-6})
