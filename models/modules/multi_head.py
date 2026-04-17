import torch
import torch.nn as nn
import torch.nn.functional as F
from . import attention
from . import utils

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_q: int = None,
        d_k: int = None,
        d_v: int = None,
        d_model: int = None,
        n_head: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0,
        use_adjacency: bool = False,
        use_distance: bool = False,
        use_A_dynamic: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = n_head
        self.hidden_dims = d_model
        self.attention = attention.MSRSA(num_heads=n_head, dropout=attn_drop, use_adjacency=use_adjacency, use_distance=use_distance, use_A_dynamic=use_A_dynamic)
        assert d_q is not None and d_k is not None and d_v is not None and d_model is not None, "Please specify the dimensions of Q, K, V and d_model"
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.W_q = nn.Linear(d_q, d_model, bias=qkv_bias)  
        self.W_k = nn.Linear(d_k, d_model, bias=qkv_bias) 
        self.W_v = nn.Linear(d_v, d_model, bias=qkv_bias)  
        self.W_o = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_a = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.use_A_dynamic = use_A_dynamic
        self.d_cutoff = 1.0
        self.w_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        adjacency_matrix: torch.Tensor = None,
        distance_matrix: torch.Tensor = None,
        e_d: torch.Tensor = None,
        e_a: torch.Tensor = None,
        D_cache: torch.Tensor = None,
    ):
        b, h, l, d_i = queries.shape[0], self.num_heads, queries.shape[1], self.hidden_dims // self.num_heads
        Q, K, V = self.W_q(queries), self.W_k(keys), self.W_v(values)
        Q, K, V = [M.view(b, l, h, d_i).permute(0, 2, 1, 3) for M in (Q, K, V)]
        if self.use_A_dynamic == True:
            X = queries
            batch_size, num_nodes, d_model = X.shape
            node_similarity = torch.matmul(X, X.transpose(-1, -2))
            transformed_e_a = self.W_a(e_a)
            e_i_expanded = transformed_e_a.unsqueeze(2)
            e_j_expanded = transformed_e_a.unsqueeze(1)
            angle_component = torch.sum(e_i_expanded + e_j_expanded, dim=-1)
            raw_attention_scores = node_similarity + angle_component
            neighbor_mask = adjacency_matrix
            masked_scores = raw_attention_scores.masked_fill(neighbor_mask == 0, -1e9)
            attention_weights = F.softmax(masked_scores, dim=-1)
            alpha = attention_weights * neighbor_mask.float()
            A_Geo = adjacency_matrix + self.w_alpha * alpha
        else:
            A_Geo = None
        attn_out = self.attention(Q, K, V, attention_mask, adjacency_matrix, distance_matrix, A_Geo)
        out, attn_weight = attn_out["out"], attn_out["attn_weight"]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, l, h * d_i)
        return {
            "out": self.W_o(out),
            "attn_weight": attn_weight,
        }

if __name__ == "__main__":
    b, l, d, h = 2, 4, 4, 2
    q, k, v = torch.randn(b, l, d), torch.randn(b, l, d), torch.randn(b, l, d)
    mask = utils.valid_length_to_mask(torch.tensor([2, 3]), max_len=l)
    attn = MultiHeadAttention(d_q=d, d_k=d, d_v=d, d_model=d, n_head=h, attn_type="SelfAttention", qkv_bias=True)
    out = attn(q, k, v, mask)
    print(out.shape)
