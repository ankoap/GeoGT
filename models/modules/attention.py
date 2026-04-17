import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

class SelfAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop_out = nn.Dropout(dropout)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attention_mask: torch.Tensor = None, attention_bias: torch.Tensor = None
    ) -> torch.Tensor:
        scale = Q.shape[-1] ** 0.5
        attention_score = (Q @ K.mT) / torch.tensor(scale)  # (b, l, l) | (b, h, l, l)
        if attention_bias is not None:
            attention_score += attention_bias
        if attention_mask is not None:
            attention_score = utils.mask_attention_score(attention_score, attention_mask)
        attention_weight = F.softmax(attention_score, dim=-1)  # (b, l, l) | (b, h, l, l)
        return self.drop_out(attention_weight) @ V  # (b, l, d) | (b, h, l, d)


class MSRSA(nn.Module):
    def __init__(self, num_heads: int = 1, dropout: float = 0.0, use_adjacency: bool = True, use_distance: bool = True, use_A_dynamic: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.use_adjacency = use_adjacency
        self.use_distance = use_distance

        self.use_A_Geo = use_A_dynamic

        self.w_Geo = nn.Parameter(torch.tensor(1.0))

        self.drop_out = nn.Dropout(dropout)
        self.weight_A, self.weight_D, self.use_A_Geo = None, None, None
        if self.use_adjacency:
            self.weight_A = torch.randn(num_heads).view(1, num_heads, 1, 1)
            self.weight_A = nn.Parameter(self.weight_A, requires_grad=True)
            self.weight_Lap = torch.randn(num_heads).view(1, num_heads, 1, 1)
            self.weight_Lap = nn.Parameter(self.weight_A, requires_grad=True)
        if self.use_distance:
            self.weight_D = torch.randn(num_heads).view(1, num_heads, 1, 1)
            self.weight_D = nn.Parameter(self.weight_D, requires_grad=True)
        if self.use_A_Geo:
            self.weight_A_Geo = torch.randn(num_heads).view(1, num_heads, 1, 1)
            self.weight_A_Geo = nn.Parameter(self.weight_A_Geo, requires_grad=True)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor = None,
        adjacency_matrix: torch.Tensor = None,
        row_subtracted_distance_matrix: torch.Tensor = None,
        A_Geo: torch.Tensor = None
    ) -> torch.Tensor:
        M, A, D_s = attention_mask, adjacency_matrix, row_subtracted_distance_matrix
        if self.use_adjacency and A is None:
            raise ValueError(f"Adjacency matrix is not provided when using adjacency matrix in {self.__class__.__name__}")
        if self.use_distance and D_s is None:
            raise ValueError(f"Subtracted distance matrix is not provided when using distance matrix in {self.__class__.__name__}")
        A = A.unsqueeze(1) if self.use_adjacency else None
        D_s = D_s.unsqueeze(1) if self.use_distance else None
        A_Geo = A_Geo.unsqueeze(1) if self.use_A_Geo else None
        scale = Q.shape[-1] ** 0.5
        attn_score = Q @ K.mT
        attn_score = utils.mask_attention_score(attn_score, M, 0.0) if M is not None else attn_score
        B_A = attn_score * (A * self.weight_A) if self.use_adjacency else None
        B_A_Geo = attn_score * (A_Geo * self.weight_A_Geo) if self.use_A_Geo else None
        attn_score = attn_score + 1.5*B_A_Geo if B_A_Geo is not None else attn_score
        attn_score = attn_score + B_A if B_A is not None else attn_score
        attn_score = attn_score / torch.tensor(scale)
        attention_weight = F.softmax(attn_score, dim=-1)
        return {
            "out": self.drop_out(attention_weight) @ V,
            "attn_weight": attention_weight.detach(),
        }
