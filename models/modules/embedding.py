import torch
import torch.nn as nn

from . import utils
from typing import Literal


class AtomEmbedding(nn.Module):
    def __init__(self, atom_embedding_dim: int, attr_reduction: Literal["mean", "sum", "cat_last_dim"] = "mean") -> None:
        super().__init__()
        self.attr_reduction = attr_reduction
        self.embedding_list = nn.ModuleList()
        vocab_dims_ls = utils.get_atom_vocab_dims()
        for vocab_dim in vocab_dims_ls:
            embedding_layer = nn.Embedding(vocab_dim + 1, atom_embedding_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(embedding_layer.weight)
            self.embedding_list.append(embedding_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_ls = torch.split(x, 1, dim=1)
        embedding_ls = [self.embedding_list[i](indices_ls[i]) for i in range(len(indices_ls))]
        if self.attr_reduction == "mean":
            return torch.mean(torch.cat(embedding_ls, dim=1), dim=1)
        elif self.attr_reduction == "sum":
            return torch.sum(torch.cat(embedding_ls, dim=1), dim=1)
        elif self.attr_reduction == "cat_last_dim":
            return torch.cat(embedding_ls, dim=-1)


class BondEmbedding(nn.Module):
    def __init__(self, bond_embedding_dim: int, attr_reduction: Literal["mean", "sum", "cat_last_dim"] = "mean") -> None:
        super().__init__()
        self.attr_reduction = attr_reduction
        self.embedding_list = nn.ModuleList()
        vocab_dims_ls = utils.get_bond_vocab_dims()
        for vocab_dim in vocab_dims_ls:
            embedding_layer = nn.Embedding(vocab_dim, bond_embedding_dim)
            torch.nn.init.xavier_uniform_(embedding_layer.weight)
            self.embedding_list.append(embedding_layer)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        indices_ls = torch.split(edge_attr, 1, dim=1)
        embedding_ls = [self.embedding_list[i](indices_ls[i]) for i in range(len(indices_ls))]
        if self.attr_reduction == "mean":
            return torch.mean(torch.cat(embedding_ls, dim=1), dim=1)
        elif self.attr_reduction == "sum":
            return torch.sum(torch.cat(embedding_ls, dim=1), dim=1)
        elif self.attr_reduction == "cat_last_dim":
            return torch.cat(embedding_ls, dim=-1)


class NodeEmbedding(nn.Module):
    def __init__(self, atom_embedding_dim: int, attr_reduction: Literal["mean", "sum", "cat_last_dim"] = "mean") -> None:
        super().__init__()
        self.atom_encoder = AtomEmbedding(atom_embedding_dim, attr_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.atom_encoder(x)
        b, l, d = x.shape
        res = self.atom_encoder(x.reshape(-1, d).to(torch.int32))
        res = res.reshape(b, l, -1)
        return res


class EdgeEmbedding(nn.Module):
    def __init__(self, bond_embedding_dim: int, attr_reduction: Literal["mean", "sum", "cat_last_dim"] = "mean") -> None:
        super().__init__()
        self.bond_encoder = BondEmbedding(bond_embedding_dim, attr_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        res = self.bond_encoder(x.reshape(-1, d).to(torch.int32))
        res = res.reshape(b, l, -1)
        return res
