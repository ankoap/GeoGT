import torch
import torch.nn as nn
from .configuration_geogt import GeoGTConfig
from ..modules import ConformerPredictionHead, GraphRegressionHead
from ..modules import ConformerPredictionOutput, GraphRegressionOutput
from ..modules import MultiHeadAttention, AddNorm, PositionWiseFFN, Residual
from ..modules.utils import make_cdist_mask, compute_distance_residual_bias
from ..modules import AtomEmbedding, NodeEmbedding  # for Ogb embedding ablation
from transformers import PretrainedConfig, PreTrainedModel

class GeoGTBlock(nn.Module):
    def __init__(self, config: PretrainedConfig = None, encoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.use_A_in_attn = getattr(config, "encoder_use_A_in_attn", False) if encoder else getattr(config, "decoder_use_A_in_attn", False)
        self.use_D_in_attn = getattr(config, "encoder_use_D_in_attn", False) if encoder else getattr(config, "decoder_use_D_in_attn", False)

        self.use_e_d = getattr(config, "encoder_use_e_d", False) if encoder else getattr(config, "decoder_use_e_d", False)
        self.use_e_a = getattr(config, "encoder_use_e_a", False) if encoder else getattr(config, "decoder_use_e_a", False)
        self.use_D_cache = getattr(config, "encoder_use_D_cache", False) if encoder else getattr(config, "decoder_use_D_cache", False)

        self.multi_attention = MultiHeadAttention(
            d_q=getattr(config, "d_q", 256),
            d_k=getattr(config, "d_k", 256),
            d_v=getattr(config, "d_v", 256),
            d_model=getattr(config, "d_model", 256),
            n_head=getattr(config, "n_head", 8),
            qkv_bias=getattr(config, "qkv_bias", True),
            attn_drop=getattr(config, "attn_drop", 0.1),
            use_adjacency=self.use_A_in_attn,
            use_distance=self.use_D_in_attn,
            use_A_dynamic=self.use_e_d,
        )
        self.add_norm01 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))
        self.position_wise_ffn = PositionWiseFFN(
            d_in=getattr(config, "d_model", 256),
            d_hidden=getattr(config, "d_ffn", 1024),
            d_out=getattr(config, "d_model", 256),
            dropout=getattr(config, "ffn_drop", 0.1),
        )
        self.add_norm02 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))

    def forward(self, **inputs):
        X, M = inputs.get("node_embedding"), inputs.get("node_mask")

        A = inputs.get("adjacency") if self.use_A_in_attn else None
        D = inputs.get("distance") if self.use_D_in_attn else None
        e_d = inputs.get("e_d") if self.use_e_d else None
        e_a = inputs.get("e_a") if self.use_e_a else None
        D_cache = inputs.get("D_cache") if self.use_D_cache else None

        attn_out = self.multi_attention(X, X, X, attention_mask=M, adjacency_matrix=A, distance_matrix=D, e_d=e_d, e_a=e_a, D_cache=D_cache)
        Y, attn_weight = attn_out["out"], attn_out["attn_weight"]
        X = self.add_norm01(X, Y)
        Y = self.position_wise_ffn(X)
        X = self.add_norm02(X, Y)
        return {
            "out": X,
            "attn_weight": attn_weight,
        }

class GeoGTPretrainedModel(PreTrainedModel):
    config_class = GeoGTConfig
    base_model_prefix = "Conformer"
    is_parallelizable = False
    main_input_name = "node_input_ids"

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

class GeoGTEncoder(GeoGTPretrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.embed_style = getattr(config, "embed_style", "atom_tokenized_ids")
        if self.embed_style == "atom_tokenized_ids":
            self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 513), getattr(config, "d_embed", 256), padding_idx=0)
        elif self.embed_style == "atom_type_ids":
            self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 119), getattr(config, "d_embed", 256), padding_idx=0)
        elif self.embed_style == "ogb":
            self.ogb_node_embedding = NodeEmbedding(atom_embedding_dim=getattr(config, "d_embed", 256), attr_reduction="sum")
        self.encoder_blocks = nn.ModuleList([GeoGTBlock(config, encoder=True) for _ in range(getattr(config, "n_encode_layers", 12))])
        self.__init_weights__()
        self.gamma = nn.Parameter(torch.ones(256))
        self.beta = nn.Parameter(torch.zeros(256))
        self.eps = 1e-5
    
    def layer_norm(self, x, N):
        device = x.device
        gamma = nn.Parameter(torch.ones(N)).to(device)
        beta = nn.Parameter(torch.zeros(N)).to(device)
        mu = torch.mean(x, dim=-1, keepdim=True)
        o = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mu) / (o + self.eps)
        normalized_x = gamma * normalized_x + beta
        return normalized_x

    def forward(self, **inputs):
        if self.embed_style == "atom_tokenized_ids":
            node_input_ids = inputs.get("node_input_ids")
            node_embedding = self.node_embedding(node_input_ids)
        elif self.embed_style == "atom_type_ids":
            node_input_ids = inputs.get("node_type")
            node_embedding = self.node_embedding(node_input_ids)
        elif self.embed_style == "ogb":
            node_embedding = self.ogb_node_embedding(inputs["node_attr"])
        u = inputs.get("lap_eigenvectors")
        A_adj = inputs.get("adjacency")
        modified_embedding = node_embedding.clone()
        modified_embedding[:, :, : u.shape[-1]] += u
        modified_embedding[:, :, u.shape[-1] : 2*u.shape[-1]] += u
        modified_embedding[:, :, : A_adj.shape[-1]] += 5*A_adj
        modified_embedding[:, :, u.shape[-1] : 2*u.shape[-1]] += 5*A_adj
        node_embedding = modified_embedding
        inputs["node_embedding"] = node_embedding

        if self.config.encoder_use_D_in_attn:
            C = inputs.get("conformer")
            D, D_M = torch.cdist(C, C), make_cdist_mask(inputs.get("node_mask"))
            D = compute_distance_residual_bias(cdist=D, cdist_mask=D_M)
            inputs["distance"] = D
        attn_weight_dict = {}
        for i, encoder_block in enumerate(self.encoder_blocks):
            block_out = encoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"encoder_block_{i}"] = attn_weight
        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}

class GeoGTDecoder(GeoGTPretrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.decoder_blocks = nn.ModuleList([GeoGTBlock(config, encoder=False) for _ in range(getattr(config, "n_decode_layers", 6))])
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.w_diag = nn.Parameter(torch.tensor(0.1))
        self.w_y = nn.Parameter(torch.tensor(0.1))
        self.__init_weights__()

    def forward(self, **inputs):
        node_embedding = inputs.get("node_embedding")
        u = inputs.get("lap_eigenvectors")
        A_adj = inputs.get("adjacency")
        lambad = inputs.get("eigenvalues")
        modified_embedding = node_embedding.clone()
        gamma = inputs.get("gamma")
        batch_size, N, _ = u.shape
        e = torch.tensor(2.71828, device=u.device)
        epsilon = 1e-5
        lambda_max = lambad[:, -1]
        gamma_exp = gamma.unsqueeze(1)
        lambda_max_exp = lambda_max.unsqueeze(1)
        term1 = gamma_exp * (1 / torch.log(e + lambad / epsilon))
        term2 = (1 - gamma_exp) * (lambad / (lambda_max_exp + epsilon))
        w_k = term1 + term2
        w_k_exp = w_k.unsqueeze(2)
        u_tilde = u * w_k_exp
        angle_term_sum = inputs["angle_term_sum"]
        A = inputs.get("adjacency")
        D = inputs.get("distance")
        A_D_product = A * D
        R_i = torch.max(A_D_product, dim=2)[0]
        epsilon = 1e-5
        k = (4 * torch.pi - angle_term_sum) / (4 * torch.pi * R_i**2 + epsilon)
        k_min, _ = torch.min(k, dim=1, keepdim=True)
        abs_diff = torch.abs(k - k_min)
        diag_vals = torch.exp(-self.beta * abs_diff)
        diag_mat = torch.diag_embed(diag_vals)
        diag_mat = self.w_diag * diag_mat + 5*A + D
        k_exp = k.unsqueeze(-1)
        b = torch.sum(k_exp * u, dim=1)
        stabilized_b = b + 1e-8
        U, S, Vh = torch.linalg.svd(stabilized_b, full_matrices=False)
        y = self.w_y * U.unsqueeze(-1) + diag_mat
        modified_embedding[:, :, : u.shape[-1]] += u
        modified_embedding[:, :, u.shape[-1] : 2*u.shape[-1]] += u
        modified_embedding[:, :, : u.shape[-1]] += 5 * A_adj
        modified_embedding[:, :, u.shape[-1] : 2*u.shape[-1]] += 5 * A_adj
        modified_embedding[:, :, : u.shape[-1]] += y
        modified_embedding[:, :, u.shape[-1] : 2*u.shape[-1]] += y
        node_embedding = modified_embedding

        inputs["node_embedding"] = node_embedding
        attn_weight_dict = {}
        for i, decoder_block in enumerate(self.decoder_blocks):
            block_out = decoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"decoder_block_{i}"] = attn_weight
        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}

class GeoGTForConformerPrediction(GeoGTPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.encoder = GeoGTEncoder(config)
        self.decoder = GeoGTDecoder(config)
        self.conformer_head = ConformerPredictionHead(hidden_X_dim=getattr(config, "d_model", 256))
        self.__init_weights__()

        self.mu = torch.tensor([1, 1.5, 2.0, 2.5, 3.0])
        self.sigma = 0.2

        self.angle_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, **inputs):
        conformer, node_mask = inputs.get("conformer"), inputs.get("node_mask")

        encoder_out = self.encoder(**inputs)
        node_embedding, encoder_attn_weight_dict = encoder_out["node_embedding"], encoder_out["attn_weight_dict"]

        cache_out = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask, compute_loss=True)
        loss_cache, conformer_cache = cache_out["loss"], cache_out["conformer_hat"]
        D_cache, D_M = torch.cdist(conformer_cache, conformer_cache), make_cdist_mask(node_mask)
        A_cache = inputs["adjacency"]
        B, N, _ = A_cache.shape
        device = A_cache.device
        batch_idx = torch.arange(B, device=device)[:, None, None, None]
        i_idx = torch.arange(N, device=device)[None, :, None, None]
        j_idx = torch.arange(N, device=device)[None, None, :, None]
        k_idx = torch.arange(N, device=device)[None, None, None, :]
        is_neighbor = A_cache[..., None] & A_cache[:, :, None, :]
        valid_pairs = (j_idx < k_idx) & (is_neighbor)
        r_i = conformer_cache[:, None, None, :, :]
        r_j = conformer_cache[:, :, None, None, :]
        r_k = conformer_cache[:, None, :, None, :]
        vec_ji = r_j - r_i
        vec_ki = r_k - r_i
        norm_ji = torch.norm(vec_ji, dim=-1, keepdim=True) + 1e-6
        norm_ki = torch.norm(vec_ki, dim=-1, keepdim=True) + 1e-6
        cosine = (vec_ji * vec_ki).sum(dim=-1) / (norm_ji * norm_ki).squeeze(-1)
        angles = torch.arccos(torch.clamp(cosine, -1.0, 1.0))

        masked_angles = angles * valid_pairs.float()
        angle_counts = valid_pairs.sum(dim=(-2, -1))
        inputs["angle_counts"] = angle_counts
        angle_sum = masked_angles.sum(dim=(-2, -1))
        inputs["angle_sum"] = angle_sum
        angle_terms = 2 * torch.pi * (1 - torch.cos(angles / 2))
        masked_angle_terms = angle_terms * valid_pairs.float()
        angle_term_sum = masked_angle_terms.sum(dim=(-2, -1))
        inputs["angle_term_sum"] = angle_term_sum
        angle_mean = angle_sum / (angle_counts + 1e-6)
        angle_sq_sum = (masked_angles ** 2).sum(dim=(-2, -1))
        angle_variance = (angle_sq_sum / (angle_counts + 1e-6)) - angle_mean ** 2
        angle_variance = torch.where(angle_counts >= 1, angle_variance, torch.zeros_like(angle_variance))
        angle_norm = (angle_variance - angle_variance.min()) / (angle_variance.max() - angle_variance.min() + 1e-6)
        e_a = self.angle_encoder(angle_norm.unsqueeze(-1))
        mean_angle_norm = angle_norm.mean(dim=1)
        alpha, beta = 1.0, 1.0
        gamma = torch.exp(-alpha * torch.pow(mean_angle_norm, beta))
        inputs["gamma"] = gamma

        inputs["e_a"] = e_a
        inputs["D_cache"] = D_cache

        D_cache = compute_distance_residual_bias(cdist=D_cache, cdist_mask=D_M)
        inputs["node_embedding"] = node_embedding
        inputs["distance"] = D_cache
        decoder_out = self.decoder(**inputs)
        node_embedding, decoder_attn_weight_dict = decoder_out["node_embedding"], decoder_out["attn_weight_dict"]
        outputs = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask, compute_loss=True)

        return ConformerPredictionOutput(
            loss=(outputs["loss"] + loss_cache) / 2,
            cdist_mae=outputs["cdist_mae"],
            cdist_mse=outputs["cdist_mse"],
            coord_rmsd=outputs["coord_rmsd"],
            conformer=outputs["conformer"],
            conformer_hat=outputs["conformer_hat"],
        )


class GeoGTForGraphRegression(GeoGTPretrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.encoder = GeoGTEncoder(config)
        self.decoder = GraphRegressionHead(hidden_X_dim=getattr(config, "d_model", 256))
        self.__init_weights__()

    def forward(self, **inputs):
        encoder_out = self.encoder(**inputs)
        graph_rep = encoder_out["node_embedding"].mean(dim=1)
        decoder_outputs = self.decoder(hidden_X=graph_rep, labels=inputs.get("labels"))
        return GraphRegressionOutput(
            loss=decoder_outputs["loss"],
            mae=decoder_outputs["mae"],
            mse=decoder_outputs["mse"],
            logits=decoder_outputs["logits"],
            labels=decoder_outputs["labels"],
        )
