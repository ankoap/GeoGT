from transformers import PretrainedConfig
from typing import Literal

class GeoGTConfig(PretrainedConfig):
    def __init__(
        self,
        n_encode_layers: int = 12,
        encoder_use_A_in_attn: bool = True,
        encoder_use_D_in_attn: bool = False,
        embed_style: Literal["atom_type_ids", "atom_tokenized_ids", "ogb"] = "atom_tokenized_ids",
        n_decode_layers: int = 6,
        decoder_use_A_in_attn: bool = True,
        decoder_use_D_in_attn: bool = True,
        atom_vocab_size: int = 513,
        d_embed: int = 768,
        pre_ln: bool = True,
        d_q: int = 768,
        d_k: int = 768,
        d_v: int = 768,
        d_model: int = 768,
        n_head: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        norm_drop: float = 0.1,
        ffn_drop: float = 0.1,
        d_ffn: int = 3072,
        **kwargs,
    ):
        self.n_encode_layers = n_encode_layers
        self.encoder_use_A_in_attn = encoder_use_A_in_attn
        self.encoder_use_D_in_attn = encoder_use_D_in_attn
        self.embed_style = embed_style
        self.n_decode_layers = n_decode_layers
        self.decoder_use_A_in_attn = decoder_use_A_in_attn
        self.decoder_use_D_in_attn = decoder_use_D_in_attn
        self.atom_vocab_size = atom_vocab_size
        self.d_embed = d_embed
        self.pre_ln = pre_ln
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_head = n_head
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.norm_drop = norm_drop
        self.ffn_drop = ffn_drop
        self.d_ffn = d_ffn
        super().__init__(**kwargs)
