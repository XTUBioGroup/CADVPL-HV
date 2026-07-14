import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv
from typing import Optional

from module import GATDotConv


# ============================================================
# 0. Utils
# ============================================================

def to_dense_batch(feat: torch.Tensor, batch_num_nodes):
    """
    feat: [N_total, D]
    batch_num_nodes: list[int] or Tensor[B]
    return:
        pad_feat: [B, Lmax, D]
        mask: [B, Lmax], True 表示 padding
    """
    device = feat.device
    if isinstance(batch_num_nodes, list):
        batch_num_nodes = torch.tensor(batch_num_nodes, device=device)

    feat_list = torch.split(feat, batch_num_nodes.tolist())
    pad_feat = torch.nn.utils.rnn.pad_sequence(
        feat_list, batch_first=True, padding_value=0.0
    )

    B, Lmax = pad_feat.shape[0], pad_feat.shape[1]
    mask = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    for i, n in enumerate(batch_num_nodes.tolist()):
        n = int(n)
        if n < Lmax:
            mask[i, n:] = True

    return pad_feat, mask


def masked_mean(x: torch.Tensor, pad_mask: torch.Tensor, dim: int = 1, eps: float = 1e-9):
    """
    x: [B, L, D]
    pad_mask: [B, L], True 表示 padding
    """
    valid = (~pad_mask).float().unsqueeze(-1)
    num = (x * valid).sum(dim=dim)
    den = valid.sum(dim=dim).clamp(min=eps)
    return num / den


def masked_max(x: torch.Tensor, pad_mask: torch.Tensor, dim: int = 1):
    """
    x: [B, L, D]
    pad_mask: [B, L], True 表示 padding
    """
    x = x.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
    out = x.max(dim=dim).values
    out[out == float("-inf")] = 0.0
    return out


# ============================================================
# 1. Prototype Aggregator
# ============================================================

class PrototypeAggregator(nn.Module):
    """
    将变长序列压缩成固定数量 K 的 prototype
    输入:
      x: [B, L, D]
      pad_mask: [B, L]
    输出:
      p: [B, K, D]
      p_mask: [B, K]，固定原型位默认都有效，因此全 False
    """

    def __init__(self, hidden_dim: int, num_prototypes: int = 128, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = int(num_prototypes)
        self.temperature = float(temperature)

        self.proto = nn.Parameter(torch.empty(self.K, hidden_dim))
        nn.init.xavier_uniform_(self.proto)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        x = self.norm(x)
        x = self.dropout(x)

        proto = self.proto.unsqueeze(0)  # [1, K, D]
        logits = torch.einsum("bkd,bld->bkl", proto, x) / self.temperature  # [B, K, L]
        logits = logits.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        alpha = torch.softmax(logits, dim=-1)      # [B, K, L]
        p = torch.einsum("bkl,bld->bkd", alpha, x) # [B, K, D]

        p_mask = torch.zeros(x.size(0), self.K, dtype=torch.bool, device=x.device)
        return p, p_mask


# ============================================================
# 2. Residue-level Structure Encoder (ProtT5 -> Graph Backbone)
# ============================================================

class ResiSC_Encoder(nn.Module):
    def __init__(self, param, node_type: str = "amino_acid"):
        super().__init__()
        self.node_type = node_type
        self.hidden_dim = param["resid_hidden_dim"]
        self.num_layers = param["resid_n_layer"]
        self.num_heads = param.get("num_heads", 4)
        self.dropout = nn.Dropout(param["dropout_ratio"])

        self.fc_dim = nn.Linear(param["in_size"], self.hidden_dim)

        # pLDDT gate
        self.gate_scale = nn.Parameter(torch.tensor(0.1))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

        self.gnnlayers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.gnnlayers.append(
                HeteroGraphConv(
                    {
                        "SEQ": GATDotConv(
                            self.hidden_dim,
                            self.hidden_dim,
                            self.num_heads,
                            param["dropout_ratio"],
                            allow_zero_in_degree=True,
                        ),
                        "STR_KNN": GATDotConv(
                            self.hidden_dim,
                            self.hidden_dim,
                            self.num_heads,
                            param["dropout_ratio"],
                            allow_zero_in_degree=True,
                        ),
                        "STR_DIS": GATDotConv(
                            self.hidden_dim,
                            self.hidden_dim,
                            self.num_heads,
                            param["dropout_ratio"],
                            allow_zero_in_degree=True,
                        ),
                    },
                    aggregate="sum",
                )
            )
            self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

    @staticmethod
    def _default_plddt(x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)

    def _get_plddt(self, batch_graph, x: torch.Tensor, plddt: Optional[torch.Tensor]) -> torch.Tensor:
        if plddt is not None:
            return plddt
        try:
            nd = batch_graph.nodes[self.node_type].data
            if "plddt" in nd:
                return nd["plddt"]
        except Exception:
            pass
        return self._default_plddt(x)

    def compute_gate(self, plddt: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_scale * plddt + self.gate_bias)

    def forward(self, batch_graph, x, plddt=None):
        plddt = self._get_plddt(batch_graph, x, plddt)
        gate = self.compute_gate(plddt)

        x = x * gate
        x = self.fc_dim(x)

        for l, layer in enumerate(self.gnnlayers):
            out = layer(batch_graph, {self.node_type: x})[self.node_type]
            x = torch.mean(out, dim=1)
            x = self.norms[l](F.relu(self.fcs[l](x)))
            if l != self.num_layers - 1:
                x = self.dropout(x)
        return x  # [N_total, hidden_dim]


# ============================================================
# 3. ESM-2 Residue-level Encoder
# ============================================================

class ESM2ResidueEncoder(nn.Module):
    """
    ESM-2 residue-level encoder
    输出 refined residue features: [B, L, D]
    """

    def __init__(self, esm_dim: int, out_dim: int, dropout: float = 0.1, kernel_size: int = 5):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = nn.LayerNorm(out_dim)

        self.local_conv = nn.Conv1d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
            bias=True,
        )

        self.local_ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.local_norm1 = nn.LayerNorm(out_dim)
        self.local_norm2 = nn.LayerNorm(out_dim)

    def forward(self, esm_dense: torch.Tensor, esm_mask: torch.Tensor):
        """
        esm_dense: [B, L, De]
        esm_mask:  [B, L]
        return:
            x: [B, L, D]
        """
        x = self.input_proj(esm_dense)   # [B, L, D]
        x = self.input_norm(x)

        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.local_norm1(x + x_conv)

        x_ffn = self.local_ffn(x)
        x = self.local_norm2(x + x_ffn)

        x = x.masked_fill(esm_mask.unsqueeze(-1), 0.0)
        return x


# ============================================================
# 4. Prototype-level Fusion
# ============================================================

class PrototypeFusion(nn.Module):
    """
    prototype-level residual fusion
    输入:
      p_prot: [B, K, D]
      p_esm:  [B, K, D]
    输出:
      p_fuse: [B, K, D]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, alpha_init: float = 0.1):
        super().__init__()
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.delta_norm = nn.LayerNorm(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, p_prot: torch.Tensor, p_esm: torch.Tensor):
        fusion_feat = torch.cat(
            [p_prot, p_esm, p_prot * p_esm, torch.abs(p_prot - p_esm)],
            dim=-1
        )  # [B, K, 4D]

        delta = self.delta_mlp(fusion_feat)   # [B, K, D]
        delta = self.delta_norm(delta)

        p_fuse = p_prot + self.alpha * delta
        p_fuse = self.out_norm(p_fuse)
        return p_fuse


# ============================================================
# 5. Fused Prototype Readout
# ============================================================

class FusedPrototypeReadout(nn.Module):
    """
    对 fused prototypes 做 mean + max pooling
    输入:
      proto_feat: [B, K, D]
      proto_mask: [B, K]
    输出:
      z: [B, out_dim]
    """

    def __init__(self, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, proto_feat: torch.Tensor, proto_mask: torch.Tensor):
        p_mean = masked_mean(proto_feat, proto_mask, dim=1)
        p_max = masked_max(proto_feat, proto_mask, dim=1)
        z = torch.cat([p_mean, p_max], dim=-1)
        z = self.proj(z)
        z = self.norm(z)
        return z


# ============================================================
# 6. Pair Classifier
# ============================================================

class PairClassifier(nn.Module):
    """
    使用 protein-level pair representation 做二分类
    pair feature = [h, v, h*v, |h-v|]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        pair_dim = hidden_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_h: torch.Tensor, z_v: torch.Tensor):
        pair_feat = torch.cat(
            [z_h, z_v, z_h * z_v, torch.abs(z_h - z_v)],
            dim=-1
        )
        logits = self.mlp(pair_feat)
        return logits


# ============================================================
# 7. Scheme A PPI Classifier (Prototype-level Fusion Version)
#    新增：对比学习辅助任务支持
# ============================================================

class PPI_Classifier_SchemeA(nn.Module):
    """
    Prototype-level fusion 版本 + 对比学习辅助任务
    增加 return_contrastive 选项，返回单模态蛋白表示
    """

    def __init__(
        self,
        param,
        human_state_dict=None,
        virus_state_dict=None,
        node_type: str = "amino_acid",
    ):
        super().__init__()
        self.node_type = node_type

        # -------------------------
        # 1) ProtT5 graph backbone
        # -------------------------
        self.human_encoder = ResiSC_Encoder(param, node_type=node_type)
        self.virus_encoder = ResiSC_Encoder(param, node_type=node_type)

        if human_state_dict is not None:
            print("[PPI_Classifier_SchemeA] Loading Human Encoder weights...")
            self._load_encoder_weights(self.human_encoder, human_state_dict)
        else:
            print("[PPI_Classifier_SchemeA] Human Encoder 随机初始化 (无预训练).")

        if virus_state_dict is not None:
            print("[PPI_Classifier_SchemeA] Loading Virus Encoder weights...")
            self._load_encoder_weights(self.virus_encoder, virus_state_dict)
        else:
            print("[PPI_Classifier_SchemeA] Virus Encoder 随机初始化 (无预训练).")

        self.hidden_dim = int(param["resid_hidden_dim"])
        self.dropout_ratio = float(param.get("dropout_ratio", 0.1))
        self.num_prototypes = int(param.get("num_prototypes", 128))
        self.proto_temp = float(param.get("proto_temperature", 1.0))
        self.protein_dim = int(param.get("protein_dim", self.hidden_dim))
        self.esm_dim = int(param["esm2_dim"])
        self.alpha_init = float(param.get("alpha_init", 0.1))

        # -------------------------
        # 2) Prot prototype aggregation
        # -------------------------
        self.agg_h = PrototypeAggregator(
            self.hidden_dim,
            self.num_prototypes,
            self.dropout_ratio,
            self.proto_temp
        )
        self.agg_v = PrototypeAggregator(
            self.hidden_dim,
            self.num_prototypes,
            self.dropout_ratio,
            self.proto_temp
        )

        # -------------------------
        # 3) ESM residue encoder + prototype aggregation
        # -------------------------
        self.esm_encoder_h = ESM2ResidueEncoder(
            esm_dim=self.esm_dim,
            out_dim=self.hidden_dim,
            dropout=self.dropout_ratio
        )
        self.esm_encoder_v = ESM2ResidueEncoder(
            esm_dim=self.esm_dim,
            out_dim=self.hidden_dim,
            dropout=self.dropout_ratio
        )

        self.esm_agg_h = PrototypeAggregator(
            self.hidden_dim,
            self.num_prototypes,
            self.dropout_ratio,
            self.proto_temp
        )
        self.esm_agg_v = PrototypeAggregator(
            self.hidden_dim,
            self.num_prototypes,
            self.dropout_ratio,
            self.proto_temp
        )

        # -------------------------
        # 4) Prototype-level fusion
        # -------------------------
        self.proto_fusion_h = PrototypeFusion(
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_ratio,
            alpha_init=self.alpha_init
        )
        self.proto_fusion_v = PrototypeFusion(
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_ratio,
            alpha_init=self.alpha_init
        )

        # -------------------------
        # 5) Readout after fused prototypes
        # -------------------------
        self.readout_h = FusedPrototypeReadout(
            hidden_dim=self.hidden_dim,
            out_dim=self.protein_dim,
            dropout=self.dropout_ratio
        )
        self.readout_v = FusedPrototypeReadout(
            hidden_dim=self.hidden_dim,
            out_dim=self.protein_dim,
            dropout=self.dropout_ratio
        )

        # -------------------------
        # 6) Pair Classifier
        # -------------------------
        self.classifier = PairClassifier(
            hidden_dim=self.protein_dim,
            dropout=self.dropout_ratio
        )

    def _load_encoder_weights(self, encoder_instance, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("Encoder."):
                new_state_dict[k[8:]] = v
            elif k.startswith("encoder."):
                new_state_dict[k[8:]] = v
            elif "Decoder" not in k and "mask_token" not in k:
                new_state_dict[k] = v
        encoder_instance.load_state_dict(new_state_dict, strict=False)

    def _encode_prot_branch(self, g, encoder, aggregator):
        """
        ProtT5 主干:
        x(ProtT5) -> graph encoder -> dense -> prototype
        return:
          proto: [B, K, D]
          proto_mask: [B, K]
        """
        x = g.nodes[self.node_type].data["x"]
        plddt = g.nodes[self.node_type].data.get("plddt", None)

        z_flat = encoder(g, x, plddt=plddt)  # [N_total, hidden_dim]
        z_dense, z_mask = to_dense_batch(z_flat, g.batch_num_nodes(self.node_type))

        proto, proto_mask = aggregator(z_dense, z_mask)
        return proto, proto_mask

    def _encode_esm_proto_branch(self, g, esm_encoder, esm_aggregator):
        """
        ESM-2 分支:
        residue-level esm2 -> dense -> residue encoder -> prototype
        return:
          proto: [B, K, D]
          proto_mask: [B, K]
        """
        nd = g.nodes[self.node_type].data
        if "esm2" not in nd:
            raise KeyError(
                "Graph is missing 'esm2' node feature. "
                "Please regenerate processed features and DGL cache."
            )

        esm = nd["esm2"]  # [N_total, esm_dim]
        if esm.dim() != 2:
            raise ValueError(
                f"Expected esm2 tensor shape [N, D], but got {tuple(esm.shape)}"
            )

        esm_dense, esm_mask = to_dense_batch(esm, g.batch_num_nodes(self.node_type))  # [B, L, De]
        esm_feat = esm_encoder(esm_dense, esm_mask)                                    # [B, L, D]
        proto, proto_mask = esm_aggregator(esm_feat, esm_mask)                        # [B, K, D]
        return proto, proto_mask

    def forward(self, g_h, g_v, return_aux=False, return_contrastive=False):
        # -------------------------
        # Human branch
        # -------------------------
        h_proto_prot, h_pmask = self._encode_prot_branch(
            g_h, self.human_encoder, self.agg_h
        )
        h_proto_esm, _ = self._encode_esm_proto_branch(
            g_h, self.esm_encoder_h, self.esm_agg_h
        )

        # 单模态蛋白级表示（用于对比学习）
        z_h_prot = self.readout_h(h_proto_prot, h_pmask)   # [B, protein_dim]
        z_h_esm = self.readout_h(h_proto_esm, h_pmask)

        # 融合后的表示
        h_proto_fuse = self.proto_fusion_h(h_proto_prot, h_proto_esm)
        z_h_fused = self.readout_h(h_proto_fuse, h_pmask)

        # -------------------------
        # Virus branch
        # -------------------------
        v_proto_prot, v_pmask = self._encode_prot_branch(
            g_v, self.virus_encoder, self.agg_v
        )
        v_proto_esm, _ = self._encode_esm_proto_branch(
            g_v, self.esm_encoder_v, self.esm_agg_v
        )

        z_v_prot = self.readout_v(v_proto_prot, v_pmask)
        z_v_esm = self.readout_v(v_proto_esm, v_pmask)

        v_proto_fuse = self.proto_fusion_v(v_proto_prot, v_proto_esm)
        z_v_fused = self.readout_v(v_proto_fuse, v_pmask)

        # -------------------------
        # Pair classification
        # -------------------------
        logits = self.classifier(z_h_fused, z_v_fused)

        if return_contrastive:
            contrast_dict = {
                "z_h_prot": z_h_prot,
                "z_h_esm": z_h_esm,
                "z_v_prot": z_v_prot,
                "z_v_esm": z_v_esm,
            }
            if return_aux:
                aux = {
                    "h_proto_prot": h_proto_prot,
                    "h_proto_esm": h_proto_esm,
                    "h_proto_fuse": h_proto_fuse,
                    "v_proto_prot": v_proto_prot,
                    "v_proto_esm": v_proto_esm,
                    "v_proto_fuse": v_proto_fuse,
                    "z_h_fused": z_h_fused,
                    "z_v_fused": z_v_fused,
                    "alpha_h": self.proto_fusion_h.alpha.detach(),
                    "alpha_v": self.proto_fusion_v.alpha.detach(),
                }
                return logits, aux, contrast_dict
            else:
                return logits, contrast_dict

        if return_aux:
            aux = {
                "h_proto_prot": h_proto_prot,
                "h_proto_esm": h_proto_esm,
                "h_proto_fuse": h_proto_fuse,
                "v_proto_prot": v_proto_prot,
                "v_proto_esm": v_proto_esm,
                "v_proto_fuse": v_proto_fuse,
                "z_h_fused": z_h_fused,
                "z_v_fused": z_v_fused,
                "alpha_h": self.proto_fusion_h.alpha.detach(),
                "alpha_v": self.proto_fusion_v.alpha.detach(),
            }
            return logits, aux

        return logits