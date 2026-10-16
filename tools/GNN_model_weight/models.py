import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.nn import global_mean_pool, EdgeConv, GATConv, GINConv, PNAConv
from torch.autograd import Function
from torch.distributions import Categorical

try:
    import pennylane as qml
    _PENNYLANE_AVAILABLE = True
except ImportError:
    _PENNYLANE_AVAILABLE = False

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 64),
                                  nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                  nn.ReLU(), nn.Linear(128, 128),nn.ReLU(), nn.Linear(128, 128)),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(256,256,),
                                  nn.ReLU(), nn.Linear(256, 256),nn.ReLU(), nn.Linear(256, 256)),aggr='add')
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.1)
        x = self.lin2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin3(x)
        #print(x.shape)
        return F.sigmoid(x)

# class LundNet(torch.nn.Module):
#     def __init__(self):
#         super(LundNet, self).__init__()
#         self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.ReLU(),
#                                             nn.Linear(32, 32), nn.ReLU()),aggr='add')
#         self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
#                                             nn.Linear(32, 32), nn.ReLU()),aggr='add')
#         self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),  nn.ReLU(),
#                                             nn.Linear(64, 64),  nn.ReLU()),aggr='add')
#         self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),  nn.ReLU(),
#                                             nn.Linear(64, 64),  nn.ReLU()),aggr='add')
#         self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),  nn.ReLU(),
#                                             nn.Linear(128, 128),  nn.ReLU()),aggr='add')
#         self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),  nn.ReLU(),
#                                             nn.Linear(128, 128),  nn.ReLU()),aggr='add')

#         self.seq1 = nn.Sequential(nn.Linear(448, 384),
#                                 nn.BatchNorm1d(num_features=384),
#                                 nn.ReLU())
#         # self.seq2 = nn.Sequential(nn.Linear(384, 256),
#         #                           nn.ReLU())
#         self.seq2 = nn.Sequential(nn.Linear(385, 256),
#                                   nn.ReLU())
#         self.lin = nn.Linear(256, 1)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         Ntrk = data.Ntrk
#         Ntrk = torch.unsqueeze(Ntrk, 1)
#         x1 = self.conv1(x, edge_index)
#         x2 = self.conv2(x1, edge_index)
#         x3 = self.conv3(x2, edge_index)
#         x4 = self.conv4(x3, edge_index)
#         x5 = self.conv5(x4, edge_index)
#         x6 = self.conv6(x5, edge_index)
#         x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
#         x = self.seq1(x)
#         x = global_mean_pool(x, batch)
#         x = torch.cat( (x, Ntrk) ,dim=1)
#         x = self.seq2(x)
#         x = F.dropout(x, p=0.1)
#         x = self.lin(x)
#         #print(x.shape)
#         return F.sigmoid(x)

class LundNet_General(torch.nn.Module):
    def __init__(self, extra_dim=1):
        """
        extra_dim: 拼接在图特征后的附加特征维度。
        - 默认 1: 仅 Ntrk (基础 LundNet)
        - 如果是 Lund + GN3X: 1 (Ntrk) + 9 (GN3X) = 10
        - 如果是 Lund + B-tag: 1 (Ntrk) + 1 (phbb) = 2
        """
        super(LundNet_General, self).__init__()
        # EdgeConv 层: 结构与 LundNet 保持完全一致
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384), nn.BatchNorm1d(384), nn.ReLU())
        
        # 核心修改点：输入维度 = 384 (GNN) + extra_dim
        self.seq2 = nn.Sequential(nn.Linear(384 + extra_dim, 256), nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data, extra_x):
        """
        extra_x: 已经拼接好的外部特征 Tensor (batch, extra_dim)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. 提取图特征
        x1 = self.conv1(x, edge_index); x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index); x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index); x6 = self.conv6(x5, edge_index)
        
        x_gnn = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x_gnn = self.seq1(x_gnn)
        x_gnn = global_mean_pool(x_gnn, batch) # (batch, 384)
        
        # 2. 拼接外部特征 (extra_x 必须包含 Ntrk)
        x_combined = torch.cat([x_gnn, extra_x], dim=1)
            
        x_combined = self.seq2(x_combined)
        x_combined = F.dropout(x_combined, p=0.1, training=self.training)
        return torch.sigmoid(self.lin(x_combined))


class FiLM_Layer(nn.Module):
    """Feature-wise Linear Modulation: scales and shifts node features using a conditioning signal."""
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta  = nn.Linear(cond_dim, feature_dim)

    def forward(self, h, cond):
        """
        h:    (N_nodes, feature_dim) — node features after EdgeConv
        cond: (N_nodes, cond_dim)    — per-node conditioning (broadcast from jet-level)
        """
        return self.gamma(cond) * h + self.beta(cond)


class LundNet_FiLM(nn.Module):
    """LundNet_General with a FiLM layer inserted after each EdgeConv block.

    The conditioning signal is GN3X_phbb (continuous b-tagging score), which is
    broadcast from jet-level to all nodes of that jet before conditioning.

    Interface is identical to LundNet_General: forward(data, extra_x)
    where extra_x[:, 0] = Ntrk and extra_x[:, 1] = GN3X_phbb.
    """
    def __init__(self, extra_dim=2, cond_dim=1):
        super().__init__()
        # EdgeConv channel sizes mirror LundNet_General exactly
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')

        # FiLM layers — one per EdgeConv, operating on its output channel dim
        self.film1 = FiLM_Layer(32,  cond_dim)
        self.film2 = FiLM_Layer(32,  cond_dim)
        self.film3 = FiLM_Layer(64,  cond_dim)
        self.film4 = FiLM_Layer(64,  cond_dim)
        self.film5 = FiLM_Layer(128, cond_dim)
        self.film6 = FiLM_Layer(128, cond_dim)

        self.seq1 = nn.Sequential(nn.Linear(448, 384), nn.BatchNorm1d(384), nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384 + extra_dim, 256), nn.ReLU())
        self.lin  = nn.Linear(256, 1)

    def forward(self, data, extra_x):
        """
        extra_x: (N_jets, extra_dim) — [Ntrk, GN3X_phbb]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Broadcast jet-level phbb to each node: extra_x[:, 1] is per-jet,
        # batch[i] maps node i to its jet index.
        cond = extra_x[:, 1:2][batch]   # (N_nodes, cond_dim)

        x1 = self.film1(self.conv1(x,  edge_index), cond)
        x2 = self.film2(self.conv2(x1, edge_index), cond)
        x3 = self.film3(self.conv3(x2, edge_index), cond)
        x4 = self.film4(self.conv4(x3, edge_index), cond)
        x5 = self.film5(self.conv5(x4, edge_index), cond)
        x6 = self.film6(self.conv6(x5, edge_index), cond)

        x_gnn = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x_gnn = self.seq1(x_gnn)
        x_gnn = global_mean_pool(x_gnn, batch)  # (N_jets, 384)

        x_combined = torch.cat([x_gnn, extra_x], dim=1)
        x_combined = self.seq2(x_combined)
        x_combined = F.dropout(x_combined, p=0.1, training=self.training)
        return torch.sigmoid(self.lin(x_combined))


class _LundFormerBlock(nn.Module):
    """Pre-LN transformer block with an additive tree-adjacency attention bias.

    The bias strength is a learned per-layer scalar, so the network can decide
    how strongly to favour parent-child declustering relations over the fully
    connected attention pattern.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff   = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)
        self.tree_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h, adj, pad_bias):
        """
        h:        (B, L, d_model)
        adj:      (B, 1, L, L) — 1.0 on parent-child pairs (both directions)
        pad_bias: (B, 1, 1, L) — 0 for valid keys, -inf for padding
        """
        B, L, _ = h.shape
        x = self.ln1(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        attn_mask = pad_bias + self.tree_scale * adj      # (B, 1, L, L) broadcast over heads
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, -1)
        h = h + self.drop(self.proj(out))
        h = h + self.drop(self.ff(self.ln2(h)))
        return h


class LundFormer(nn.Module):
    """Transformer over the full Lund declustering tree.

    Tokens are the same nodes LundNet consumes (identical 3 input features per
    node), so any performance difference w.r.t. LundNet_General is attributable
    to the architecture alone. Tree structure enters through (a) a node-depth
    embedding and (b) a per-layer learned parent-child attention bias.
    Interface matches LundNet_General: forward(data, extra_x) -> sigmoid score.
    """
    def __init__(self, extra_dim=1, node_dim=3, d_model=96, n_heads=8,
                 n_layers=4, d_ff=288, max_depth=64, dropout=0.1):
        # Defaults give ~405k parameters, aligned with LundNet_General (~382k)
        # so architecture comparisons are not confounded by capacity.
        super().__init__()
        self.max_depth = max_depth
        self.embed     = nn.Sequential(nn.Linear(node_dim, d_model), nn.LayerNorm(d_model))
        self.depth_emb = nn.Embedding(max_depth, d_model)
        self.blocks    = nn.ModuleList([
            _LundFormerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        # Head mirrors LundNet_General: Linear(pool+extra -> 256) -> 1
        self.seq2 = nn.Sequential(nn.Linear(d_model + extra_dim, 256), nn.ReLU())
        self.lin  = nn.Linear(256, 1)

    @staticmethod
    def _node_depth(edge_index, num_nodes, max_iters=32):
        """Depth of each node in its declustering tree (root = 0), computed by
        iterative longest-path propagation along edges. Works on the flat
        batched node indexing used by PyG."""
        depth = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        if edge_index.numel() == 0:
            return depth
        src, dst = edge_index[0], edge_index[1]
        for _ in range(max_iters):
            prev = depth
            cand  = depth[src] + 1
            depth = depth.scatter_reduce(0, dst, cand, reduce="amax", include_self=True)
            if torch.equal(depth, prev):
                break
        return depth

    def forward(self, data, extra_x):
        from torch_geometric.utils import to_dense_batch
        x, edge_index, batch = data.x, data.edge_index, data.batch
        N = x.size(0)

        depth = self._node_depth(edge_index, N).clamp(max=self.max_depth - 1)
        tok   = self.embed(x) + self.depth_emb(depth)

        h, mask = to_dense_batch(tok, batch)             # (B, L, d), (B, L) True=valid
        B, L, _ = h.shape

        # Flat node index -> (graph, position) for densifying the adjacency
        ptr = data.ptr if hasattr(data, "ptr") else None
        if ptr is None:
            counts = torch.bincount(batch, minlength=B)
            ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        pos = torch.arange(N, device=x.device) - ptr[batch]

        adj = h.new_zeros(B, L, L)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            b_e = batch[src]
            adj[b_e, pos[src], pos[dst]] = 1.0
            adj[b_e, pos[dst], pos[src]] = 1.0
        adj = adj.unsqueeze(1)                            # (B, 1, L, L)

        pad_bias = h.new_zeros(B, 1, 1, L)
        pad_bias.masked_fill_(~mask.view(B, 1, 1, L), float("-inf"))

        for blk in self.blocks:
            h = blk(h, adj, pad_bias)
        h = self.ln_out(h)

        # Masked mean pool (same pooling semantics as LundNet's global_mean_pool)
        maskf = mask.unsqueeze(-1).float()
        pooled = (h * maskf).sum(dim=1) / maskf.sum(dim=1).clamp(min=1.0)

        out = torch.cat([pooled, extra_x], dim=1)
        out = self.seq2(out)
        out = F.dropout(out, p=0.1, training=self.training)
        return torch.sigmoid(self.lin(out))


def _build_qnn_layer(n_qubits, n_layers):
    """Build a PennyLane TorchLayer VQC (angle embedding + ring-CNOT entangling layers)."""
    if not _PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane is required for LundNet_Quantum. "
                          "Install with: pip install pennylane")
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class LundNet_Quantum(nn.Module):
    """LundNet_General with the classical MLP head replaced by a Variational Quantum Circuit.

    Architecture (Route C – hybrid):
      • Same 6 EdgeConv blocks as LundNet_General (trained jointly with the QNN head)
      • After global_mean_pool: Linear(384+extra_dim → n_qubits) + tanh  [classical pre-circuit layer]
      • VQC: AngleEmbedding + n_layers × (RY+RZ per qubit + ring CNOT)  [quantum classification]
      • Output: sigmoid(<Z₀>)

    NOTE: PennyLane's default.qubit runs on CPU.  If the graph data lives on CUDA
    the pooled embedding is moved to CPU before the VQC and the result is moved back.
    Gradients flow end-to-end via backprop through the state-vector simulator.

    Args:
        extra_dim (int): number of jet-level scalar features appended after pooling (default 1 = Ntrk).
        n_qubits  (int): number of qubits / width of the quantum head (default 6).
        n_layers  (int): number of VQC entangling layers (default 2).
    """
    def __init__(self, extra_dim=1, n_qubits=6, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits

        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')

        self.seq1  = nn.Sequential(nn.Linear(448, 384), nn.BatchNorm1d(384), nn.ReLU())
        # Classical pre-circuit: map jet embedding to qubit angles in [-1, 1]
        self.pre_q = nn.Linear(384 + extra_dim, n_qubits)
        self.qlayer = _build_qnn_layer(n_qubits, n_layers)

    def forward(self, data, extra_x):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv1(x,  edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)

        x_gnn = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x_gnn = self.seq1(x_gnn)
        x_gnn = global_mean_pool(x_gnn, batch)         # (N_jets, 384)

        x_combined = torch.cat([x_gnn, extra_x], dim=1)  # (N_jets, 384+extra_dim)
        q_angles = torch.tanh(self.pre_q(x_combined))    # (N_jets, n_qubits), values in (-1, 1)

        # VQC runs on CPU; handle CUDA input transparently
        device = q_angles.device
        if device.type != "cpu":
            q_angles_cpu = q_angles.cpu()
            q_out = self.qlayer(q_angles_cpu).to(device)
        else:
            q_out = self.qlayer(q_angles)               # (N_jets,), expectation ∈ (-1, 1)

        return torch.sigmoid(q_out.unsqueeze(-1))        # (N_jets, 1)


class LundNet(torch.nn.Module):
    def __init__(self):
        super(LundNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        # self.seq2 = nn.Sequential(nn.Linear(384, 256),
        #                           nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(385, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Ntrk = data.Ntrk
        Ntrk = torch.unsqueeze(Ntrk, 1)
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = torch.cat( (x, Ntrk) ,dim=1)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class LundNet_plus_GN2X(torch.nn.Module):
    def __init__(self):
        super(LundNet_plus_GN2X, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        # self.seq2 = nn.Sequential(nn.Linear(384, 256),
        #                           nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(389, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Ntrk = data.Ntrk
        Ntrk = torch.unsqueeze(Ntrk, 1)
        
        GN2X_pqcd = data.GN2X_pqcd
        GN2X_pqcd = torch.unsqueeze(GN2X_pqcd, 1)
        GN2X_phbb = data.GN2X_phbb
        GN2X_phbb = torch.unsqueeze(GN2X_phbb, 1)
        GN2X_ptop = data.GN2X_ptop
        GN2X_ptop = torch.unsqueeze(GN2X_ptop, 1)
        GN2X_phcc = data.GN2X_phcc
        GN2X_phcc = torch.unsqueeze(GN2X_phcc, 1)
        
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        #x = torch.cat( (x, Ntrk) ,dim=1)
        x = torch.cat( (x, Ntrk, GN2X_pqcd, GN2X_phbb, GN2X_ptop, GN2X_phcc) ,dim=1)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class LundNet_plus_GN3X(torch.nn.Module):
    def __init__(self):
        super(LundNet_plus_GN3X, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()), aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()), aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()), aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                  nn.BatchNorm1d(384),
                                  nn.ReLU())
        
        self.seq2 = nn.Sequential(nn.Linear(394, 256),  # 384 + 1 (Ntrk) + 9 GN3X
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Ntrk = torch.unsqueeze(data.Ntrk, 1)

        # GN3X tagger outputs
        g_keys = [
            "GN3X_phtautauhad", "GN3X_phbb", "GN3X_phcc", "GN3X_ptop",
            "GN3X_pqcdbb", "GN3X_pqcdbx", "GN3X_pqcdcx", "GN3X_pqcdll", "GN3X_pWqq"
        ]
        g_list = [torch.unsqueeze(getattr(data, k), 1) for k in g_keys]

        # EdgeConv stack
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)  # 448
        x = self.seq1(x)
        x = global_mean_pool(x, batch)  # (batch_size, 384)

        x = torch.cat([x, Ntrk] + g_list, dim=1)  # final dim: 384 + 1 + 9 = 394
        x = self.seq2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        return torch.sigmoid(x)

class LundNet_old(torch.nn.Module):
    def __init__(self):
        super(LundNet_old, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        '''
        print("x1->",x1)
        print( len(x1[0]))
        print("x2->",x2)
        print( len(x2[0]))
        print("x3->",x3)
        print( len(x3[0]))
        print("x4->",x4)
        print( len(x4[0]))
        print("x5->",x5)
        print( len(x5[0]))
        print("x6->",x6)
        print( len(x6[0]))
        print( len(x[30]) )
        print( len(x) )
        print(x[:2])
        '''
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class GATNet(torch.nn.Module):
    def __init__(self):
        in_channels = 3
        super(GATNet, self).__init__()
        self.conv1 = GATConv(4, 16, heads=8, dropout=0.1)
        self.conv2 = GATConv(16 * 8, 16, heads=8, dropout=0.1)
        self.conv3 = GATConv(16 * 8, 32, heads=8, dropout=0.1)
        self.conv4 = GATConv(32 * 8, 32, heads=16, dropout=0.1)
        self.conv5 = GATConv(16 * 32, 64, heads=16, dropout=0.1)
        self.conv6 = GATConv(64 * 16, 64, heads=16, dropout=0.1)
        self.seq1 = nn.Sequential(nn.Linear(3072, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = torch.nn.Linear(256, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(3, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(32,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv4 = GINConv(nn.Sequential(nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(64, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        self.conv6 = GINConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class EdgeGinNet(torch.nn.Module):
    def __init__(self):
        super(EdgeGinNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv4 = GINConv(nn.Sequential(nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(64, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        self.conv6 = GINConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)



class PNANet(torch.nn.Module):
    def __init__(self,in_channels):
        aggregators = ['sum','mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']
        in_channels = 3
        super(PNANet, self).__init__()
        self.conv1 = PNAConv(in_channels, out_channels=32, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNAConv(in_channels=32, out_channels=64, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNAConv(in_channels=64, out_channels=128, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv4 = PNAConv(in_channels=128, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv5 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv6 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.seq1 = nn.Sequential(nn.Linear(992, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2,x3,x4,x5,x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        return F.sigmoid(x)

class ConcatDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return xA, xB

    def __len__(self):
        return len(self.datasetA)



################################################################################
## from adv_script
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input

        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.bn1 = nn.BatchNorm1d(num_features=in_features)
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        #self.sigma = nn.Sequential(
        #    nn.Linear(in_features, out_features * num_gaussians),
        #    nn.Softplus()
        #)

        #self.mu = nn.Sequential(
        #    nn.Linear(in_features, out_features * num_gaussians),
        #    nn.Sigmoid()
        #)

        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        minibatch = self.bn1(minibatch)
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    ret = torch.where(ret == 0, ret + 1E-20, ret)
    return torch.prod(ret, 2)



def mdn_loss(pi, sigma, mu, target, weight):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """

    #print("pi->",pi.size() )
    #print("sigma->",sigma.size() )
    #print("mu->",mu.size() )
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -weight*torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


###################################################################----------------------------------------------------
class MDN_new(nn.Module):
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN_new, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.bn1 = nn.BatchNorm1d(num_features=in_features)
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, out_features * num_gaussians),
            nn.Softplus()
        )

        self.mu = nn.Sequential(
            nn.Linear(in_features, out_features * num_gaussians),
            nn.Sigmoid()
        )
    def forward(self, minibatch):
        minibatch = self.bn1(minibatch)
        pi = self.pi(minibatch)
        #sigma = torch.exp(self.sigma(minibatch))
        sigma = self.sigma(minibatch)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability_new(sigma, mu, target):
    target = (target - 40)/( 300 - 40 )
    #print("mass_0to1",target[:2])
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    ret = torch.where(ret == 0, ret + 1E-20, ret)
    return torch.prod(ret, 2)


def pi_redefinition(device, pi, sigma, mu):
    # redefine pi in order to obtain normalized distributions inside [0,1] interval
    # mu = Mean
    # sigma = width
    t_ones = torch.ones( mu.size() )
    t_zeros = torch.zeros( mu.size() )
    t_ones = t_ones.to(device)
    t_zeros = t_zeros.to(device)

    z0 = (t_zeros - mu) / sigma
    z1 = (t_ones - mu) / sigma
    out_interval = 0.5 * (1. + torch.erf(z1 / np.sqrt(2.)))  - 0.5 * (1. + torch.erf(z0 / np.sqrt(2.)))  # area inside [0,1]
    #print("pi size->",pi.size(),"   out_interval size->",out_interval.size() )
    # Clamp: a component whose mass inside [0,1] underflows to 0 (mu at the
    # sigmoid edge + tiny sigma) would otherwise give pi/0 -> inf -> NaN loss.
    out_interval = out_interval.clamp(min=1e-6)
    pi_2 = pi / out_interval.view(pi.size())
    #print("pi_2[0]",pi_2[0])
    '''
    for i in range (0,len(pi_2) ): # sum all gaussians must be =1
        #pi_2[i] = pi_2[i]/torch.sum(pi_2[i])
        pi_2[i] /= torch.sum(pi_2[i])
    '''

    #print("pi size after:",pi_2.size() )
    #print(pi_2.size,"  ",pi_2[:2])
    return pi_2

def mdn_loss_new(device, pi, sigma, mu, target, weight):
    #print("pi---------------------------------------")
    #print(pi[:2])

    sigma = sigma.clamp(min=1e-3)   # Softplus can underflow to ~0 -> div by zero
    pi_2 = pi_redefinition(device, pi, sigma, mu)
    #pi_2 = pi

    ##redefine mu inside interval [-0.3,1.3] ~~ mu*1.6 - 0.3
    mu_shift = 0.3 * torch.ones( mu.size() )
    mu_shift = mu_shift.to(device)
    mu_2 = 1.6 * mu - mu_shift

    '''
    print("---------------------------------------")
    print("mass",target[:2])
    print("---------------------------------------")
    print("mu size->", mu.size(), "   pi size->",pi.size() ,"   sigma size->", sigma.size()  )
    print("mu_2---------------------------------------")
    print(mu_2[:2])
    print("pi_2---------------------------------------")
    print(pi_2[:2])
    print("sigma---------------------------------------")
    print(sigma[:2])
    print("---------------------------------------")
    '''
    prob = pi_2 * gaussian_probability_new(sigma, mu_2, target)
    nll = -weight*torch.log(torch.sum(prob, dim=1).clamp(min=1e-20))
    #nll = weight*torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)

class Adversary_new(nn.Module):
    def __init__(self, lambda_parameter, num_gaussians):
        super(Adversary_new, self).__init__()
        self.gauss = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    #nn.LeakyReLU(),
    MDN_new(64, 1, num_gaussians)
)
        self.revgrad = GradientReversal(lambda_parameter)
        #print("lambda = {}".format(lambda_parameter))
    def forward(self, x):
        x = self.revgrad(x) # important hyperparameter, the scale,   # tells by how much the classifier is punished
        x = self.gauss(x)
        return x

###################################################################----------------------------------------------------

def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


# The architecture of the adversary could be changed
class Adversary(nn.Module):
    def __init__(self, lambda_parameter, num_gaussians):
        super(Adversary, self).__init__()
        self.gauss = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    MDN(64, 1, num_gaussians)
)
        self.revgrad = GradientReversal(lambda_parameter)
        #print("lambda = {}".format(lambda_parameter))

    def forward(self, x):
        x = self.revgrad(x) # important hyperparameter, the scale,
                                     # tells by how much the classifier is punished
        x = self.gauss(x)
        return x
