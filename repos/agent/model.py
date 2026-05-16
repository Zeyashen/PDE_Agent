"""
model.py — DeepONet for 1D Burgers

范式A：直接算子映射，无自回归误差累积。
  Branch net : 编码初始条件 u(x, t=0~9) → latent vector (B, p)
  Trunk net  : 编码查询坐标 (t, x)      → basis vectors (B, N_q, p)
  输出       : dot(branch, trunk) + bias → u(x,t) at query points

三种 Branch net 供 Agent 探索：
  mlp         : flatten → MLP
  cnn         : 1D CNN，保留空间结构
  fno_encoder : Fourier 编码器

Task-2 扩展预留：
  条件化接口 build_model(cfg, task=2) 会在 branch net 输入上
  拼接 Nu embedding（训练时可选，推理时置零），
  不改动 Task-1 的任何逻辑。

Agent 可修改区域：
  AGENT_BRANCH_BEGIN / AGENT_BRANCH_END
  AGENT_TRUNK_BEGIN  / AGENT_TRUNK_END
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str) -> nn.Module:
    return {
        "relu":  nn.ReLU(),
        "tanh":  nn.Tanh(),
        "gelu":  nn.GELU(),
        "silu":  nn.SiLU(),
        "swish": nn.SiLU(),
    }.get(name, nn.Tanh())


# ============================================================
# Branch Net 变体
# ============================================================
class MLPBranch(nn.Module):
    def __init__(self, T_in, X, latent_dim, depth, width, activation="tanh"):
        super().__init__()
        in_dim = T_in * X
        layers = [nn.Linear(in_dim, width), get_activation(activation)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), get_activation(activation)]
        layers.append(nn.Linear(width, latent_dim))
        self.net  = nn.Sequential(*layers)
        self.T_in = T_in
        self.X    = X

    def forward(self, u0):
        return self.net(u0.reshape(u0.shape[0], -1))


class CNNBranch(nn.Module):
    def __init__(self, T_in, X, latent_dim, depth, width,
                 activation="gelu", kernel_size=5):
        super().__init__()
        self.act   = get_activation(activation)
        convs      = [nn.Conv1d(T_in, width, kernel_size, padding=kernel_size//2)]
        for _ in range(depth - 1):
            convs.append(nn.Conv1d(width, width, kernel_size,
                                   padding=kernel_size//2))
        self.convs = nn.ModuleList(convs)
        self.fc    = nn.Linear(width, latent_dim)

    def forward(self, u0):
        x = u0
        for conv in self.convs:
            x = self.act(conv(x))
        return self.fc(x.mean(dim=-1))


class _SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes   = modes
        self.weights = nn.Parameter(
            (1/(in_ch*out_ch)) *
            torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B  = x.shape[0]
        xf = torch.fft.rfft(x)
        out = torch.zeros(B, self.weights.shape[1], x.shape[-1]//2+1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", xf[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out, n=x.shape[-1])


class FNOEncoderBranch(nn.Module):
    def __init__(self, T_in, X, latent_dim, depth, width,
                 modes=16, activation="gelu"):
        super().__init__()
        self.act       = get_activation(activation)
        self.lift      = nn.Linear(T_in, width)
        self.spec_convs = nn.ModuleList(
            [_SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.res_convs  = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc         = nn.Linear(width, latent_dim)

    def forward(self, u0):
        x = self.lift(u0.permute(0, 2, 1)).permute(0, 2, 1)
        for sc, rc in zip(self.spec_convs, self.res_convs):
            x = self.act(sc(x) + rc(x))
        return self.fc(x.mean(dim=-1))


# ============================================================
# Trunk Net（Fourier 特征嵌入）
# ============================================================
class TrunkNet(nn.Module):
    """
    输入: (B, N_query, coord_dim)  coord_dim=2(task1) 或 3(task2,含Nu)
    输出: (B, N_query, latent_dim)
    """
    def __init__(self, latent_dim, depth, width,
                 activation="tanh", fourier_features=64, coord_dim=2):
        super().__init__()
        self.register_buffer(
            "B_mat", torch.randn(coord_dim, fourier_features) * 10.0)
        in_dim = 2 * fourier_features
        layers = [nn.Linear(in_dim, width), get_activation(activation)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), get_activation(activation)]
        layers.append(nn.Linear(width, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        proj = coords @ self.B_mat
        ff   = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(ff)


# ============================================================
# DeepONet 主模型
# ============================================================
class DeepONet(nn.Module):
    """
    Task-1 / Task-2 通用 DeepONet。
    task=1: branch 输入 (B, T_in, X)，trunk 坐标 2D (t,x)
    task=2: 同 task=1，trunk 坐标 2D (t,x)，Nu 信息由 branch 隐式编码
            或通过 nu_embed 显式注入（训练时可选，推理时置零）
    """
    def __init__(
        self,
        T_in:             int   = 10,
        X:                int   = 256,
        latent_dim:       int   = 128,
        branch_type:      str   = "mlp",
        branch_depth:     int   = 4,
        branch_width:     int   = 256,
        trunk_depth:      int   = 4,
        trunk_width:      int   = 256,
        activation:       str   = "tanh",
        fourier_features: int   = 64,
        # Task-2 扩展：Nu 条件化（task=1 时忽略）
        use_nu_embed:     bool  = False,
        nu_embed_dim:     int   = 16,
    ):
        super().__init__()
        self.use_nu_embed = use_nu_embed

        # Nu embedding（Task-2 用，Task-1 不初始化）
        if use_nu_embed:
            self.nu_embed = nn.Sequential(
                nn.Linear(1, nu_embed_dim),
                nn.Tanh(),
                nn.Linear(nu_embed_dim, latent_dim),
            )

        # ===== AGENT_BRANCH_BEGIN =====
        if branch_type == "mlp":
            self.branch = MLPBranch(
                T_in, X, latent_dim, branch_depth, branch_width, activation)
        elif branch_type == "cnn":
            self.branch = CNNBranch(
                T_in, X, latent_dim, branch_depth, branch_width, activation)
        elif branch_type == "fno_encoder":
            self.branch = FNOEncoderBranch(
                T_in, X, latent_dim, branch_depth, branch_width,
                activation=activation)
        else:
            raise ValueError(f"未知 branch_type: {branch_type}")
        # ===== AGENT_BRANCH_END =====

        # ===== AGENT_TRUNK_BEGIN =====
        self.trunk = TrunkNet(
            latent_dim, trunk_depth, trunk_width,
            activation, fourier_features, coord_dim=2)
        # ===== AGENT_TRUNK_END =====

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, coords, nu=None):
        """
        u0    : (B, T_in, X)
        coords: (B, N_query, 2)
        nu    : (B, 1) 可选，Task-2 训练时传入，推理时传 None
        """
        b_out = self.branch(u0)                          # (B, latent_dim)
        if self.use_nu_embed and nu is not None:
            b_out = b_out + self.nu_embed(nu)            # 残差注入
        t_out = self.trunk(coords)                       # (B, N_q, latent_dim)
        return (b_out.unsqueeze(1) * t_out).sum(-1) + self.bias

    def predict_full(self, u0, t_steps=190, x_points=256,
                     device=None, chunk=8192):
        """
        范式A全量推理，分块处理防 OOM。
        返回: (B, t_steps, x_points)
        """
        if device is None:
            device = next(self.parameters()).device
        B = u0.shape[0]
        t_n = torch.linspace(0, 1, t_steps,  device=device)
        x_n = torch.linspace(0, 1, x_points, device=device)
        tt, xx   = torch.meshgrid(t_n, x_n, indexing='ij')
        coords_all = torch.stack([tt.flatten(), xx.flatten()], -1)
        total_q  = t_steps * x_points
        pred_flat = torch.zeros(B, total_q, device=device)
        with torch.no_grad():
            for s in range(0, total_q, chunk):
                e = min(s + chunk, total_q)
                c = coords_all[s:e].unsqueeze(0).expand(B, -1, -1)
                pred_flat[:, s:e] = self.forward(u0, c)
        return pred_flat.reshape(B, t_steps, x_points)


def build_model(cfg: dict, task: int = 1) -> DeepONet:
    """从 registry 配置字典构建模型。task=1 或 2。"""
    return DeepONet(
        T_in             = cfg.get("T_in",            10),
        X                = cfg.get("X",              256),
        latent_dim       = cfg.get("latent_dim",      128),
        branch_type      = cfg.get("branch_type",    "mlp"),
        branch_depth     = cfg.get("branch_depth",     4),
        branch_width     = cfg.get("branch_width",   256),
        trunk_depth      = cfg.get("trunk_depth",      4),
        trunk_width      = cfg.get("trunk_width",    256),
        activation       = cfg.get("activation",    "tanh"),
        fourier_features = cfg.get("fourier_features", 64),
        use_nu_embed     = (task == 2 and cfg.get("use_nu_embed", False)),
        nu_embed_dim     = cfg.get("nu_embed_dim",    16),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)