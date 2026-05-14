"""FNO 模型定义 + 工厂函数。

- FNO1d         我们的标准 FNO (width/n_layers 可配)
- ClassicFNO1d  官方 PDEBench 结构 (4 层固定，用于加载官方 ckpt)

LLM 通过 model.architecture 字段切换，不能改这里的代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, inp, weights):
        return torch.einsum("bix,iox->box", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """通用 1D FNO，width/n_layers 可配。

    输入: (B, in_channels, X) — in_channels = window_size + 1
    输出: (B, out_channels, X) — out_channels=1 (预测下一时间步)
    """

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int,
        out_channels: int,
        n_layers: int,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(in_channels, width)
        self.convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


class ClassicFNO1d(nn.Module):
    """官方 PDEBench FNO 结构 (4 层固定，key 命名 conv0..conv3)。

    in_channels 固定为 11，out_channels 可调。
    用于加载官方 ckpt (modes=12, width=20)。
    """

    def __init__(self, modes: int = 12, width: int = 20, out_channels: int = 1):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(11, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


def build_model(cfg: ModelConfig) -> nn.Module:
    """根据 ModelConfig 构造模型。"""
    if cfg.architecture == "FNO1d":
        return FNO1d(
            modes=cfg.modes,
            width=cfg.width,
            in_channels=cfg.window_size + 1,
            out_channels=1,
            n_layers=cfg.n_layers,
        )
    elif cfg.architecture == "ClassicFNO1d":
        return ClassicFNO1d(
            modes=cfg.modes,
            width=cfg.width,
            out_channels=1,
        )
    elif cfg.architecture == "DeepONet1d":
        return DeepONet1d(
            x_dim=256,
            hidden_dim=cfg.hidden_dim,
            p_dim=cfg.p_dim,
            branch_layers=cfg.branch_layers,
            trunk_layers=cfg.trunk_layers,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")


def detect_architecture(state_dict: dict) -> str:
    """从 state_dict key 推断结构。"""
    if any(k.startswith("conv0.") for k in state_dict):
        return "ClassicFNO1d"
    if any(k.startswith("convs.0.") for k in state_dict):
        return "FNO1d"
    if any(k.startswith("branch.") for k in state_dict):
        return "DeepONet1d"
    raise ValueError(
        f"无法识别的 state_dict, top-level keys: {list(state_dict)[:5]}..."
    )


def load_ckpt(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    """通用 ckpt 加载。

    支持:
      1. 我们训练得到的 (含 'model_state_dict' + 'args')
      2. 官方 PDEBench (纯 state_dict)

    返回 (model, meta)，meta 含 epoch / val_loss / args / architecture。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        args = ckpt.get("args", {})
        meta = {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "args": args,
        }
    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]
    ):
        state_dict = ckpt
        args = {}
        meta = {"epoch": None, "val_loss": None, "args": {}}
    else:
        raise ValueError(f"无法识别的 ckpt 格式: {ckpt_path}")

    arch = detect_architecture(state_dict)

    if arch == "ClassicFNO1d":
        width = state_dict["fc0.weight"].shape[0]
        modes = state_dict["conv0.weights1"].shape[-1]
        out_channels = state_dict["fc2.weight"].shape[0]
        model = ClassicFNO1d(modes=modes, width=width, out_channels=out_channels)
    elif arch == "DeepONet1d":
        x_dim      = state_dict["branch.U_layer.weight"].shape[1]
        hidden_dim = state_dict["branch.U_layer.weight"].shape[0]
        p_dim      = state_dict["branch.out_layer.weight"].shape[0]
        branch_layers = sum(1 for k in state_dict if k.startswith("branch.hidden.")) // 2
        trunk_layers  = sum(1 for k in state_dict if k.startswith("trunk.hidden."))  // 2
        model = DeepONet1d(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            p_dim=p_dim,
            branch_layers=branch_layers,
            trunk_layers=trunk_layers,
        )
    else:
        modes = state_dict["convs.0.weights1"].shape[-1]
        width = state_dict["fc0.weight"].shape[0]
        in_channels = state_dict["fc0.weight"].shape[1]
        n_layers = max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("convs.")
        ) + 1
        model = FNO1d(
            modes=modes,
            width=width,
            in_channels=in_channels,
            out_channels=1,
            n_layers=n_layers,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    meta["architecture"] = arch
    return model, meta


# ============================================================
# DeepONet1d (Physics-Informed)
# 参考: PI-DeepONet_core/Burger/PI_DeepONet_Burger.ipynb
# ============================================================

class ModifiedMLP(nn.Module):
    """带门控机制的 MLP（官方 modified_MLP 的 PyTorch 实现）。
    
    每层: h = tanh(Wh+b)
    输出: h * U + (1-h) * V  （门控融合）
    """
    def __init__(self, layers: list):
        super().__init__()
        self.U_layer = nn.Linear(layers[0], layers[1])
        self.V_layer = nn.Linear(layers[0], layers[1])
        self.hidden  = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-2)]
        )
        self.out_layer = nn.Linear(layers[-2], layers[-1])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        U = torch.tanh(self.U_layer(x))
        V = torch.tanh(self.V_layer(x))
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
            h = h * U + (1 - h) * V
        return self.out_layer(h)


class DeepONet1d(nn.Module):
    """Physics-Informed DeepONet for 1D Burgers.

    输入:
        u0: (B, X)      初始条件
        xt: (B, N, 2)   查询坐标 (t_norm, x_norm)，归一化到[0,1]
    输出:
        s:  (B, N)      预测解 u(x,t)
    """
    def __init__(
        self,
        x_dim: int = 256,
        hidden_dim: int = 128,
        p_dim: int = 128,
        branch_layers: int = 6,
        trunk_layers: int = 6,
    ):
        super().__init__()
        b_layers = [x_dim]    + [hidden_dim] * branch_layers + [p_dim]
        t_layers = [2]        + [hidden_dim] * trunk_layers  + [p_dim]
        self.branch  = ModifiedMLP(b_layers)
        self.trunk   = ModifiedMLP(t_layers)
        self.bias    = nn.Parameter(torch.zeros(1))
        self.x_dim   = x_dim
        self.t_scale = 10.0  # 训练数据最大时间（秒）

    def forward(self, u0, xt):
        # u0: (B, X), xt: (B, N, 2)
        b = self.branch(u0)              # (B, p)
        B, N, _ = xt.shape
        xt_flat = xt.reshape(B*N, 2)
        t_flat  = self.trunk(xt_flat)    # (B*N, p)
        t_out   = t_flat.reshape(B, N, -1)  # (B, N, p)
        out = torch.einsum('bp,bnp->bn', b, t_out) + self.bias  # (B, N)
        return out
