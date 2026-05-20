"""
model.py — DeepONet Task-2 框架层模型

这是框架层文件，由人工维护，不提交到 code/。
Agent 通过 train.py 调用这里定义的模型。

提供三种 Branch Net 供 Agent 在 train.py 里选择：
  DeepONet(branch_type="cnn")         ← 推荐，性能最好
  DeepONet(branch_type="mlp")         ← 最简单
  DeepONet(branch_type="fno_encoder") ← 频域特征

Task2 特有功能：Nu 条件化（Nu Dropout 机制）
  DeepONet(nu_dim=0)  ← 默认，不使用 Nu（和 Task1 一样）
  DeepONet(nu_dim=1)  ← 启用 Nu 条件化

  训练时传 nu：model(u0, coords, nu=nu_tensor)
  推理时不传：model.predict_full(u0)  ← 永远只用初始条件

  Nu Dropout：train.py 里随机把 nu 置 0（建议 dropout_rate=0.3）
  让模型同时学会有 Nu 和无 Nu 两种情况，推理时传 nu=0 效果更好。
"""

import torch
import torch.nn as nn
import numpy as np


# ── Branch Net 变体 ───────────────────────────────────────────

class MLPBranch(nn.Module):
    def __init__(self, T_in=10, X=256, width=256, depth=4,
                 latent_dim=128, activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        in_dim = T_in * X
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers.append(nn.Linear(width, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, u0):
        return self.net(u0.reshape(u0.shape[0], -1))


class CNNBranch(nn.Module):
    def __init__(self, T_in=10, width=128, depth=4,
                 latent_dim=128, activation="gelu", kernel_size=5):
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers = [nn.Conv1d(T_in, width, kernel_size, padding=kernel_size//2), act()]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(width, width, kernel_size, padding=kernel_size//2), act()]
        self.convs = nn.Sequential(*layers)
        self.fc    = nn.Linear(width, latent_dim)

    def forward(self, u0):
        x = self.convs(u0)
        x = x.mean(dim=-1)
        return self.fc(x)


class _SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes   = modes
        self.weights = nn.Parameter(
            (1/(in_ch*out_ch)) * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        B   = x.shape[0]
        xf  = torch.fft.rfft(x)
        out = torch.zeros(B, self.weights.shape[1], x.shape[-1]//2+1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", xf[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out, n=x.shape[-1])


class FNOEncoderBranch(nn.Module):
    def __init__(self, T_in=10, width=64, depth=4,
                 latent_dim=128, modes=16, activation="gelu"):
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        self.lift       = nn.Linear(T_in, width)
        self.spec_convs = nn.ModuleList([_SpectralConv1d(width, width, modes)
                                         for _ in range(depth)])
        self.res_convs  = nn.ModuleList([nn.Conv1d(width, width, 1)
                                         for _ in range(depth)])
        self.act        = act()
        self.fc         = nn.Linear(width, latent_dim)

    def forward(self, u0):
        x = self.lift(u0.permute(0, 2, 1)).permute(0, 2, 1)
        for sc, rc in zip(self.spec_convs, self.res_convs):
            x = self.act(sc(x) + rc(x))
        x = x.mean(dim=-1)
        return self.fc(x)


# ── Trunk Net ─────────────────────────────────────────────────

class TrunkNet(nn.Module):
    def __init__(self, latent_dim=128, width=256, depth=4,
                 ff_dim=64, activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        self.register_buffer('B_mat', torch.randn(2, ff_dim) * 10.0)
        in_dim = 2 * ff_dim
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers.append(nn.Linear(width, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        proj = coords @ self.B_mat
        ff   = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(ff)


# ── Nu Encoder ────────────────────────────────────────────────

class NuEncoder(nn.Module):
    """
    把标量 Nu 值编码成 latent_dim 维向量，加到 branch 输出上。
    输入 nu: (B, 1)，输出 (B, latent_dim)
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, nu):
        # nu: (B, 1) 或 (B,)
        if nu.dim() == 1:
            nu = nu.unsqueeze(-1)
        return self.net(nu)  # (B, latent_dim)


# ── DeepONet 主模型 ───────────────────────────────────────────

class DeepONet(nn.Module):
    """
    DeepONet 范式A（Task2 版本，支持 Nu 条件化）。

    forward(u0, coords, nu=None) → (B, N_q)
      u0:    (B, T_in, X)
      coords:(B, N_q, 2)
      nu:    (B,) 或 (B, 1)，可选。nu=None 或全零时不使用。

    predict_full(u0, t_steps=200) → numpy (B, t_steps, 256)
      推理时永远不传 Nu，只用初始条件。

    Nu 使用方式（train.py 里）：
      # 启用 Nu 条件化
      model = DeepONet(nu_dim=1)

      # Nu Dropout（建议 dropout_rate=0.3）
      mask = torch.rand(B) > dropout_rate
      nu_input = nu * mask.float().to(device)
      pred = model(u0, coords, nu=nu_input)

      # 默认不用 Nu
      model = DeepONet(nu_dim=0)
      pred = model(u0, coords)
    """
    def __init__(
        self,
        T_in:         int = 10,
        X:            int = 256,
        latent_dim:   int = 128,
        branch_type:  str = "cnn",
        branch_depth: int = 4,
        branch_width: int = 128,
        trunk_depth:  int = 4,
        trunk_width:  int = 256,
        activation:   str = "gelu",
        ff_dim:       int = 64,
        nu_dim:       int = 0,      # 0=不用Nu, 1=启用Nu条件化
    ):
        super().__init__()
        self.T_in       = T_in
        self.latent_dim = latent_dim
        self.nu_dim     = nu_dim

        if branch_type == "cnn":
            self.branch = CNNBranch(
                T_in=T_in, width=branch_width, depth=branch_depth,
                latent_dim=latent_dim, activation=activation)
        elif branch_type == "mlp":
            self.branch = MLPBranch(
                T_in=T_in, X=X, width=branch_width, depth=branch_depth,
                latent_dim=latent_dim, activation=activation)
        elif branch_type == "fno_encoder":
            self.branch = FNOEncoderBranch(
                T_in=T_in, width=branch_width, depth=branch_depth,
                latent_dim=latent_dim, activation=activation)
        else:
            raise ValueError(f"未知 branch_type: {branch_type}")

        self.trunk = TrunkNet(
            latent_dim=latent_dim, width=trunk_width, depth=trunk_depth,
            ff_dim=ff_dim, activation=activation)

        # Nu encoder（只有 nu_dim > 0 时才创建）
        self.nu_encoder = NuEncoder(latent_dim) if nu_dim > 0 else None

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0: torch.Tensor,
                coords: torch.Tensor,
                nu: torch.Tensor = None) -> torch.Tensor:
        b = self.branch(u0)      # (B, latent_dim)

        # Nu 条件化：把 Nu 编码加到 branch 输出上
        if self.nu_encoder is not None and nu is not None:
            b = b + self.nu_encoder(nu)  # (B, latent_dim)

        t = self.trunk(coords)   # (B, N_q, latent_dim)
        return (b.unsqueeze(1) * t).sum(dim=-1) + self.bias  # (B, N_q)

    def predict_full(self, u0: torch.Tensor,
                     t_steps: int = 200,
                     x_points: int = 256) -> np.ndarray:
        """
        推理接口：永远不传 Nu，只用初始条件预测。
        返回: numpy (B, t_steps, x_points)
        """
        device = next(self.parameters()).device
        B      = u0.shape[0]

        t_norm = torch.linspace(0, 1, t_steps,  device=device)
        x_norm = torch.linspace(0, 1, x_points, device=device)
        tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
        coords_all = torch.stack([tt.flatten(), xx.flatten()], dim=-1)
        coords_all = coords_all.unsqueeze(0).expand(B, -1, -1)

        total     = t_steps * x_points
        batch_q   = x_points * 50
        pred_flat = torch.zeros(B, total, device=device)

        with torch.no_grad():
            for start in range(0, total, batch_q):
                end = min(start + batch_q, total)
                pred_flat[:, start:end] = self.forward(
                    u0, coords_all[:, start:end, :], nu=None)

        return pred_flat.reshape(B, t_steps, x_points).cpu().numpy()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import tempfile, os
    print("=" * 50)
    print("model.py (Task2) unittest")
    print("=" * 50)

    errors = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    def check(name, fn):
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, str(e)))

    B, N_q = 2, 512

    # Test 1: nu_dim=0 forward（和Task1一样）
    print("[1] nu_dim=0 forward shape")
    def t1():
        m   = DeepONet(nu_dim=0).to(device)
        u0  = torch.randn(B, 10, 256, device=device)
        c   = torch.rand(B, N_q, 2, device=device)
        out = m(u0, c)
        assert out.shape == (B, N_q)
        assert m.nu_encoder is None
    check("nu_dim=0 forward", t1)

    # Test 2: nu_dim=1 forward，传 nu
    print("\n[2] nu_dim=1 forward with nu")
    def t2():
        m   = DeepONet(nu_dim=1).to(device)
        u0  = torch.randn(B, 10, 256, device=device)
        c   = torch.rand(B, N_q, 2, device=device)
        nu  = torch.rand(B, device=device) * 0.01
        out = m(u0, c, nu=nu)
        assert out.shape == (B, N_q)
        assert m.nu_encoder is not None
    check("nu_dim=1 forward with nu", t2)

    # Test 3: nu_dim=1 forward，不传 nu（推理时）
    print("\n[3] nu_dim=1 forward without nu（推理模式）")
    def t3():
        m   = DeepONet(nu_dim=1).to(device)
        u0  = torch.randn(B, 10, 256, device=device)
        c   = torch.rand(B, N_q, 2, device=device)
        out = m(u0, c, nu=None)  # 推理时不传nu
        assert out.shape == (B, N_q)
    check("nu_dim=1 forward without nu", t3)

    # Test 4: Nu Dropout 模拟
    print("\n[4] Nu Dropout 模拟")
    def t4():
        m   = DeepONet(nu_dim=1).to(device)
        u0  = torch.randn(B, 10, 256, device=device)
        c   = torch.rand(B, N_q, 2, device=device)
        nu  = torch.rand(B, device=device) * 0.01
        # 30% dropout
        mask    = (torch.rand(B, device=device) > 0.3).float()
        nu_drop = nu * mask
        out = m(u0, c, nu=nu_drop)
        assert out.shape == (B, N_q)
    check("Nu Dropout 模拟", t4)

    # Test 5: predict_full 默认200步
    print("\n[5] predict_full shape（200步）")
    def t5():
        m    = DeepONet(nu_dim=1).to(device)
        u0   = torch.randn(B, 10, 256, device=device)
        pred = m.predict_full(u0)
        assert pred.shape == (B, 200, 256), f"shape={pred.shape}"
        assert isinstance(pred, np.ndarray)
        assert not np.isnan(pred).any()
    check("predict_full 200步", t5)

    # Test 6: 三种 branch_type
    print("\n[6] 三种 branch_type")
    for bt in ["cnn", "mlp", "fno_encoder"]:
        def t6(bt=bt):
            m   = DeepONet(branch_type=bt, nu_dim=1).to(device)
            u0  = torch.randn(B, 10, 256, device=device)
            c   = torch.rand(B, N_q, 2, device=device)
            nu  = torch.rand(B, device=device) * 0.01
            out = m(u0, c, nu=nu)
            assert out.shape == (B, N_q)
        check(f"branch_type={bt}", t6)

    # Test 7: checkpoint 保存和加载
    print("\n[7] checkpoint 保存和加载")
    def t7():
        m = DeepONet(nu_dim=1, branch_width=64).to(device)
        u0 = torch.randn(B, 10, 256, device=device)
        c  = torch.rand(B, N_q, 2, device=device)
        nu = torch.rand(B, device=device) * 0.01
        m.eval()
        with torch.no_grad():
            out1 = m(u0, c, nu=nu).cpu().numpy()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            torch.save({
                "epoch": 1, "val_loss": 0.1,
                "model_state_dict": m.state_dict(),
                "nu_dim": 1, "branch_width": 64,
            }, ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            m2 = DeepONet(nu_dim=ckpt["nu_dim"],
                          branch_width=ckpt["branch_width"]).to(device)
            m2.load_state_dict(ckpt["model_state_dict"])
            m2.eval()
            with torch.no_grad():
                out2 = m2(u0, c, nu=nu).cpu().numpy()
            assert np.allclose(out1, out2, atol=1e-5)
        finally:
            os.unlink(ckpt_path)
    check("checkpoint 保存/加载/一致性", t7)

    print("\n" + "=" * 50)
    if errors:
        print(f"❌ {len(errors)} 个测试失败:")
        for name, err in errors:
            print(f"   {name}: {err}")
    else:
        print("✅ 所有测试通过")
    print("=" * 50)