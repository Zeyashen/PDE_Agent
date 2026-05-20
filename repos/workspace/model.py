"""
model.py — DeepONet Task-1 框架层模型

这是框架层文件，由人工维护，不提交到 code/。
Agent 通过 train.py 调用这里定义的模型。

提供三种 Branch Net 供 Agent 在 train.py 里选择：
  DeepONet(branch_type="cnn")         ← 推荐，性能最好
  DeepONet(branch_type="mlp")         ← 最简单
  DeepONet(branch_type="fno_encoder") ← 频域特征
"""

import torch
import torch.nn as nn
import numpy as np


# ── Branch Net 变体 ───────────────────────────────────────────

class MLPBranch(nn.Module):
    """MLP branch：flatten 初始10步 → 全连接层"""
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
        # u0: (B, T_in, X)
        return self.net(u0.reshape(u0.shape[0], -1))


class CNNBranch(nn.Module):
    """CNN branch：1D 卷积保留空间结构，全局平均池化"""
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
        # u0: (B, T_in, X)
        x = self.convs(u0)     # (B, width, X)
        x = x.mean(dim=-1)     # (B, width)
        return self.fc(x)      # (B, latent_dim)


class _SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes   = modes
        self.weights = nn.Parameter(
            (1/(in_ch*out_ch)) * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B    = x.shape[0]
        xf   = torch.fft.rfft(x)
        out  = torch.zeros(B, self.weights.shape[1], x.shape[-1]//2+1,
                           dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", xf[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out, n=x.shape[-1])


class FNOEncoderBranch(nn.Module):
    """FNO encoder branch：Fourier 谱卷积提取频域特征"""
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
        # u0: (B, T_in, X)
        x = self.lift(u0.permute(0, 2, 1)).permute(0, 2, 1)  # (B, width, X)
        for sc, rc in zip(self.spec_convs, self.res_convs):
            x = self.act(sc(x) + rc(x))
        x = x.mean(dim=-1)   # (B, width)
        return self.fc(x)    # (B, latent_dim)


# ── Trunk Net ─────────────────────────────────────────────────

class TrunkNet(nn.Module):
    """
    Trunk net：编码查询坐标 (t, x) → basis vectors
    使用 Fourier 特征嵌入，对捕捉 shock 高频结构至关重要。
    """
    def __init__(self, latent_dim=128, width=256, depth=4,
                 ff_dim=64, activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        # 随机 Fourier 特征矩阵（固定不训练）
        self.register_buffer('B_mat', torch.randn(2, ff_dim) * 10.0)
        in_dim = 2 * ff_dim   # sin + cos
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers.append(nn.Linear(width, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        # coords: (B, N_q, 2)  归一化到 [0,1]
        proj = coords @ self.B_mat             # (B, N_q, ff_dim)
        ff   = torch.cat([torch.sin(proj),
                           torch.cos(proj)], dim=-1)
        return self.net(ff)                    # (B, N_q, latent_dim)


# ── DeepONet 主模型 ───────────────────────────────────────────

class DeepONet(nn.Module):
    """
    DeepONet 范式A：直接算子映射，无自回归误差累积。

    forward(u0, coords) → u(x,t) at query points
      u0:    (B, T_in, X)       初始10步
      coords:(B, N_q, 2)        查询时空坐标，归一化到 [0,1]
      return:(B, N_q)           预测值

    predict_full(u0) → (B, 190, 256)
      直接预测所有190步×256空间点，分批处理防 OOM
    """
    def __init__(
        self,
        T_in:         int = 10,
        X:            int = 256,
        latent_dim:   int = 128,
        branch_type:  str = "cnn",    # "cnn" | "mlp" | "fno_encoder"
        branch_depth: int = 4,
        branch_width: int = 128,
        trunk_depth:  int = 4,
        trunk_width:  int = 256,
        activation:   str = "gelu",
        ff_dim:       int = 64,
    ):
        super().__init__()
        self.T_in       = T_in
        self.latent_dim = latent_dim

        # Branch net
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

        # Trunk net
        self.trunk = TrunkNet(
            latent_dim=latent_dim, width=trunk_width, depth=trunk_depth,
            ff_dim=ff_dim, activation=activation)

        # 可学习偏置（DeepONet 原论文）
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0: torch.Tensor,
                coords: torch.Tensor) -> torch.Tensor:
        b = self.branch(u0)      # (B, latent_dim)
        t = self.trunk(coords)   # (B, N_q, latent_dim)
        return (b.unsqueeze(1) * t).sum(dim=-1) + self.bias  # (B, N_q)

    def predict_full(self, u0: torch.Tensor,
                     t_steps: int = 190,
                     x_points: int = 256) -> np.ndarray:
        """
        推理接口：预测完整的 (t_steps, x_points) 网格。
        返回: numpy (B, t_steps, x_points)
        分批处理查询点防 OOM（每批50个时间步）。
        """
        device = next(self.parameters()).device
        B      = u0.shape[0]

        t_norm = torch.linspace(0, 1, t_steps,  device=device)
        x_norm = torch.linspace(0, 1, x_points, device=device)
        tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
        coords_all = torch.stack([tt.flatten(), xx.flatten()], dim=-1)
        coords_all = coords_all.unsqueeze(0).expand(B, -1, -1)

        total    = t_steps * x_points
        batch_q  = x_points * 50
        pred_flat = torch.zeros(B, total, device=device)

        with torch.no_grad():
            for start in range(0, total, batch_q):
                end = min(start + batch_q, total)
                pred_flat[:, start:end] = self.forward(
                    u0, coords_all[:, start:end, :])

        return pred_flat.reshape(B, t_steps, x_points).cpu().numpy()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import tempfile, os
    print("=" * 50)
    print("model.py unittest")
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

    # Test 1: 三种 branch_type forward shape
    print("[1] forward shape — 三种 branch_type")
    for bt in ["cnn", "mlp", "fno_encoder"]:
        def t1(bt=bt):
            m   = DeepONet(branch_type=bt).to(device)
            u0  = torch.randn(B, 10, 256, device=device)
            c   = torch.rand(B, N_q, 2, device=device)
            out = m(u0, c)
            assert out.shape == (B, N_q), f"shape={out.shape}"
        check(f"branch_type={bt}", t1)

    # Test 2: 输出无 NaN/Inf
    print("\n[2] 输出数值合理性")
    for bt in ["cnn", "mlp", "fno_encoder"]:
        def t2(bt=bt):
            m   = DeepONet(branch_type=bt).to(device)
            u0  = torch.randn(B, 10, 256, device=device)
            c   = torch.rand(B, N_q, 2, device=device)
            out = m(u0, c)
            assert not torch.isnan(out).any(), "输出含 NaN"
            assert not torch.isinf(out).any(), "输出含 Inf"
        check(f"无NaN/Inf branch_type={bt}", t2)

    # Test 3: predict_full shape 和返回类型
    print("\n[3] predict_full shape 和类型")
    def t3():
        m    = DeepONet(branch_type="cnn").to(device)
        u0   = torch.randn(B, 10, 256, device=device)
        pred = m.predict_full(u0)
        assert isinstance(pred, np.ndarray),     f"返回类型: {type(pred)}"
        assert pred.shape == (B, 190, 256),       f"shape={pred.shape}"
        assert pred.dtype == np.float32,          f"dtype={pred.dtype}"
        assert not np.isnan(pred).any(),          "含 NaN"
        assert not np.isinf(pred).any(),          "含 Inf"
    check("predict_full shape/dtype/NaN", t3)

    # Test 4: predict_full 自定义步数
    print("\n[4] predict_full 自定义步数")
    def t4():
        m    = DeepONet(branch_type="cnn").to(device)
        u0   = torch.randn(B, 10, 256, device=device)
        pred = m.predict_full(u0, t_steps=50, x_points=128)
        assert pred.shape == (B, 50, 128), f"shape={pred.shape}"
    check("predict_full t_steps=50 x_points=128", t4)

    # Test 5: checkpoint 保存和加载，输出一致
    print("\n[5] checkpoint 保存和加载")
    def t5():
        m   = DeepONet(branch_type="cnn").to(device)
        u0  = torch.randn(B, 10, 256, device=device)
        c   = torch.rand(B, N_q, 2, device=device)
        m.eval()
        with torch.no_grad():
            out1 = m(u0, c).cpu().numpy()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            torch.save({
                "epoch":            1,
                "model_state_dict": m.state_dict(),
                "val_loss":         0.05,
            }, ckpt_path)

            m2 = DeepONet(branch_type="cnn").to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            m2.load_state_dict(ckpt["model_state_dict"])
            m2.eval()
            with torch.no_grad():
                out2 = m2(u0, c).cpu().numpy()

            assert ckpt["val_loss"] == 0.05,       "val_loss 不一致"
            assert np.allclose(out1, out2, atol=1e-5), "加载后输出不一致"
        finally:
            os.unlink(ckpt_path)
    check("checkpoint 保存/加载/一致性", t5)

    # Test 6: count_params 参数量合理
    print("\n[6] count_params 参数量")
    def t6():
        params = {}
        for bt in ["mlp", "cnn", "fno_encoder"]:
            m = DeepONet(branch_type=bt)
            p = count_params(m)
            params[bt] = p
            assert p > 0, f"{bt} 参数量为0"
            assert p < 100_000_000, f"{bt} 参数量异常大: {p:,}"
            print(f"       {bt}: {p:,} params")
    check("参数量合理", t6)

    # Test 7: 错误 branch_type 抛出 ValueError
    print("\n[7] 非法 branch_type 抛出 ValueError")
    def t7():
        try:
            DeepONet(branch_type="invalid")
            raise AssertionError("应该抛出 ValueError 但没有")
        except ValueError:
            pass
    check("非法 branch_type → ValueError", t7)

    print("\n" + "=" * 50)
    if errors:
        print(f"❌ {len(errors)} 个测试失败:")
        for name, err in errors:
            print(f"   {name}: {err}")
    else:
        print("✅ 所有测试通过")
    print("=" * 50)