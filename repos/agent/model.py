import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(
            x_ft[:, :, :self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator.

    输入: (B, in_channels, X)
    输出: (B, out_channels, X)

    参数:
        modes      : 保留的Fourier模数（默认12，可调）
        width      : 隐层宽度（默认64，比官方baseline的20更宽）
        in_channels: 输入通道数（默认11 = 10个时间步历史 + 1个空间坐标）
        out_channels: 输出通道数（默认1，预测下一步）
        n_layers   : FNO层数（默认4）
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 64,
        in_channels: int = 11,
        out_channels: int = 1,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # 输入提升层
        self.fc0 = nn.Linear(in_channels, width)

        # FNO层
        self.convs = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])

        # 输出投影层
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, in_channels, X)
        x = x.permute(0, 2, 1)       # (B, X, in_channels)
        x = self.fc0(x)               # (B, X, width)
        x = x.permute(0, 2, 1)       # (B, width, X)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 1)       # (B, X, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)               # (B, X, out_channels)
        x = x.permute(0, 2, 1)       # (B, out_channels, X)
        return x


def load_official_checkpoint(ckpt_path: str, device='cpu') -> FNO1d:
    """
    加载官方PDEBench FNO checkpoint。
    官方key: conv0/w0 -> 我们的key: convs.0/ws.0，需要重映射。
    """
    import re
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    new_sd = {}
    for k, v in state_dict.items():
        k2 = re.sub(r'^conv(\d+)\.', lambda m: f'convs.{m.group(1)}.', k)
        k2 = re.sub(r'^w(\d+)\.', lambda m: f'ws.{m.group(1)}.', k2)
        new_sd[k2] = v

    out_channels = new_sd['fc2.weight'].shape[0]
    model = FNO1d(modes=12, width=20, in_channels=11,
                  out_channels=out_channels, n_layers=4)
    model.load_state_dict(new_sd, strict=True)
    model.to(device)
    return model
