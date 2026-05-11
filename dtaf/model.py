import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DTAFConfig:
    seq_len: int = 96
    pred_len: int = 96
    enc_in: int = 1
    d_model: int = 32
    e_layers: int = 1
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    heads: int = 2
    k: int = 1
    moving_avg: int = 25
    aggregated_norm: int = 1
    expert_num: int = 2
    kan_div: int = 4


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class LinearExtractor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.d_model
        self.pred_len = configs.d_model
        self.decomposition = SeriesDecomp(configs.moving_avg)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )
        self.linear_trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        return self.linear_seasonal(seasonal_init) + self.linear_trend(trend_init)


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order],
                noise,
            )
            scale = self.scale_spline if not self.enable_standalone_scale_spline else 1.0
            self.spline_weight.data.copy_(scale * coeff)
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x):
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError("KANLinear expects shape [batch, in_features].")

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (
                grid[:, k:-1] - grid[:, : -(k + 1)]
            )
            right = (grid[:, k + 1 :] - x) / (
                grid[:, k + 1 :] - grid[:, 1:-k]
            )
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x, y):
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError("Spline x must have shape [batch, in_features].")
        if y.size() != (x.size(0), self.in_features, self.out_features):
            raise ValueError("Spline y has an unexpected shape.")

        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x):
        if x.size(-1) != self.in_features:
            raise ValueError("KANLinear received an unexpected feature dimension.")
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return (base_output + spline_output).reshape(
            *original_shape[:-1], self.out_features
        )


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
                for in_features, out_features in zip(layers_hidden, layers_hidden[1:])
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_dim, div):
        super().__init__()
        hidden_dim = max(1, input_dim // max(1, div))
        self.network = KAN(layers_hidden=[input_dim, hidden_dim, input_dim])

    def forward(self, x):
        return self.network(x)


class MOE(nn.Module):
    def __init__(self, expert_num, input_dim, div):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim=input_dim, div=div) for _ in range(expert_num)]
        )
        self.router = KANLinear(input_dim, expert_num)

    def forward(self, x):
        router = self.router(x).softmax(-1)
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.einsum("bpn,bpnd->bpd", router, experts_out)


class TFS(nn.Module):
    def __init__(self, input_dim, configs, patch_num):
        super().__init__()
        self.configs = configs
        self.mlp = nn.Linear(input_dim, input_dim)
        self.extractor_his = LinearExtractor(configs)
        self.weight_linear = nn.Linear(input_dim, patch_num)
        self.dropout = nn.Dropout(configs.dropout)
        self.extractor_cur = LinearExtractor(configs)
        self.gate = nn.Linear(input_dim, 1)
        self.norm = nn.LayerNorm(input_dim) if configs.aggregated_norm == 1 else None
        self.moe = (
            MOE(expert_num=configs.expert_num, input_dim=input_dim, div=configs.kan_div)
            if configs.expert_num > 0
            else None
        )

    def forward(self, x):
        origin = x
        if self.moe is not None:
            x = x - self.moe(x)

        history = self.extractor_his(x)
        current_weight = self.gate(self.extractor_cur(origin)).repeat(
            1, 1, origin.shape[-1]
        )
        weight = self.weight_linear(history).softmax(dim=-1)
        aggregated = torch.matmul(torch.tril(weight, diagonal=0), x)

        history_out = self.dropout(self.mlp(aggregated))
        current_out = self.dropout(current_weight) * x
        out = history_out + current_out
        if self.norm is not None:
            out = self.norm(out)
        return out, x


class Attention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_output))


class Predictor(nn.Module):
    def __init__(self, nf, target_window, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class DTAF(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.config = configs
        if self.config.e_layers < 1:
            raise ValueError("DTAF requires at least one TFS layer.")
        if self.config.seq_len < self.config.patch_len:
            raise ValueError("seq_len must be greater than or equal to patch_len.")
        if self.config.d_model % self.config.heads != 0:
            raise ValueError("d_model must be divisible by heads.")

        self.patch_num = int(
            (self.config.seq_len - self.config.patch_len) / self.config.stride + 2
        )
        self.tfss = nn.ModuleList(
            [TFS(self.config.d_model, self.config, self.patch_num) for _ in range(self.config.e_layers)]
        )
        self.predictor = Predictor(
            2 * self.config.d_model * self.patch_num,
            self.config.pred_len,
            self.config.dropout,
        )
        self.patch_embedding = PatchEmbedding(
            self.config.d_model,
            self.config.patch_len,
            self.config.stride,
            self.config.stride,
            self.config.dropout,
        )
        self.temporal_attention = Attention(
            self.config.d_model, self.config.heads, self.config.dropout
        )
        self.frequency_attention = Attention(
            self.config.d_model, self.config.heads, self.config.dropout
        )
        self.drop = nn.Dropout(self.config.dropout)
        self.norm = nn.LayerNorm(self.config.d_model)

    @staticmethod
    def _normalize(x):
        means = x.mean(1, keepdim=True)
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return x / stdev, means, stdev

    def forward(self, x_enc):
        batch_size, _, n_vars = x_enc.size()
        x_enc, means, stdev = self._normalize(x_enc)

        enc_out, _ = self.patch_embedding(x_enc.transpose(1, 2))
        enc_out_tfs = enc_out
        stables = None
        for tfs in self.tfss:
            aggregated, stables = tfs(enc_out_tfs)
            enc_out_tfs = self.norm(self.drop(aggregated) + enc_out_tfs)

        h_t = enc_out_tfs
        freq = torch.fft.rfft(enc_out_tfs, dim=-1)
        wave = torch.zeros(
            enc_out_tfs.shape[0],
            enc_out_tfs.shape[1],
            freq.shape[-1],
            device=enc_out_tfs.device,
            dtype=enc_out_tfs.dtype,
        )
        wave[:, 1:, :] = torch.exp(
            torch.abs(freq[:, 1:, :]) - torch.abs(freq[:, :-1, :])
        )

        top_k = min(max(1, self.config.k), wave.shape[-1])
        _, topk_indices = torch.topk(wave, top_k, dim=-1)
        mask = torch.zeros_like(freq, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)

        filtered_freq = torch.where(mask, freq, torch.zeros_like(freq))
        h_f = torch.fft.irfft(filtered_freq, n=enc_out_tfs.size(-1), dim=-1)
        h_f[:, 0, :] = enc_out_tfs[:, 0, :]

        h_f = self.frequency_attention(h_f)
        h_t = self.temporal_attention(h_t)
        enc_out = torch.cat([h_t, h_f], dim=-2)

        enc_out = torch.reshape(
            enc_out, (batch_size, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)
        out = self.predictor(enc_out).permute(0, 2, 1)
        out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.config.pred_len, 1)
        out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.config.pred_len, 1)
        return out, stables
