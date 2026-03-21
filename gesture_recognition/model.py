import torch, torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool
from einops import rearrange, reduce

class ConvBlock(nn.Sequential):
    def __init__(self, ni, nf, ks=3, do_act=True, p=0.0, groups=1, is_2d=False):
        conv, batch_norm = (
            (nn.Conv2d, nn.BatchNorm2d) if is_2d else (nn.Conv1d, nn.BatchNorm1d)
        )
        activation = nn.ReLU if do_act else nn.Identity
        super().__init__(
            conv(ni, nf, ks, padding=ks//2, bias=False, groups=groups),
            batch_norm(nf), activation(), nn.Dropout(p)
        )

class ResBlock(nn.Module):
    def __init__(self, ni: int, nf: int, p=0.0):
        super().__init__()
        self.cnns = nn.Sequential(
            ConvBlock(ni, nf, do_act=True, p=0), ConvBlock(nf, nf, do_act=False, p=p)
        )
        self.skip = nn.Identity() if ni==nf else nn.Conv1d(ni, nf, 1, bias=False)

    def forward(self, x):
        return nn.functional.relu(self.cnns(x) + self.skip(x))
    
class TemporalStem(nn.Sequential):
    """Performs depthwise convolution in the first layer"""
    def __init__(self, in_ch, d_model):
        K = d_model//in_ch   # depthwise multiplier
        super().__init__(
            nn.BatchNorm1d(in_ch), ConvBlock(in_ch, K*in_ch, groups=in_ch),
            ResBlock(K*in_ch, d_model), nn.MaxPool1d(2)
        )
    
class ToFStem(nn.Module):
    def __init__(self, d_model: int=32):
        super().__init__()
        K = d_model//5
        self.spatial_stem = nn.Sequential(
            ConvBlock(5, K*5, is_2d=True, groups=5), nn.MaxPool2d(2),
            ConvBlock(K*5, d_model, is_2d=True), nn.MaxPool2d(2),
            ConvBlock(d_model, d_model, is_2d=True), nn.MaxPool2d(2),
        )
        self.temporal_stem = TemporalStem(d_model, d_model)

    def forward(self, tofs: Float[Tensor, "bs L 5 8 8"]):
        bs = tofs.size(0)
        tofs = rearrange(tofs, "bs L n h w -> (bs L) n h w")
        fs: Float[Tensor, "(bs L) d_model 1 1"] = self.spatial_stem(tofs)
        fs = rearrange(fs, "(bs L) d_model 1 1 -> bs d_model L", bs=bs)
        fs = self.temporal_stem(fs)
        return fs
    
class SensorFusion(nn.Module):
    def __init__(self, d_model: int, d_prox: int):
        super().__init__()
        self.imu_proj = ConvBlock(3*d_model, d_model, ks=1)
        self.thm_proj = ConvBlock(d_prox, d_model, ks=1)
        self.tof_proj = ConvBlock(d_prox, d_model, ks=1)
        self.thm_gating = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Sigmoid()
        )
        self.tof_gating = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Sigmoid()
        )
    
    def forward(
        self, imu_fs: Float[Tensor, "bs 3*d L"], thm_fs: Float[Tensor, "bs d_prox L"],
        tof_fs: Float[Tensor, "bs d_prox L"], proximity_mask: Bool[Tensor, "bs"]
    ) -> Float[Tensor, "bs d_model L"]:
        imu_fs = self.imu_proj(imu_fs)
        thm_fs = self.thm_proj(thm_fs)
        tof_fs = self.tof_proj(tof_fs)

        imu_pooled, thm_pooled, tof_pooled = [
            reduce(x, "bs d L -> bs d", "mean") for x in (imu_fs, thm_fs, tof_fs)
        ]
        thm_gate_in = torch.cat([imu_pooled, thm_pooled], dim=1)
        tof_gate_in = torch.cat([imu_pooled, tof_pooled], dim=1)

        thm_gate: Float[Tensor, "bs d"] = self.thm_gating(thm_gate_in)
        thm_gate = (proximity_mask[...,None] * thm_gate)
        tof_gate: Float[Tensor, "bs d"] = self.tof_gating(tof_gate_in)
        tof_gate = (proximity_mask[...,None] * tof_gate)

        fused_fs = imu_fs + thm_gate[...,None]*thm_fs + tof_gate[...,None]*tof_fs
        return fused_fs
    
class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Conv1d(d_model, 1, 1)

    def forward(self, x: Float[Tensor, "bs d_model L"]) -> Float[Tensor, "bs d_model"]:
        wts: Float[Tensor, "bs 1 L"] = torch.softmax(self.score(x), dim=-1)
        pooled: Float[Tensor, "bs d_model"] = (x*wts).sum(-1)
        return pooled

class Model(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_classes: int, p=0.0):
        super().__init__()
        self.imu_stems = nn.ModuleList(TemporalStem(3, d_model) for _ in range(3))
        self.thm_stem = TemporalStem(5, d_model//2)
        self.tof_stem = ToFStem(d_model//2)
        self.fusion = SensorFusion(d_model, d_model//2)
        self.encoder = nn.Sequential(
            *[ResBlock(d_model, d_model, p=p) for _ in range(num_layers)],
        )
        self.head = nn.Sequential(AttentionPooling(d_model), nn.Linear(d_model, n_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        # Zero-init last BN in each residual branch (ResNet-style)
        for block in self.encoder:
            if isinstance(block, ResBlock):
                # block.cnns[1] is the last ConvBlock in residual branch
                # ConvBlock is Sequential: [Conv1d, BatchNorm1d, Act, Dropout]
                last_bn = block.cnns[1][1]
                if isinstance(last_bn, nn.BatchNorm1d):
                    nn.init.zeros_(last_bn.weight)

    def forward(
        self, *imus: Float[Tensor, "bs 3 L"], thms: Float[Tensor, "bs 5 L"],
        tofs: Float[Tensor, "bs L 5 8 8"], proximity_mask: Bool[Tensor, "bs"]
    ) -> Float[Tensor, "bs n_classes"]:
        imu_f_li = [stem(x) for x, stem in zip(imus, self.imu_stems)]
        imu_fs: Float[Tensor, "bs d_model*3 L"] = torch.cat(imu_f_li, dim=1)
        thm_fs: Float[Tensor, "bs d_model//2 L"] = self.thm_stem(thms)
        tof_fs: Float[Tensor, "bs d_model//2 L"] = self.tof_stem(tofs)
        fused_fs = self.fusion(imu_fs, thm_fs, tof_fs, proximity_mask)
        logits = self.head(self.encoder(fused_fs))
        return logits