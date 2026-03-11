import torch, torch.nn as nn
from torch import Tensor
from jaxtyping import Float

class ConvBlock(nn.Sequential):
    def __init__(self, ni, nf, ks=3, do_act=True, p=0.0, groups=1):
        activation = nn.ReLU if do_act else nn.Identity
        super().__init__(
            nn.Conv1d(
                ni, nf, ks, padding=ks//2, bias=False, groups=groups
            ), nn.BatchNorm1d(nf), activation(), nn.Dropout(p)
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
        self.acc_stem = self._init_stem(3, d_model)
        self.lin_acc_stem = self._init_stem(3, d_model)
        self.rel_rot_stem = self._init_stem(3, d_model)
        self.proj = ConvBlock(3*d_model, d_model, 1)
        self.encoder = nn.Sequential(
            *[ResBlock(d_model, d_model, p=p) for i in range(num_layers)],
        )
        self.head = nn.Sequential(AttentionPooling(d_model), nn.Linear(d_model, n_classes))
        self._init_weights()

    def _init_stem(self, in_ch: int, d_model: int):
        """Performs depthwise convolution in the first layer"""
        K = d_model//in_ch   # depthwise multiplier
        stem = nn.Sequential(
            nn.BatchNorm1d(in_ch), ConvBlock(in_ch, K*in_ch, groups=in_ch),
            ConvBlock(K*in_ch, d_model, ks=1), nn.MaxPool1d(2)
        )
        return stem

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
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
        self, accs: Float[Tensor, "bs 3 L"], lin_accs: Float[Tensor, "bs 3 L"],
        rel_rots: Float[Tensor, "bs 3 L"]
    ) -> Float[Tensor, "bs n_classes"]:
        acc_fs = self.acc_stem(accs)
        lin_acc_fs = self.lin_acc_stem(lin_accs)
        rel_rot_fs = self.rel_rot_stem(rel_rots)

        fs = torch.cat([acc_fs, lin_acc_fs, rel_rot_fs], dim=1)
        fs = self.encoder(self.proj(fs))
        logits = self.head(fs)
        return logits