"""
models.py
---------
Teeno models ka structure same hai:

    frames -> CNN (har frame se feature) -> Temporal module -> mean pooling -> embedding -> classifier

Sirf "Temporal module" badalta hai: LSTM / TCN / Transformer.
Isse comparison fair rehta hai.

forward() do cheezein return karta hai:
    logits    : training (classification loss) ke liye
    embedding : evaluation (gallery/probe matching) ke liye
"""
import torch
import torch.nn as nn


class FrameCNN(nn.Module):
    """Ek frame (1x64x64) -> feature vector. Spatial info yahan se aati hai."""

    def __init__(self, out_dim=128):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # 64 -> 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, out_dim))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):  # x: (N, 1, H, W)
        return self.pool(self.features(x)).flatten(1)  # (N, out_dim)


class TemporalLSTM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # bidirectional: output dim = 2 * hidden = dim
        self.lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x):  # (B, T, dim)
        out, _ = self.lstm(x)
        return out


class TCNBlock(nn.Module):
    """Dilated 1D conv + residual connection."""

    def __init__(self, ch, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class TemporalTCN(nn.Module):
    def __init__(self, dim, dilations=(1, 2, 4)):
        super().__init__()
        self.blocks = nn.Sequential(*[TCNBlock(dim, d) for d in dilations])

    def forward(self, x):  # (B, T, dim)
        x = x.transpose(1, 2)  # Conv1d ko (B, dim, T) chahiye
        x = self.blocks(x)
        return x.transpose(1, 2)


class TemporalTransformer(nn.Module):
    def __init__(self, dim, max_len=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))  # learnable positional embedding
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(self, x):  # (B, T, dim)
        T = x.size(1)
        return self.enc(x + self.pos[:, :T])


class GaitNet(nn.Module):
    def __init__(self, temporal="lstm", num_classes=74, feat_dim=128, emb_dim=128, dropout=0.3):
        super().__init__()
        self.cnn = FrameCNN(feat_dim)
        if temporal == "lstm":
            self.temporal = TemporalLSTM(feat_dim)
        elif temporal == "tcn":
            self.temporal = TemporalTCN(feat_dim)
        elif temporal == "transformer":
            self.temporal = TemporalTransformer(feat_dim)
        else:
            raise ValueError("temporal must be lstm / tcn / transformer")
        self.emb = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.BatchNorm1d(emb_dim))
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):  # x: (B, T, 1, H, W)
        B, T, C, H, W = x.shape
        f = self.cnn(x.reshape(B * T, C, H, W)).reshape(B, T, -1)  # (B, T, feat)
        f = self.temporal(f)                                       # (B, T, feat)
        e = self.emb(f.mean(dim=1))                                # (B, emb)
        return self.cls(self.drop(e)), e


if __name__ == "__main__":
    # shape check: python src/models.py
    x = torch.randn(4, 30, 1, 64, 64)
    for name in ["lstm", "tcn", "transformer"]:
        m = GaitNet(name, num_classes=74)
        logits, emb = m(x)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name:12s} logits {tuple(logits.shape)} emb {tuple(emb.shape)} params {n_params:.2f}M")