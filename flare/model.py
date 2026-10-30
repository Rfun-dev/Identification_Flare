"""FlareFormer: CNN ringan + Transformer spatio-temporal untuk prediksi flare.

Alur data (B = batch, T = jumlah frame):
    magnetogram [B, T, 1, 224, 224]
      -> ResNet18 per frame        : peta fitur [B*T, 512, 7, 7]  (pola lokal: kutub, garis netral)
      -> proyeksi 1x1              : 49 token per frame, masing-masing vektor d_model
      -> + posisi ruang + posisi waktu, tambah token [CLS]
      -> Transformer encoder       : setiap token "melihat" semua token lain di semua frame
      -> token [CLS] -> Linear     : 1 logit = skor flare
"""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from flare.config import Config


class FlareFormer(nn.Module):
    def __init__(self, n_frames=4, img_size=224, d_model=128, n_heads=4, n_layers=2,
                 dropout=0.1, pretrained=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # Magnetogram hanya 1 channel: jumlahkan bobot filter RGB pretrained menjadi 1 channel.
        rgb_weight = backbone.conv1.weight.data
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = rgb_weight.sum(dim=1, keepdim=True)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # buang avgpool & fc

        n_patches = (img_size // 32) ** 2                          # 224/32 = 7 -> 7x7 = 49 token
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_space = nn.Parameter(torch.randn(1, 1, n_patches, d_model) * 0.02)   # "di mana"
        self.pos_time = nn.Parameter(torch.randn(1, n_frames, 1, d_model) * 0.02)     # "kapan"

        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        feat = self.proj(self.cnn(x.flatten(0, 1)))            # [B*T, d, 7, 7]
        tokens = feat.flatten(2).transpose(1, 2)                # [B*T, 49, d]
        tokens = tokens.view(B, T, -1, tokens.shape[-1])        # [B, T, 49, d]
        tokens = tokens + self.pos_space + self.pos_time[:, :T]
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens.flatten(1, 2)], dim=1)
        out = self.transformer(tokens)
        return self.head(self.norm(out[:, 0])).squeeze(1)       # [B] logit


def build_model(cfg, pretrained=None) -> FlareFormer:
    return FlareFormer(n_frames=cfg.n_frames, d_model=cfg.d_model, n_heads=cfg.n_heads,
                       n_layers=cfg.n_layers, dropout=cfg.dropout,
                       pretrained=cfg.pretrained if pretrained is None else pretrained)


def load_checkpoint(path, device):
    """Muat model terbaik beserta config & threshold yang dipilih saat validasi.
    Kembali: (model siap prediksi, config, isi checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = Config(**ckpt["config"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt
