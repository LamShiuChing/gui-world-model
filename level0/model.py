"""Shared parts: EYES (frozen DINOv2-S) and the IMAGINER (screen latent + click -> next screen latent)."""
import torch, torch.nn as nn

dev = "cuda"
c = (torch.arange(16, device=dev) + 0.5) / 16
cy, cx = torch.meshgrid(c, c, indexing="ij")
CENTERS = torch.stack([cx.flatten(), cy.flatten()], 1)  # 256,2 patch centres (x,y) in [0,1]


def heat(a):  # Gaussian click map over patches, sigma = 1 patch
    return torch.exp(-((CENTERS[None] - a[:, None]) ** 2).sum(-1) / (2 / 16**2))


def seg_heat(s, sigma=1.0):  # s: (B,4) segment (x1,y1,x2,y2) in [0,1] -> (B,256) closeness of each patch to the line
    p, a, b = CENTERS[None], s[:, None, :2], s[:, None, 2:]
    ab = b - a
    t = (((p - a) * ab).sum(-1, keepdim=True) / (ab.pow(2).sum(-1, keepdim=True) + 1e-9)).clamp(0, 1)
    return torch.exp(-(p - (a + t * ab)).pow(2).sum(-1) / (2 * (sigma / 16) ** 2))


def fourier(a, n=8):
    f = a[..., None] * (2 ** torch.arange(n, device=a.device)) * torch.pi
    return torch.cat([f.sin(), f.cos()], -1).flatten(1)  # B,4n


class Imaginer(nn.Module):
    def __init__(self, d=384, w=256, layers=4, keys=0, tokens=256, history=False, segments=False):  # tokens: 336px -> 576
        super().__init__()
        self.seg = nn.Linear(64, w) if segments else None  # drag strokes: (x1,y1,x2,y2); a click is a 0-length stroke
        self.key = nn.Embedding(keys, w) if keys else None  # discrete actions (key presses)
        self.prev = nn.Linear(d, w) if history else None  # last step's change (z_t - z_{t-1}): reveals hidden timers
        if history:
            nn.init.zeros_(self.prev.weight), nn.init.zeros_(self.prev.bias)
        self.inp, self.pos = nn.Linear(d, w), nn.Parameter(torch.zeros(1, tokens, w))
        self.act, self.click = nn.Linear(32, w), nn.Parameter(torch.randn(w) * 0.02)
        block = nn.TransformerEncoderLayer(w, 8, 4 * w, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(block, layers)
        self.delta, self.mask = nn.Linear(w, d), nn.Linear(w, 1)
        nn.init.zeros_(self.delta.weight), nn.init.zeros_(self.delta.bias)  # starts as "nothing changes"

    def features(self, z, a=None, k=None, z_prev=None, s=None):  # a: click (x,y) in [0,1], k: key id, s: segment (x1,y1,x2,y2)
        h = self.inp(z) + self.pos
        if s is not None:
            h = h + self.seg(fourier(s))[:, None] + seg_heat(s)[..., None] * self.click
        if z_prev is not None:
            h = h + self.prev(z - z_prev)
        if a is not None:
            h = h + self.act(fourier(a))[:, None] + heat(a)[..., None] * self.click
        if k is not None:
            h = h + self.key(k)[:, None]
        return self.tf(h)

    def forward(self, z, a=None, k=None, z_prev=None, s=None):
        h = self.features(z, a, k, z_prev, s)
        return z + self.delta(h), self.mask(h).squeeze(-1)


class MultiImaginer(Imaginer):
    """IMAGINER with K possible futures. Shared change (delta) + a per-future change from a small MLP that sees
    (patch features + future code), so each code can put e.g. the new food in a different place.
    prior(h) = how likely each future is. Trained winner-takes-all: only the closest future is corrected."""

    def __init__(self, K=64, d=384, w=256, concat=False, **kw):  # concat: code joined to (not added to) the features
        super().__init__(d=d, w=w, **kw)
        self.K, self.code, self.concat = K, nn.Embedding(K, w), concat
        self.head = nn.Sequential(nn.LayerNorm((2 if concat else 1) * w), nn.Linear((2 if concat else 1) * w, 2 * w), nn.GELU(), nn.Linear(2 * w, d))
        nn.init.zeros_(self.head[-1].weight), nn.init.zeros_(self.head[-1].bias)
        self.prior = nn.Linear(w, K)

    def forward(self, z, a=None, k=None):  # -> all K futures (B,K,256,d), mask logits (B,256), prior logits (B,K)
        h = self.features(z, a, k)
        futures = (z + self.delta(h))[:, None] + self.head(self.mix(h[:, None], self.code.weight[None, :, None]))
        return futures, self.mask(h).squeeze(-1), self.prior(h.mean(1))


    def mix(self, h, c):  # h (...,T,w), c (...,1,w) broadcastable -> head input
        if not self.concat:
            return h + c
        h, c = torch.broadcast_tensors(h, c)
        return torch.cat([h, c], -1)


class Inverse(nn.Module):
    """INVERSE: (screen before, screen after) -> which action happened.
    Clicks: one logit per patch (click = that patch centre). Keys: one logit per key id."""

    def __init__(self, d=384, w=256, layers=4, keys=0):
        super().__init__()
        self.inp, self.pos = nn.Linear(2 * d, w), nn.Parameter(torch.zeros(1, 256, w))
        block = nn.TransformerEncoderLayer(w, 8, 4 * w, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(block, layers)
        self.click = nn.Linear(w, 1)
        self.key = nn.Linear(w, keys) if keys else None

    def forward(self, z, z1):  # -> click logits (B,256), key logits (B,keys) or None
        h = self.tf(self.inp(torch.cat([z, z1 - z], -1)) + self.pos)
        return self.click(h).squeeze(-1), self.key(h.mean(1)) if self.key is not None else None


class Eyes:  # same preprocessing as encode.py: fp32, ImageNet mean/std, drop CLS -> 256 patch tokens
    def __init__(self):
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs):  # imgs: B,224,224,3 uint8 numpy -> B,256,384
        x = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255
        return self.enc(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 1:].half().float()
