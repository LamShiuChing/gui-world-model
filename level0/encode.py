"""EYES: encode every unique state screenshot once with frozen DINOv2-S (fp32, per report 15)."""
import json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModel

D = Path(__file__).parent.parent / "data" / "level0"
S = len(json.loads((D / "states.json").read_text()))
imgs = np.stack([np.asarray(Image.open(D / "img" / f"{i}.png").convert("RGB")) for i in range(S)])  # S,224,224,3
np.save(D / "images.npy", imgs)

enc = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
out = []
with torch.no_grad():
    for i in range(0, S, 128):
        x = torch.from_numpy(imgs[i:i + 128]).cuda().permute(0, 3, 1, 2).float() / 255
        out.append(enc(pixel_values=(x - mean) / std).last_hidden_state[:, 1:].half().cpu())  # drop CLS -> 256 patch tokens
feats = torch.cat(out).numpy()
np.save(D / "feats.npy", feats)
print("feats", feats.shape, feats.dtype)
