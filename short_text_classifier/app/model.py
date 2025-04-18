import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from .model_tool import segment_ckip, get_mix_vec, index_label, label_chinese, mlp_path

device = "cuda" if torch.cuda.is_available() else "cpu"

multi_class = len(index_label)

# ---------- 定義模型 ----------
class MLP(nn.Module):
    def __init__(self, in_dim=100, hid_dim=64, out_dim=multi_class):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

# ---------- 載入模型 ----------
MODEL = MLP().to(device)
MODEL.load_state_dict(torch.load(mlp_path, map_location=device))
MODEL.eval()

# ---------- 預測 ----------
import torch.nn.functional as F  

@torch.no_grad()
def predict(title: str) -> str:
    print("接收到的 title：", title)
    tokens = segment_ckip(title)
    print("tokens：", tokens)

    vec = get_mix_vec(tokens)
    #print("vector 前5維：", vec[:5])
    x = torch.from_numpy(np.array([vec])).float().to(device) 
    logits = MODEL(x) 

    probs = F.softmax(logits, dim=1)
    probs_np = probs.cpu().numpy()[0]

    print("每類機率分數：")
    for i, p in enumerate(probs_np):
        print(f"  {i} - {index_label[i]:<12}: {p:.4f}")

    idx = probs.argmax(1).item()
    confidence = probs[0, idx].item()

    print("最終預測結果：", index_label[idx])
    print("預測信心值：", confidence)

    return {
    "label": index_label[idx],
    "label_chinese": label_chinese[index_label[idx]],
    "confidence": round(confidence, 4)  
}



