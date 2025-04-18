import os
import numpy as np
from ckip_transformers.nlp import CkipWordSegmenter
from gensim.models import Doc2Vec

root = os.path.dirname(os.path.dirname(__file__))
d2v_dm_model_path = os.path.join(root, 'model', '_d2v_dm.model')
d2v_dbow_model_path = os.path.join(root, 'model', '_d2v_dbow.model')
mlp_path = os.path.join(root, 'model', '_mlp.pt')

# ---------- 初始化模型 ----------
ws_driver = CkipWordSegmenter(model="bert-base", device=-1)
dm_model   = Doc2Vec.load(d2v_dm_model_path)
dbow_model = Doc2Vec.load(d2v_dbow_model_path)

# ---------- 標籤轉換 ----------
label_index = {
    "Boy-Girl": 0, "LifeMoney": 1, "Military": 2, "Tech_Job": 3,
    "Baseball": 4, "C_Chat": 5, "HatePolitics": 6, "PC_Shopping": 7, "Stock": 8
}
index_label = {v: k for k, v in label_index.items()}

label_chinese = {
    "Boy-Girl": "男女",
    "LifeMoney": "省錢",
    "Military": "軍事",
    "Tech_Job": "科技",
    "Baseball": "棒球",
    "C_Chat": "動漫遊戲",
    "HatePolitics": "政黑",
    "PC_Shopping": "電蝦",
    "Stock": "股票"
}

# ---------- 分詞處理 ----------

exclued_tokens = {"[", "]", "(", ")", "！", "?", "？", "》", "｜", "「", "」",
                  "《", "、", "。", "，", "；", "：", "…", "～", "—", "─", "——",
                  "-", "_", "+", "*", "&", "^", "%", "$", "#", "@", "!", "~", "／"}

def segment_ckip(text: str) -> list[str]:
    ws_result = ws_driver([text], use_delim=False)[0]
    return [tok for tok in ws_result if tok.strip() and tok not in exclued_tokens]

# ---------- 向量轉換 ----------
def get_mix_vec(tokens: list[str]) -> list[float]:
    v1 = dm_model.infer_vector(tokens)
    v2 = dbow_model.infer_vector(tokens)
    return np.concatenate([v1, v2]).astype("float32")
