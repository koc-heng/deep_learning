import torch
import csv
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

# check gpu 能不能用
print("CUDA available：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name：", torch.cuda.get_device_name(0))
    
# 初始化 CKIP 的分詞器與詞性標註器
ws_driver = CkipWordSegmenter(model="bert-base", device=-1)
pos_driver = CkipPosTagger(model="bert-base", device=-1)

# 定義標點符號
EXCLUDE_TOKENS = {"[", "]", "(", ")", "！", "?", "？", "》", "｜", "「", "」", "《", "、", "。", "，", "；", "：", "…", "～", "—", "─", "——", "-", "_", "+", "*", "&", "^", "%", "$", "#", "@", "!", "~", "／",}

input_csv = 'test_data.csv'
output_csv = 'test_output.csv'
###-----------------------------###
#自由切換
#input_csv = 'cleaned_data.csv'
#output_csv = 'tokenized_output.csv'

with open(input_csv, 'r', encoding='utf-8-sig') as infile, \
     open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = ['board', 'tokenized_title']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    rows = list(reader)
    texts = [row['title'] for row in rows]
    
    # 做批次比較快
    batch_size = 1000
    total_texts = len(texts)
    
    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_rows = rows[i:i+batch_size]
        
        # 用 CKIP 處理斷詞與詞性標註
        ws_results = ws_driver(batch_texts, use_delim=False)
        pos_results = pos_driver(ws_results, use_delim=False)
        
        for row, tokens, pos_tags in zip(batch_rows, ws_results, pos_results):
            filtered_tokens = []
            for token, tag in zip(tokens, pos_tags):
                if token.strip() == "":
                    continue
                if token in EXCLUDE_TOKENS:
                    continue
                filtered_tokens.append(token)
            tokenized_title = " ".join(filtered_tokens)
            writer.writerow({
                'board': row['board'],
                'tokenized_title': tokenized_title
            })
        
        del ws_results, pos_results, batch_texts, batch_rows
        print(f"ok Processed {min(i+batch_size, total_texts)}/{total_texts} data. :D")