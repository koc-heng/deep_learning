import os
import pandas as pd

def clean_title(title: str):
    title = title.strip().lower()
    if title.startswith("re:") or title.startswith("fw:"):
        return None
    return title


data_folder = "data"
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
print("find CSV data:", csv_files)

dfs = []
for csv_file in csv_files:
    csv_path = os.path.join(data_folder, csv_file)
    print(f"[read] {csv_path}")
    df = pd.read_csv(csv_path)
    df["title"] = df["title"].apply(lambda t: clean_title(t) if isinstance(t, str) else None)
    df = df.dropna(subset=["title"])
    dfs.append(df)

if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv("cleaned_data.csv", index=False, encoding="utf-8")
    print(f"[ok] cleaned_data.csv, total {len(merged_df)} count data.")


