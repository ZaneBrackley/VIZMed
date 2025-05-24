import os
import pandas as pd
from collections import Counter

def build_cui_vocab(df: pd.DataFrame, source: str, umls_lookup: dict = None, output_dir: str = "VIZMed/data/vocab"):
    """
    Build a CUI vocabulary CSV from a dataset DataFrame.
    Filters out empty or malformed CUI entries.
    """

    os.makedirs(output_dir, exist_ok=True)

    valid_cuis = []
    for row in df["concepts"]:
        if isinstance(row, list):
            valid_cuis.extend([cui for cui in row if isinstance(cui, str) and cui.strip()])

    unique_cuis = sorted(set(valid_cuis))
    freq = Counter(valid_cuis)

    vocab_df = pd.DataFrame({
        "index": range(len(unique_cuis)),
        "cui": unique_cuis,
        "name": [umls_lookup.get(cui, "") if umls_lookup else "" for cui in unique_cuis],
        "count": [freq[cui] for cui in unique_cuis],
        "source": source
    })

    out_path = os.path.join(output_dir, f"{source}_vocab.csv")
    vocab_df.to_csv(out_path, index=False)
    print(f"[✓] Saved CUI vocab to {out_path}")

    return vocab_df